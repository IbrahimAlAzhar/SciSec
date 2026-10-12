#!/usr/bin/env python3
"""
v8_rollouts.py — sample k completions per agent prompt and score each with v8_rewards.

TWO ROLLOUT TYPES
    --task detect     every agent of the detect cascade, on both documents of a pair:
                        LOCATOR (per window), EVIDENCE, VERIFIER, LEADER
    --task localize   LEADER in repair mode, on documents the greedy cascade flagged, plus
                      a sample of unflagged clean ones so the adapter learns RESTRAINT
                      (deleting nothing) and not only excision

HOW DOWNSTREAM AGENTS ARE CONDITIONED
    At inference EVIDENCE / VERIFIER / LEADER see the GREEDY output of the step before --
    mistakes included. So each downstream prompt here is built from a temperature-0 call of
    the upstream agent, exactly as at inference, and only the agent being sampled runs at
    temperature. (v4 conditioned on the best accepted sample instead; that made training
    prompts cleaner than inference prompts, and the leader never learned to reject a bad
    candidate because it never saw one.)

    One exception, stated plainly: when the greedy locator MISSES the gold span on an
    attacked document, a gold item is added with probability --inject-gold (default 0.5).
    Without it the verifier would only ever see attacks the locator already found, and
    would never learn to CONFIRM the ones it misses. Injected items are marked
    `injected_gold` in every sample they reach.

OUTPUT
    rollouts/<model>/<task>/samples.jsonl   one line per sampled completion
    rollouts/<model>/<task>/stats.json      accept rate, reward, and the share of groups
                                            with reward variance (the only groups GRPO and
                                            DPO can learn from)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Sequence

import pandas as pd

import ft_config as C
import v8_agents as A
import v8_cascade as K
import v8_rewards as RW
import v8_tools as T
from v8_llm import Backend, Routing, build_routing, make_backend


def gid(*parts) -> str:
    return hashlib.sha1("\x1f".join(str(p) for p in parts).encode()).hexdigest()[:16]


class Writer:
    def __init__(self, path: str):
        self.f = open(path, "w", buffering=1, encoding="utf-8")
        self.lock = threading.Lock()
        self.stats = defaultdict(lambda: {"n": 0, "usable": 0, "parse_fail": 0,
                                          "reward_sum": 0.0})
        self.groups = defaultdict(list)

    def put(self, rec: Dict) -> None:
        with self.lock:
            self.f.write(json.dumps(T.scrub(rec), ensure_ascii=False) + "\n")
            s = self.stats[rec["agent"]]
            s["n"] += 1
            s["usable"] += int(rec["score"]["usable"])
            s["parse_fail"] += int(not rec["score"]["parsed"])
            s["reward_sum"] += rec["score"]["reward"]
            if not rec.get("greedy"):
                self.groups[(rec["agent"], rec["group_id"])].append(rec["score"]["reward"])

    def close(self) -> None:
        self.f.close()


def sample(llm: Backend, route: Routing, task: str, agent: str, system: str, user: str,
           cfg: K.CascadeConfig, k: int, temp: float, max_tokens: int, seed: int
           ) -> List[str]:
    msgs = A.messages(system, user, system_role=cfg.system_role)
    try:
        return llm.chat(msgs, model=route.name(task, agent), temperature=temp,
                        max_tokens=max_tokens, n=k, seed=seed)
    except Exception as e:
        print(f"  [gen] {agent}: {type(e).__name__}: {str(e)[:160]}")
        return []


# ======================================================================================
# detect rollouts for one document
# ======================================================================================
def detect_doc(doc: str, twin: str, meta: Dict, gold, label: int, llm, route, cfg, W: Writer,
               k: int, temp: float, rng: random.Random, inject_gold: float, seed: int):
    family = meta["threat_family"]
    base = dict(meta, label=label, task="detect")
    greedy = K.CascadeConfig(**{**cfg.__dict__, "temperature": 0.0})

    def emit(agent, group, system, user, outs, scorer, extra=None, greedy_flag=False):
        for j, raw in enumerate(outs):
            sc = scorer(raw)
            rec = dict(base, agent=agent, group_id=group, k=j, greedy=greedy_flag,
                       system=system, user=user, completion=raw, score=sc.as_dict(),
                       **(extra or {}))
            if agent == "LEADER":
                rec["provenance"] = RW.score_provenance(
                    raw, doc=doc, gold=gold, label=label, twin_doc=twin).as_dict()
            W.put(rec)

    sigs, wins = K.prepare(doc, cfg)
    loc_greedy = []
    for wi, w in enumerate(wins):
        s, u = K.locator_prompt(doc, w, sigs)
        g = gid("detect", "LOCATOR", meta["doc_id"], wi)
        gr = sample(llm, route, "detect", "LOCATOR", s, u, greedy, 1, 0.0,
                    A.ROLES["LOCATOR"].max_tokens, seed)
        loc_greedy.append(gr[0] if gr else "")
        outs = sample(llm, route, "detect", "LOCATOR", s, u, cfg, k, temp,
                      A.ROLES["LOCATOR"].max_tokens, seed)
        emit("LOCATOR", g, s, u, outs, lambda raw, w=w: RW.score_locator(
            raw, doc=doc, window=(w.start, w.end), gold=gold, label=label),
            {"window": [w.start, w.end], "window_why": w.why})
    items, _ = K.assemble_items(doc, wins, loc_greedy, sigs, cfg.max_items)
    injected = False
    if label == 1 and gold and not any(A.item_is_gold(it, gold) for it in items) \
            and rng.random() < inject_gold:
        a, b = T.merge(gold)[0]
        items = items[: cfg.max_items - 1] + [A.Item(0, doc[a:b], a, b, "candidate",
                                                     "directive", "(teacher-forced)")]
        items.sort(key=lambda it: it.start)
        for i, it in enumerate(items, 1):
            it.id = i
        injected = True
    if not items:
        return
    ex = {"items": [it.as_dict() for it in items], "injected_gold": injected}
    retrieved = K.retrieve_for(doc, items, cfg.evidence_k)

    s, u = A.build_evidence(items, retrieved)
    ev_g = sample(llm, route, "detect", "EVIDENCE", s, u, greedy, 1, 0.0,
                  A.ROLES["EVIDENCE"].max_tokens, seed)
    outs = sample(llm, route, "detect", "EVIDENCE", s, u, cfg, k, temp,
                  A.ROLES["EVIDENCE"].max_tokens, seed)
    emit("EVIDENCE", gid("detect", "EVIDENCE", meta["doc_id"]), s, u, outs,
         lambda raw: RW.score_evidence(raw, items=items, gold=gold, family=family,
                                       retrieved=retrieved), ex)
    ev = A.parse_evidence(A.extract_json(ev_g[0] if ev_g else ""))

    s, u = A.build_verifier(items, doc)
    vf_g = sample(llm, route, "detect", "VERIFIER", s, u, greedy, 1, 0.0,
                  A.ROLES["VERIFIER"].max_tokens, seed)
    outs = sample(llm, route, "detect", "VERIFIER", s, u, cfg, k, temp,
                  A.ROLES["VERIFIER"].max_tokens, seed)
    emit("VERIFIER", gid("detect", "VERIFIER", meta["doc_id"]), s, u, outs,
         lambda raw: RW.score_verifier(raw, items=items, gold=gold), ex)
    vf = A.parse_verifier(A.extract_json(vf_g[0] if vf_g else ""))

    s, u = A.build_leader(doc, sigs, items, ev, vf, region_chars=cfg.region_chars)
    outs = sample(llm, route, "detect", "LEADER", s, u, cfg, k, temp,
                  A.ROLES["LEADER"].max_tokens, seed)
    emit("LEADER", gid("detect", "LEADER", meta["doc_id"]), s, u, outs,
         lambda raw: RW.score_leader(raw, doc=doc, gold=gold, label=label), ex)


# ======================================================================================
# localize rollouts for one document
# ======================================================================================
def localize_doc(doc: str, meta: Dict, gold, label: int, llm, route, cfg, W: Writer,
                 k: int, temp: float, rng: random.Random, clean_keep: float, seed: int):
    greedy = K.CascadeConfig(**{**cfg.__dict__, "temperature": 0.0})
    det = K.detect(doc, llm, route, greedy)
    # At inference localize runs on FLAGGED documents only. Attacked documents are always
    # used (a missed attack still teaches where the text is); clean documents only when
    # the cascade flagged them (the realistic false-positive input) or, at a small rate,
    # unflagged -- the restraint examples.
    if label == 0 and not det.get("pred") and rng.random() >= clean_keep:
        return
    items = [A.Item(**d) for d in det.get("items") or []]
    vf = {int(x): v for x, v in (det.get("verifier") or {}).items()}
    s, u = A.build_localize(doc, items, vf)
    outs = sample(llm, route, "localize", "LEADER", s, u, cfg, k, temp,
                  A.ROLES["LEADER_LOCALIZE"].max_tokens, seed)
    g = gid("localize", "LEADER", meta["doc_id"])
    for j, raw in enumerate(outs):
        sc = RW.score_localize(raw, doc=doc, gold=gold, label=label)
        W.put(dict(meta, label=label, task="localize", agent="LEADER", group_id=g, k=j,
                   greedy=False, system=s, user=u, completion=raw, score=sc.as_dict(),
                   flagged_by_cascade=int(det.get("pred", 0))))


# ======================================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="v8 rollouts (detect | localize)")
    ap.add_argument("--model", required=True, choices=sorted(C.MODELS))
    ap.add_argument("--task", required=True, choices=list(C.TASKS))
    ap.add_argument("--inputs", default="")
    ap.add_argument("--policy-ablation", dest="policy", default="zeroshot",
                    help="which adapters generate the rollouts (zeroshot = base model)")
    ap.add_argument("--n-samples", dest="k", type=int, default=C.GEN.n_samples)
    ap.add_argument("--temperature", type=float, default=C.GEN.temperature)
    ap.add_argument("--max-pairs", dest="max_pairs", type=int, default=0)
    ap.add_argument("--sources", nargs="*", default=[])
    ap.add_argument("--inject-gold", dest="inject_gold", type=float, default=0.5)
    ap.add_argument("--clean-keep", dest="clean_keep", type=float, default=0.3)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", dest="out_dir", default="")
    a = ap.parse_args()

    spec = C.MODELS[a.model]
    if spec.api:
        raise SystemExit("[FATAL] rollouts train an adapter; an API model has none")
    mk = a.model.replace("/", "_")
    inp = a.inputs or C.wpath(C.D_INPUTS, mk, "pairs_all.csv.gz")
    df = pd.read_csv(inp, low_memory=False)
    df = df[df["split"] == "train"]
    if a.sources:
        df = df[df["input_source"].isin(a.sources)]
    df = df.sample(frac=1.0, random_state=a.seed).reset_index(drop=True)
    if a.max_pairs > 0 and len(df) > a.max_pairs:
        # The small sources (ARGAR, BTR) are kept WHOLE and insertion fills the rest: a
        # random cap would otherwise leave the repackaging/perturbation threats with a
        # handful of rollouts, or none.
        small = df[df["input_source"] != "insertion"]
        if len(small) >= a.max_pairs:
            small = small.head(a.max_pairs)
        ins = df[df["input_source"] == "insertion"].head(max(0, a.max_pairs - len(small)))
        df = pd.concat([small, ins]).sample(frac=1.0, random_state=a.seed).reset_index(drop=True)
    if df.empty:
        raise SystemExit(f"[FATAL] no TRAIN pairs in {inp}. See inputs/<model>/manifest.json:"
                         f" ARGAR/BTR on TEST papers are test-only under --split-mode "
                         f"transfer, and insertion TRAIN needs --insertion-train.")
    out_dir = a.out_dir or C.wpath(C.D_TRAJ, mk, a.task)
    os.makedirs(out_dir, exist_ok=True)

    llm = make_backend(a.model)
    route = build_routing(a.model, a.policy, llm.base)
    if route.adapters():
        llm.require(route.adapters())
    cfg = K.CascadeConfig(system_role=spec.system_role, temperature=a.temperature)
    W = Writer(os.path.join(out_dir, "samples.jsonl"))
    print(f"[rollout] {a.model} task={a.task} policy={a.policy} pairs={len(df)} k={a.k} "
          f"T={a.temperature}")
    print(f"[rollout] sources {dict(df['input_source'].value_counts())}")

    def one(i: int, r: pd.Series):
        rng = random.Random(a.seed * 100003 + i)
        gold = [tuple(x) for x in json.loads(r.get("gold_offsets") or "[]")]
        for label, doc, twin in ((0, str(r["benign_document"]), str(r["attacked_document"])),
                                 (1, str(r["attacked_document"]), str(r["benign_document"]))):
            meta = {"model": a.model, "pair_uid": r["pair_uid"],
                    "doc_id": f"{r['pair_uid']}|{'A' if label else 'B'}",
                    "paper_id": str(r["paper_id"]), "condition": str(r["condition"]),
                    "threat_family": r["threat_family"], "input_source": r["input_source"]}
            g = gold if label == 1 else []
            if a.task == "detect":
                detect_doc(doc, twin, meta, g, label, llm, route, cfg, W, a.k,
                           a.temperature, rng, a.inject_gold, a.seed)
            else:
                localize_doc(doc, meta, g, label, llm, route, cfg, W, a.k, a.temperature,
                             rng, a.clean_keep, a.seed)

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, a.concurrency)) as ex:
        futs = [ex.submit(one, i, r) for i, r in df.iterrows()]
        for fu in as_completed(futs):
            try:
                fu.result()
            except Exception as e:
                print(f"[error] {type(e).__name__}: {e}")
            done += 1
            if done % 10 == 0 or done == len(futs):
                print(f"[rollout] {done}/{len(futs)} pairs  {(time.time()-t0)/60:.1f} min  "
                      f"calls {llm.n_calls}", flush=True)
    W.close()

    stats = {"model": a.model, "task": a.task, "policy": a.policy, "k": a.k,
             "n_pairs": len(df), "minutes": (time.time() - t0) / 60, "llm_calls": llm.n_calls,
             "by_agent": {}}
    print(f"\n  {'agent':<10} {'samples':>8} {'usable':>8} {'mean r':>8} {'parse':>6} "
          f"{'groups':>7} {'w/ var':>7}")
    for ag, s in sorted(W.stats.items()):
        grps = [v for (g_ag, _), v in W.groups.items() if g_ag == ag]
        var = sum(1 for v in grps if RW.group_advantages(v) is not None)
        s2 = dict(s, mean_reward=s["reward_sum"] / max(1, s["n"]), groups=len(grps),
                  groups_with_variance=var)
        stats["by_agent"][ag] = s2
        print(f"  {ag:<10} {s['n']:>8} {s['usable']:>8} {s2['mean_reward']:>+8.2f} "
              f"{s['parse_fail']:>6} {len(grps):>7} {var:>7}")
    print("\n  'w/ var' is what DPO and GRPO can learn from. A group whose k samples all "
          "scored the same carries no preference and no advantage.")
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[saved] {out_dir}/samples.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())