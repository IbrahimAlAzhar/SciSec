#!/usr/bin/env python3
"""
sr_attacker.py — SafeReview Phase 1 (attack evolution) in three offline steps, plus the
injection regeneration Phase 2 and the adaptive test set need.

    rollout   Algorithm 1 lines 3-8: sample B training papers; for each, G injections
              tau_{i,g} ~ A_{t-1}(.|p_i) from the SERVED attacker.
    score     lines 5, 9-13: s_orig = Extract(R_{t-1}(p_i)), s_adv = Extract(R_{t-1}(p_i+tau)),
              r = s_adv - s_orig (Eq. 1), A = (r - mean_g r) / std_g r (Eq. 4).
    regen     "After GRPO training converges, we use the updated attacker A_t to regenerate
              a fresh set of injections over the training paper set" (Sec. 3.2). With
              --test it attacks the TEST papers instead and writes an adaptive test set
              (SafeReview's own threat model: a GRPO attacker transferred to the test set).

WHY THREE STEPS AND NOT ONE
    The paper serves the reviewer on 4 GPUs as an online reward model while 4 more train
    the attacker. Your jobs get one 40 GiB card, so the attacker and the reviewer cannot be
    resident together: rollout (attacker served), score (reviewer served) and update
    (sr_train_attacker.py, nothing served) run one after another. That is exactly v7's
    offline GRPO design, for the same reason.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

import numpy as np
import pandas as pd

import sr_config as S
import sr_data as D
import sr_inject as I
import sr_review as R
from sr_vllm import ChatClient, ChatError, pmap, require_served


def _attacker_client(a) -> ChatClient:
    spec = S.ATTACKERS[a.attacker]
    return ChatClient(a.vllm_url, a.served, chat_kwargs=spec.chat_kwargs)


def _train_keys(a, clean: Dict[str, Dict]) -> List[str]:
    rec, _ = S.inputs_paths(a.model, "")
    test_pids = D.test_paper_ids(rec)
    keys, n_drop = D.disjoint(list(clean), test_pids, a.train_papers)
    if n_drop:
        print(f"[papers] --train-papers disjoint: dropped {n_drop} training papers that are "
              f"test papers (ARGAR/BTR train halves)")
    return keys


def _write_jsonl(path: str, rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _read_jsonl(path: str) -> List[Dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


# ======================================================================================
# rollout
# ======================================================================================
def cmd_rollout(a) -> int:
    train = D.load_train(a.train_docs)
    clean = D.clean_docs(train)
    keys = D.sample_keys(_train_keys(a, clean), a.n_papers, a.seed, a.model, a.round,
                         a.iter, "rollout")
    cl = _attacker_client(a)
    require_served(cl, a.served)
    print(f"[rollout] attacker {a.served}: {len(keys)} papers x G={a.group_size}  "
          f"T={a.temperature} top_p={a.top_p} max_new={a.max_new_tokens}")

    def one(k: str) -> List[Dict]:
        msgs = I.attacker_messages(clean[k]["document"], a.ctx_chars)
        try:
            outs = cl.chat(msgs, n=a.group_size, temperature=a.temperature, top_p=a.top_p,
                           max_tokens=a.max_new_tokens)
        except ChatError as e:
            print(f"[rollout] {k}: {e}")
            return []
        rows = []
        for g, raw in enumerate(outs):
            inj, ns, ok, why = I.clean_injection(raw, max_chars=S.SR.atk_max_chars)
            ok = ok and ns >= a.min_sentences
            rows.append({"paper_key": k, "group_index": g, "round": a.round,
                         "iter": a.iter, "messages": msgs, "completion": raw,
                         "injection": inj, "n_sentences": ns, "valid": bool(ok),
                         "why": why or ("" if ok else f"{ns} sentences")})
        return rows

    res = pmap(one, keys, workers=a.workers, label="rollout")
    rows = [r for rs in res for r in rs]
    _write_jsonl(a.out, rows)
    v = [r for r in rows if r["valid"]]
    print(f"[rollout] {len(rows)} completions, {len(v)} valid "
          f"({100*len(v)/max(1,len(rows)):.1f}%), mean sentences "
          f"{np.mean([r['n_sentences'] for r in v]) if v else 0:.1f}  -> {a.out}")
    if len(v) < 0.5 * len(rows):
        print("[rollout] FEWER THAN HALF the completions are usable injections. Check one "
              "by hand: a refusal or a summary of the paper scores the invalid reward.")
    return 0


# ======================================================================================
# score
# ======================================================================================
def rank_term(s_orig_all: Dict[str, float], k: str, s_adv: float) -> float:
    """Ranking displacement within the batch, in [-1, 1]: how many of the other papers'
    CLEAN scores the attacked paper overtook, normalised by B-1. Positive = moved up. Only
    used when --rank-weight > 0 (Eq. 1 has the score increase alone)."""
    others = [v for kk, v in s_orig_all.items() if kk != k]
    if not others:
        return 0.0
    s0 = s_orig_all[k]
    above_before = sum(1 for v in others if v > s0)
    above_after = sum(1 for v in others if v > s_adv)
    return (above_before - above_after) / len(others)


def group_advantages(by: Dict[str, List[Dict]], min_std: float = 1e-6):
    """Eq. 4 in place: A = (r - mean_g r) / std_g r per paper. A group whose G rewards are
    all equal carries no gradient and is DROPPED (v7's min_group_std rule), not kept at 0.
    Returns (n_groups, n_flat)."""
    n_groups = n_flat = 0
    for _k, grp in by.items():
        n_groups += 1
        rs = np.array([g["reward"] for g in grp], float)
        sd = float(rs.std(ddof=1)) if len(rs) > 1 else 0.0
        if sd < min_std:
            n_flat += 1
            for g in grp:
                g.update({"advantage": 0.0, "group_dropped": True,
                          "drop_reason": "no reward variance"})
            continue
        mu = float(rs.mean())
        for g in grp:
            g.update({"advantage": (g["reward"] - mu) / sd, "group_dropped": False,
                      "drop_reason": ""})
    return n_groups, n_flat


def cmd_score(a) -> int:
    rows = _read_jsonl(a.rollouts)
    train = D.load_train(a.train_docs)
    clean = D.clean_docs(train)
    rv = R.make_reviewer(a, model_key=a.model, url=a.vllm_url, served=a.served,
                         adapter_dir=a.adapter_dir)
    require_served(rv.cl, a.served)
    print(rv.describe())
    keys = sorted({r["paper_key"] for r in rows})
    t0 = time.time()
    s0 = rv.review_many([clean[k]["document"] for k in keys], workers=a.workers,
                        label="clean")
    s_orig = {k: r["rating"] for k, r in zip(keys, s0)}

    # build every attacked document, then review them in parallel
    todo = []
    for r in rows:
        r.update({"s_orig": s_orig.get(r["paper_key"]), "s_adv": None, "delta": None,
                  "section": "", "ins_start": -1, "ins_end": -1, "visible": None,
                  "review_failed": False})
        if not r["valid"] or r["s_orig"] is None:
            continue
        rng = I.rng_for(a.seed, a.model, a.round, a.iter, r["paper_key"], r["group_index"])
        ins = I.insert(clean[r["paper_key"]]["document"], r["injection"], rng=rng,
                       visible_chars=a.review_max_chars)
        r.update({"section": ins["section"], "ins_start": ins["ins_start"],
                  "ins_end": ins["ins_end"], "visible": ins["visible"]})
        todo.append((r, ins["document"]))
    adv = rv.review_many([d for _, d in todo], workers=a.workers, label="attacked")
    for (r, _), res in zip(todo, adv):
        if res["rating"] is None:
            r["review_failed"] = True
            r["s_adv"] = r["s_orig"]
        else:
            r["s_adv"] = res["rating"]
        r["delta"] = r["s_adv"] - r["s_orig"]

    # rewards and group-normalised advantages (Eq. 1, Eq. 4)
    s_valid = {k: v for k, v in s_orig.items() if v is not None}
    by: Dict[str, List[Dict]] = {}
    for r in rows:
        if r["s_orig"] is None:
            r.update({"reward": None, "advantage": 0.0, "group_dropped": True,
                      "drop_reason": "clean review failed"})
            continue
        if r["valid"]:
            r["reward"] = float(r["delta"])
            if a.rank_weight:
                r["reward"] += a.rank_weight * rank_term(s_valid, r["paper_key"], r["s_adv"])
        else:
            r["reward"] = float(a.invalid_reward)
        by.setdefault(r["paper_key"], []).append(r)
    n_groups, n_flat = group_advantages(by)
    _write_jsonl(a.out, rows)

    v = [r for r in rows if r["valid"] and r["delta"] is not None]
    st = {"model": a.model, "reviewer": a.served, "round": a.round, "iter": a.iter,
          "n_papers": len(keys), "n_rollouts": len(rows),
          "valid_frac": len(v) / max(1, len(rows)),
          "mean_s_orig": float(np.mean([x for x in s_valid.values()])) if s_valid else None,
          "mean_delta_valid": float(np.mean([r["delta"] for r in v])) if v else None,
          "asr_valid": float(np.mean([r["delta"] > 0 for r in v])) if v else None,
          "mean_reward": float(np.mean([r["reward"] for r in rows
                                        if r.get("reward") is not None] or [0])),
          "no_signal_group_frac": n_flat / max(1, n_groups),
          "invisible_injections": sum(1 for r in v if r["visible"] is False),
          "review_failures": sum(1 for r in rows if r.get("review_failed")),
          "rank_weight": a.rank_weight, "minutes": (time.time() - t0) / 60}
    with open(os.path.join(os.path.dirname(a.out), "stats.json"), "w") as f:
        json.dump(st, f, indent=2)
    print("\n" + "=" * 80)
    print(f"ATTACK ROLLOUT SCORED — round {a.round}, iter {a.iter}, reviewer {a.served}")
    print("=" * 80)
    for k in ("n_papers", "n_rollouts", "valid_frac", "mean_s_orig", "mean_delta_valid",
              "asr_valid", "mean_reward", "no_signal_group_frac", "review_failures"):
        print(f"  {k:<24} {st[k]}")
    if st["no_signal_group_frac"] > 0.5:
        print("  MORE THAN HALF the groups carry no gradient (all G injections moved the "
              "rating by the same amount). Raise ATK_TEMP, or --n-reviews with a review "
              "temperature for a finer-grained reward.")
    return 0


# ======================================================================================
# regen
# ======================================================================================
def _gen_one(cl: ChatClient, doc: str, a, tries: int = 4) -> Dict:
    msgs = I.attacker_messages(doc, a.ctx_chars)
    try:
        outs = cl.chat(msgs, n=tries, temperature=a.temperature, top_p=a.top_p,
                       max_tokens=a.max_new_tokens)
    except ChatError as e:
        return {"raw": "", "injection": "", "valid": False, "why": str(e)[:200]}
    for raw in outs:
        inj, ns, ok, why = I.clean_injection(raw, max_chars=S.SR.atk_max_chars)
        if ok and ns >= a.min_sentences:
            return {"raw": raw, "injection": inj, "n_sentences": ns, "valid": True, "why": ""}
    return {"raw": outs[0] if outs else "", "injection": "", "valid": False,
            "why": "no valid sample"}


def cmd_regen(a) -> int:
    cl = _attacker_client(a)
    require_served(cl, a.served)
    if a.test:
        return regen_test(a, cl)
    train = D.load_train(a.train_docs)
    clean = D.clean_docs(train)
    keys = D.sample_keys(_train_keys(a, clean), a.n_papers, a.seed, a.model, a.round,
                         "regen")
    print(f"[regen] attacker {a.served}: one injection for each of {len(keys)} training "
          f"papers (Phase 2, round {a.round})")
    gens = pmap(lambda k: _gen_one(cl, clean[k]["document"], a), keys, workers=a.workers,
                label="regen")
    rows = []
    for k, g in zip(keys, gens):
        if not g["valid"]:
            continue
        rng = I.rng_for(a.seed, a.model, a.round, "regen", k)
        ins = I.insert(clean[k]["document"], g["injection"], rng=rng,
                       visible_chars=a.review_max_chars)
        rows.append({"paper_key": k, "round": a.round, "attacker": a.served,
                     "injection": g["injection"], "raw": g["raw"],
                     "n_sentences": g["n_sentences"], "section": ins["section"],
                     "ins_start": ins["ins_start"], "ins_end": ins["ins_end"],
                     "visible": ins["visible"],
                     "input_source": clean[k]["input_source"],
                     "attacked_document": ins["document"]})
    _write_jsonl(a.out, rows)
    print(f"[regen] {len(rows)}/{len(keys)} papers carry a valid injection -> {a.out}")
    return 0


def regen_test(a, cl: ChatClient) -> int:
    """The adaptive test set: the final attacker A_T attacks each test paper's CLEAN
    document once. Written in the same records/index format as the main test set, as
    consecutive clean/attacked pairs, so every row is scored on it by the same code."""
    rec, idx = S.inputs_paths(a.model, a.run_tag)
    te = D.load_test(rec, idx)
    cl_docs = te[(te["label"] == 0) & te["input_source"].isin(a.test_sources)]
    cl_docs = cl_docs.drop_duplicates("pair_uid")
    if a.max_rows:
        cl_docs = cl_docs.head(a.max_rows)
    print(f"[regen-test] {len(cl_docs)} clean test documents from {a.test_sources}")
    gens = pmap(lambda d: _gen_one(cl, d, a), list(cl_docs["document"]), workers=a.workers,
                label="regen-test")
    recs, ix = [], []
    tag = a.set_name
    for (_, r), g in zip(cl_docs.iterrows(), gens):
        if not g["valid"]:
            continue
        rng = I.rng_for(a.seed, a.model, tag, r["pair_uid"])
        ins = I.insert(r["document"], g["injection"], rng=rng,
                       visible_chars=a.review_max_chars)
        uid = f"{tag}|{r['pair_uid']}"
        base = {"paper_id": str(r["paper_id"]), "row_index": int(len(recs) // 2),
                "slot": str(r.get("slot") or "limitations_end"),
                "max_chars": int(r.get("max_chars") or 30000)}
        c = dict(base, doc_id=f"{uid}|clean", label=0, condition="A0_clean",
                 group="adaptive", document=r["document"], insert="",
                 base_rating=None, attacked_rating=None, d_rating=None, asr=None)
        t = dict(base, doc_id=f"{uid}|attacked", label=1, condition=f"G_{tag}",
                 group="adaptive", document=ins["document"], insert=g["injection"],
                 base_rating=None, attacked_rating=None, d_rating=None, asr=None)
        for rr in (c, t):
            ix.append({"eval_index": len(recs), "doc_id": rr["doc_id"], "pair_uid": uid,
                       "input_source": "grpo_attacker",
                       "data_origin": r["input_source"], "threat_family": "adaptive",
                       "attack_generator": a.served, "label": rr["label"],
                       "condition": rr["condition"], "harmful": 0,
                       "seen_in_train": int(r.get("seen_in_train", 0)),
                       "section": ins["section"] if rr["label"] else ""})
            recs.append(rr)
    rp, ip = S.inputs_paths(a.model, a.run_tag, tag)
    _write_jsonl(rp, recs)
    pd.DataFrame(ix).to_csv(ip, index=False)
    print(f"[regen-test] {len(recs)//2} adaptive pairs -> {rp}\n             index -> {ip}")
    return 0


# ======================================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="SafeReview attacker: rollout / score / regen")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--model", required=True, help="the REVIEWER this attacker targets")
        p.add_argument("--attacker", default=S.DEFAULT_ATTACKER, choices=sorted(S.ATTACKERS))
        p.add_argument("--served", required=True,
                       help="served model name (base path, or an adapter's name)")
        p.add_argument("--vllm-url", dest="vllm_url", default=os.environ.get("VLLM_URL", ""))
        p.add_argument("--round", type=int, default=1)
        p.add_argument("--iter", type=int, default=1)
        p.add_argument("--train-docs", dest="train_docs",
                       default=S.wpath(S.D_INPUTS, "train_documents.csv"))
        p.add_argument("--train-papers", dest="train_papers", default="all",
                       choices=["all", "disjoint"])
        p.add_argument("--out", default="")
        p.add_argument("--seed", type=int, default=S.SR.seed)

    def gen_args(p):
        p.add_argument("--workers", type=int, default=16)
        p.add_argument("--temperature", type=float, default=S.SR.atk_temperature)
        p.add_argument("--top-p", dest="top_p", type=float, default=S.SR.atk_top_p)
        p.add_argument("--max-new-tokens", dest="max_new_tokens", type=int,
                       default=S.SR.atk_max_new_tokens)
        p.add_argument("--ctx-chars", dest="ctx_chars", type=int, default=S.SR.atk_ctx_chars)
        p.add_argument("--min-sentences", dest="min_sentences", type=int,
                       default=S.SR.atk_min_sentences)

    p = sub.add_parser("rollout")
    common(p); gen_args(p)
    p.add_argument("--n-papers", dest="n_papers", type=int, default=S.SR.atk_papers)
    p.add_argument("--group-size", dest="group_size", type=int, default=S.SR.group_size)

    p = sub.add_parser("score")
    common(p)
    R.add_reviewer_args(p)
    p.add_argument("--rollouts", required=True)
    p.add_argument("--adapter-dir", dest="adapter_dir", default="",
                   help="the defender adapter's directory (cache fingerprint); '' = base")
    p.add_argument("--invalid-reward", dest="invalid_reward", type=float,
                   default=S.SR.invalid_reward)
    p.add_argument("--rank-weight", dest="rank_weight", type=float,
                   default=S.SR.rank_weight)

    p = sub.add_parser("regen")
    common(p); gen_args(p)
    p.add_argument("--n-papers", dest="n_papers", type=int, default=S.SR.dpo_papers)
    p.add_argument("--review-max-chars", dest="review_max_chars", type=int,
                   default=S.SR.review_max_chars)
    p.add_argument("--test", action="store_true",
                   help="attack the TEST papers' clean documents -> adaptive test set")
    p.add_argument("--test-sources", dest="test_sources", nargs="+",
                   default=["insertion"])
    p.add_argument("--set-name", dest="set_name", default="adaptive")
    p.add_argument("--run-tag", dest="run_tag", default="")
    p.add_argument("--max-rows", dest="max_rows", type=int, default=0)

    a = ap.parse_args()
    if not a.vllm_url:
        raise SystemExit("[FATAL] --vllm-url (or VLLM_URL) is required")
    if a.cmd == "rollout":
        a.out = a.out or os.path.join(S.round_dir(a.model, a.round), f"atk_iter{a.iter}",
                                      "rollouts.jsonl")
        return cmd_rollout(a)
    if a.cmd == "score":
        a.out = a.out or os.path.join(os.path.dirname(a.rollouts), "scored.jsonl")
        return cmd_score(a)
    a.out = a.out or os.path.join(S.round_dir(a.model, a.round), "regen.jsonl")
    return cmd_regen(a)


if __name__ == "__main__":
    raise SystemExit(main())
