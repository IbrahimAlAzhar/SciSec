#!/usr/bin/env python3
"""
v8_pairs.py — turn scored rollouts into the three training sets.

    datasets/sft/<model>/<task>/{all,<AGENT>}.{train,val}.jsonl    prompt / completion
    datasets/dpo/<model>/<task>/{all,<AGENT>}.{train,val}.jsonl    prompt / chosen / rejected
    datasets/grpo/<model>/<task>/{train,val}.jsonl                 messages / completion / adv

Formats match what V4's train_sft.py / train_dpo.py already consume (chat-message lists,
so TRL applies each model's own template -- gemma has no system role).

SFT     the best USABLE sample of each prompt; the gold target when none was usable.
DPO     chosen = best sample (or gold), rejected = the WORST sample of the SAME prompt, and
        only when the reward gap is >= --min-margin. A near-tie is not a preference.
        When every sample was correct, a targeted corruption stands in -- capped at
        --max-synth (default 35%) so the model is not trained to beat a straw man.
GRPO    groups with reward variance only; advantage = group-normalised reward. For the
        LEADER the PROVENANCE reward is used when --reward provenance.

SPLIT BY PAPER, never by row: one paper is several prompts, and a row split puts the same
paper on both sides and makes the validation loss meaningless.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd

import ft_config as C
from v8_tools import scrub as _scrub
import v8_agents as A
import v8_cascade as K
import v8_rewards as RW
import v8_tools as T


def write_jsonl(path: str, rows) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(_scrub(r), ensure_ascii=False) + "\n")
            n += 1
    return n


def val_papers(papers, frac: float, seed: int) -> set:
    ps = sorted(set(papers))
    random.Random(seed).shuffle(ps)
    return set(ps[:max(1, int(round(frac * len(ps))))]) if ps else set()


# ======================================================================================
# gold targets and targeted corruptions, reconstructed from the pair
# ======================================================================================
def gold_for(s: Dict, doc: str, gold) -> Optional[str]:
    ag, task, label, fam = s["agent"], s["task"], int(s["label"]), s["threat_family"]
    items = [A.Item(**d) for d in s.get("items") or []]
    if task == "localize":
        return A.render(A.gold_localize(doc, gold, label))
    if ag == "LOCATOR":
        w = tuple(s.get("window") or (0, len(doc)))
        return A.render(A.gold_locator(doc, w, gold, label, fam))
    if ag == "EVIDENCE":
        return A.render(A.gold_evidence(items, gold, fam, K.retrieve_for(doc, items, 3)))
    if ag == "VERIFIER":
        return A.render(A.gold_verifier(items, gold, fam))
    if ag == "LEADER":
        return A.render(A.gold_leader(doc, gold, label, fam))
    return None


def _author_sentence(doc: str, gold, rng: random.Random) -> str:
    body = T.excise(doc, gold) if gold else doc
    tail = body[-5000:]
    sents = [x.strip() for x in re.split(r"(?<=[.!?])\s+", tail) if 60 <= len(x.strip()) <= 300]
    return rng.choice(sents) if sents else ""


def corrupt(s: Dict, chosen: str, doc: str, gold, rng: random.Random) -> Optional[Tuple[str, str]]:
    """A controlled wrong answer aimed at a failure the tables actually show."""
    ag, task, label = s["agent"], s["task"], int(s["label"])
    obj = A.extract_json(chosen) or {}
    if task == "localize":
        if label == 1 and gold:
            a, b = T.merge(gold)[0]
            if rng.random() < 0.5:
                return A.render({"delete": [doc[a:a + max(12, (b - a) // 2)]],
                                 "confidence": 0.8, "note": "removed the instruction"}), "partial"
            sent = _author_sentence(doc, gold, rng)
            if sent:
                return A.render({"delete": [doc[a:b], sent], "confidence": 0.8,
                                 "note": "removed the manipulation"}), "collateral"
        sent = _author_sentence(doc, gold, rng)
        return (A.render({"delete": [sent], "confidence": 0.7,
                          "note": "removed a softened limitation"}), "delete_author") \
            if sent else None
    if ag == "LEADER":
        if label == 0:
            sent = _author_sentence(doc, gold, rng)
            return (A.render({"verdict": "ATTACKED", "confidence": 0.8, "attack_type":
                              "spin", "span": sent, "reason": "The limitations are phrased "
                              "to soften criticism."}), "false_positive") if sent else None
        return (A.render({"verdict": "CLEAN", "confidence": 0.7, "attack_type": "", "span": "",
                          "reason": "Ordinary discussion of scope."}), "miss")
    if ag == "LOCATOR":
        w = s.get("window") or [0, len(doc)]
        if label == 1 and T.intersect([tuple(w)], gold) > 0:
            return A.render({"candidates": [], "claims_to_check": [], "confidence": 0.3}), "miss"
        sent = _author_sentence(doc[w[0]:w[1]], [], rng)
        return (A.render({"candidates": [{"span": sent, "requested_behavior": "soften "
                          "criticism", "task_conflict": "tone", "kind": "pressure"}],
                          "claims_to_check": [], "confidence": 0.7}), "false_alarm") \
            if sent else None
    if ag in ("VERIFIER", "EVIDENCE") and isinstance(obj.get("items"), list):
        flip = []
        for it in obj["items"]:
            it = dict(it)
            if ag == "VERIFIER":
                it["decision"] = {"CONFIRMED": "DISMISSED", "DISMISSED": "CONFIRMED"}.get(
                    str(it.get("decision", "")).upper(), "CONFIRMED")
            else:
                it["reviewer_redirection"] = not bool(it.get("reviewer_redirection"))
            flip.append(it)
        return A.render({"items": flip}), "flipped"
    return None


# ======================================================================================
def build(model: str, task: str, *, inputs: str, samples: str, out_root: str,
          val_frac: float, min_margin: float, max_synth: float, reward: str,
          seed: int) -> Dict:
    spec = C.MODELS[model]
    pairs = pd.read_csv(inputs, low_memory=False).set_index("pair_uid")
    recs = [json.loads(l) for l in open(samples, encoding="utf-8") if l.strip()]
    groups: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for r in recs:
        if not r.get("greedy"):
            groups[(r["agent"], r["group_id"])].append(r)
    rng = random.Random(seed)
    vp = val_papers([r["paper_id"] for r in recs], val_frac, seed)
    sft, dpo, grpo = [], [], []
    st = defaultdict(lambda: defaultdict(int))
    order = sorted(groups)
    rng.shuffle(order)
    synth_budget = int(max_synth * len(order)) + 1
    n_synth = 0

    def doc_gold(s):
        p = pairs.loc[s["pair_uid"]]
        doc = str(p["attacked_document"] if int(s["label"]) == 1 else p["benign_document"])
        g = [tuple(x) for x in json.loads(p.get("gold_offsets") or "[]")] \
            if int(s["label"]) == 1 else []
        return doc, g

    def rew(s):
        if reward == "provenance" and s["agent"] == "LEADER" and s.get("provenance"):
            return float(s["provenance"]["reward"])
        return float(s["score"]["reward"])

    for key in order:
        g = groups[key]
        s0 = g[0]
        ag = s0["agent"]
        d = st[ag]
        d["groups"] += 1
        msgs = A.messages(s0["system"], s0["user"], system_role=spec.system_role)
        base = {"agent": ag, "task": task, "paper_id": s0["paper_id"],
                "condition": s0["condition"], "label": int(s0["label"]),
                "input_source": s0.get("input_source", ""), "group_id": s0["group_id"]}
        usable = sorted([s for s in g if s["score"]["usable"]], key=rew, reverse=True)
        ranked = sorted(g, key=rew)

        # ---- SFT ---------------------------------------------------------------------
        if usable:
            chosen, csrc, crew = usable[0]["completion"].strip(), "sampled", rew(usable[0])
        else:
            doc, gold = doc_gold(s0)
            gt = gold_for(s0, doc, gold)
            if gt is None:
                continue
            chosen, csrc, crew = gt, "gold", float("inf")
            d["gold_targets"] += 1
        sft.append(dict(base, prompt=msgs,
                        completion=[{"role": "assistant", "content": chosen}], source=csrc))
        d["sft"] += 1

        # ---- DPO ---------------------------------------------------------------------
        rej, rsrc = None, ""
        worst = ranked[0] if ranked else None
        if worst is not None and worst["completion"].strip() != chosen and \
                (crew - rew(worst)) >= min_margin and worst["score"]["parsed"]:
            rej, rsrc = worst["completion"].strip(), "sampled"
        elif n_synth < synth_budget:
            doc, gold = doc_gold(s0)
            c = corrupt(s0, chosen, doc, gold, rng)
            if c is not None and c[0] != chosen:
                rej, rsrc = c[0], f"synth:{c[1]}"
                n_synth += 1
                d["synth"] += 1
        if rej:
            dpo.append(dict(base, prompt=msgs,
                            chosen=[{"role": "assistant", "content": chosen}],
                            rejected=[{"role": "assistant", "content": rej}],
                            chosen_source=csrc, rejected_source=rsrc))
            d["dpo"] += 1

        # ---- GRPO --------------------------------------------------------------------
        rs = [rew(s) for s in g]
        adv = RW.group_advantages(rs)
        if adv is None:
            d["grpo_dropped_no_variance"] += 1
            continue
        for s, av, rr in zip(g, adv, rs):
            if s["completion"].strip():
                grpo.append(dict(base, messages=msgs, completion=s["completion"],
                                 adv=av, reward=rr))
        d["grpo_groups"] += 1

    out = {}
    # SFT/DPO are defined by the STANDARD reward. A provenance pass only (re)writes the
    # GRPO set, into its own directory, so it can never overwrite the SFT/DPO data.
    for kind, rows in ((("sft", sft), ("dpo", dpo)) if reward == "standard" else ()):
        root = os.path.join(out_root, "datasets", kind, model.replace("/", "_"), task)
        tr = [r for r in rows if r["paper_id"] not in vp]
        va = [r for r in rows if r["paper_id"] in vp]
        out[kind] = {"train": write_jsonl(os.path.join(root, "all.train.jsonl"), tr),
                     "val": write_jsonl(os.path.join(root, "all.val.jsonl"), va)}
        for ag in C.TASK_AGENTS[task]:
            write_jsonl(os.path.join(root, f"{ag}.train.jsonl"), [r for r in tr if r["agent"] == ag])
            write_jsonl(os.path.join(root, f"{ag}.val.jsonl"), [r for r in va if r["agent"] == ag])
    groot = os.path.join(out_root, "datasets", "grpo", model.replace("/", "_"), task, reward)
    out["grpo"] = {"train": write_jsonl(os.path.join(groot, "train.jsonl"),
                                        [r for r in grpo if r["paper_id"] not in vp]),
                   "val": write_jsonl(os.path.join(groot, "val.jsonl"),
                                      [r for r in grpo if r["paper_id"] in vp]),
                   "reward": reward}
    stats = {"model": model, "task": task, "n_samples": len(recs), "n_groups": len(groups),
             "synthetic_rejected": n_synth, "val_papers": len(vp), "files": out,
             "by_agent": {k: dict(v) for k, v in st.items()}}
    sdir = os.path.join(out_root, "datasets", "grpo" if reward != "standard" else "sft",
                        model.replace("/", "_"), task)
    os.makedirs(sdir, exist_ok=True)
    with open(os.path.join(sdir, f"stats_{reward}.json"), "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description="v8 SFT / DPO / GRPO datasets from rollouts")
    ap.add_argument("--model", required=True, choices=sorted(C.MODELS))
    ap.add_argument("--task", required=True, choices=list(C.TASKS))
    ap.add_argument("--inputs", default="")
    ap.add_argument("--samples", default="")
    ap.add_argument("--val-frac", dest="val_frac", type=float, default=0.1)
    ap.add_argument("--min-margin", dest="min_margin", type=float, default=0.5)
    ap.add_argument("--max-synth", dest="max_synth", type=float,
                    default=C.GEN.max_synthetic_rejected_frac)
    ap.add_argument("--reward", choices=["standard", "provenance"], default="standard")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    mk = a.model.replace("/", "_")
    inputs = a.inputs or C.wpath(C.D_INPUTS, mk, "pairs_all.csv.gz")
    samples = a.samples or C.wpath(C.D_TRAJ, mk, a.task, "samples.jsonl")
    for p in (inputs, samples):
        if not os.path.exists(p):
            raise SystemExit(f"[FATAL] {p} not found")
    st = build(a.model, a.task, inputs=inputs, samples=samples, out_root=C.WORK,
               val_frac=a.val_frac, min_margin=a.min_margin, max_synth=a.max_synth,
               reward=a.reward, seed=a.seed)
    print(json.dumps(st, indent=2))
    for ag, d in st["by_agent"].items():
        if d.get("sft", 0) < 100:
            print(f"[warn] {a.task}/{ag}: {d.get('sft', 0)} SFT examples is thin for a "
                  f"rank-{C.TRAIN.lora_r} adapter; the shared 'all' set is the safer choice.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())