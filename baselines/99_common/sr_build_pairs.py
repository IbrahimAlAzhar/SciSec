#!/usr/bin/env python3
"""
sr_build_pairs.py — SafeReview Phase 2 (Algorithm 1 lines 17-21): the preference data.

    for each (p_i, tau_i):   y+ = R_{t-1}(p_i)   y- = R_{t-1}(p_i + tau_i)
                             D_pref <- D_pref + {(p_i + tau_i, y+, y-)}

    --source injections   the attacker's regenerated injections (regen.jsonl) -> SafeReview
    --source corpus       your EXISTING training attacks (train_documents.csv, label 1 rows
                          with their clean twin) -> the cheap Static-DPO row

Each pair is written in the paper's App. G format,
    {"chosen":   [system, user(ATTACKED paper), assistant(clean review)],
     "rejected": [system, user(ATTACKED paper), assistant(attacked review)], ...metadata}
(gemma has no system role: its system text is merged into the user turn, exactly as at
inference), and once more in TRL's prompt/chosen/rejected layout (pairs_trl.jsonl) so any
DPO trainer -- including V4's -- can read it.

App. G: "we deliberately do NOT include the anti-injection notice during training, so
that the defender must internalize anti-injection behavior in the model weights rather
than rely on the prompt". The prompt here is the plain reviewer prompt.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

import numpy as np

import sr_config as S
import sr_data as D
import sr_review as R
from sr_vllm import require_served


def main() -> int:
    ap = argparse.ArgumentParser(description="SafeReview preference pairs")
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", required=True, choices=["injections", "corpus"])
    ap.add_argument("--injections", default="", help="regen.jsonl (source=injections)")
    ap.add_argument("--train-docs", dest="train_docs",
                    default=S.wpath(S.D_INPUTS, "train_documents.csv"))
    ap.add_argument("--train-papers", dest="train_papers", default="all",
                    choices=["all", "disjoint"])
    ap.add_argument("--max-pairs", dest="max_pairs", type=int, default=S.SR.static_max_pairs)
    ap.add_argument("--pair-filter", dest="pair_filter", default=S.SR.pair_filter,
                    choices=["all", "changed", "raised"])
    ap.add_argument("--served", required=True, help="R_{t-1}'s served name")
    ap.add_argument("--adapter-dir", dest="adapter_dir", default="")
    ap.add_argument("--vllm-url", dest="vllm_url", default=os.environ.get("VLLM_URL", ""))
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=S.SR.seed)
    R.add_reviewer_args(ap)
    a = ap.parse_args()

    train = D.load_train(a.train_docs)
    clean = D.clean_docs(train)
    rec, _ = S.inputs_paths(a.model, "")
    test_pids = D.test_paper_ids(rec)

    if a.source == "corpus":
        items = D.attacked_pairs(train)
        keep, n_drop = D.disjoint([it["paper_key"] for it in items], test_pids, a.train_papers)
        keep = set(keep)
        items = [it for it in items if it["paper_key"] in keep]
        items = D.stratified(items, a.max_pairs, "input_source", a.seed)
        print(f"[pairs] corpus: {len(items)} (clean, attacked) training pairs"
              + (f" ({n_drop} on test papers dropped)" if n_drop else ""))
    else:
        if not os.path.exists(a.injections):
            raise SystemExit(f"[FATAL] {a.injections} not found (run sr_attacker.py regen)")
        items = []
        for l in open(a.injections, encoding="utf-8"):
            if not l.strip():
                continue
            g = json.loads(l)
            k = g["paper_key"]
            if k not in clean:
                continue
            items.append({"paper_key": k, "clean": clean[k]["document"],
                          "attacked": g["attacked_document"], "insert": g["injection"],
                          "condition": f"G_grpo_r{g.get('round', '')}",
                          "input_source": g.get("input_source", ""),
                          "section": g.get("section", "")})
        print(f"[pairs] injections: {len(items)} regenerated attacks from {a.injections}")

    rv = R.make_reviewer(a, model_key=a.model, url=a.vllm_url, served=a.served,
                         adapter_dir=a.adapter_dir, variant="plain")
    require_served(rv.cl, a.served)
    print(rv.describe())
    t0 = time.time()
    rc = rv.review_many([it["clean"] for it in items], workers=a.workers, label="y+ clean")
    ra = rv.review_many([it["attacked"] for it in items], workers=a.workers,
                        label="y- attacked")

    out: List[Dict] = []
    n_fail = n_filtered = 0
    for it, c, r in zip(items, rc, ra):
        if c["rating"] is None or r["rating"] is None or not c["raw"] or not r["raw"]:
            n_fail += 1
            continue
        d = r["rating"] - c["rating"]
        if (a.pair_filter == "changed" and d == 0) or (a.pair_filter == "raised" and d <= 0):
            n_filtered += 1
            continue
        msgs = rv.messages_for(it["attacked"])        # p_i + tau_i
        out.append({
            "chosen": msgs + [{"role": "assistant", "content": c["raw"]}],
            "rejected": msgs + [{"role": "assistant", "content": r["raw"]}],
            "paper_key": it["paper_key"], "source": a.source,
            "condition": it["condition"], "input_source": it["input_source"],
            "s_orig": c["rating"], "s_adv": r["rating"], "delta": d,
            "reviewer": a.served, "prompt_sha1": rv.sys_hash})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out + ".tmp", "w", encoding="utf-8") as f:
        for p in out:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    os.replace(a.out + ".tmp", a.out)
    trl = a.out.replace(".jsonl", "_trl.jsonl")
    with open(trl, "w", encoding="utf-8") as f:
        for p in out:
            prompt = [m for m in p["chosen"] if m["role"] != "assistant"]
            f.write(json.dumps({"prompt": prompt, "chosen": [p["chosen"][-1]],
                                "rejected": [p["rejected"][-1]],
                                "paper_key": p["paper_key"]}, ensure_ascii=False) + "\n")
    deltas = np.array([p["delta"] for p in out], float) if out else np.array([0.0])
    st = {"model": a.model, "source": a.source, "reviewer": a.served,
          "n_items": len(items), "n_pairs": len(out), "n_review_failures": n_fail,
          "n_filtered": n_filtered, "pair_filter": a.pair_filter,
          "mean_s_orig": float(np.mean([p["s_orig"] for p in out])) if out else None,
          "mean_s_adv": float(np.mean([p["s_adv"] for p in out])) if out else None,
          "mean_delta": float(deltas.mean()), "frac_raised": float((deltas > 0).mean()),
          "frac_unchanged": float((deltas == 0).mean()),
          "minutes": (time.time() - t0) / 60}
    with open(a.out.replace(".jsonl", "_stats.json"), "w") as f:
        json.dump(st, f, indent=2)
    print("\n" + "=" * 80)
    print(f"PREFERENCE PAIRS — {a.model}, source={a.source}, reviewer={a.served}")
    print("=" * 80)
    for k in ("n_items", "n_pairs", "n_review_failures", "n_filtered", "mean_s_orig",
              "mean_s_adv", "mean_delta", "frac_raised", "frac_unchanged"):
        print(f"  {k:<20} {st[k]}")
    if len(out) < S.SR.dpo_steps * S.SR.dpo_batch:
        print(f"  fewer pairs than {S.SR.dpo_steps} x {S.SR.dpo_batch}: the DPO trainer will "
              f"cycle through them more than once")
    print(f"[saved] {a.out}\n[saved] {trl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
