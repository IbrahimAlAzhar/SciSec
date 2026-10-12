#!/usr/bin/env python3
"""
sr_eval.py — score ONE row on ONE test set: every document (clean and attacked) reviewed
by that row's reviewer, one rating per document.

    python sr_eval.py --model gemma3-27b-it --row SR_undefended --served $MODEL_DIR
    python sr_eval.py --model gemma3-27b-it --row SR_sysdef     --served $MODEL_DIR
    python sr_eval.py --model gemma3-27b-it --row SR_static_dpo --served gemma3-27b-it-sr_static_dpo
    python sr_eval.py --model gemma3-27b-it --row SR_safereview --round 3 \
                      --served gemma3-27b-it-sr_r3
    ... --set adaptive_r3        the adaptive (GRPO-attacker) test set
    ... --run-tag pilot5         inputs/<model>/eval_records__pilot5.jsonl

Writes runs/<model>/<ROW>[__set-<SET>][__tag-<TAG>]/
    review_detail.csv    one row per document, keyed by eval_index (the row number of the
                         records file, exactly as runner_def's stage1_detail.csv)
    reviews.jsonl        the raw reviews (the transcripts)
    review_summary.json

SafeReview flags nothing, so it writes no stage1_detail.csv: its detection row is "n/a".
sr_report.py joins these files on eval_index with the same eval_index.csv v7ab_report.py
uses, so every number is reported in total AND per source.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import pandas as pd

import sr_config as S
import sr_data as D
import sr_review as R
from sr_vllm import require_served


def main() -> int:
    ap = argparse.ArgumentParser(description="evaluate one SafeReview row")
    ap.add_argument("--model", required=True)
    ap.add_argument("--row", required=True, choices=[r.key for r in S.ROWS if r.kind == "review"])
    ap.add_argument("--round", type=int, default=None, help="SR_safereview: which round")
    ap.add_argument("--served", required=True)
    ap.add_argument("--adapter-dir", dest="adapter_dir", default="")
    ap.add_argument("--vllm-url", dest="vllm_url", default=os.environ.get("VLLM_URL", ""))
    ap.add_argument("--api-key-env", dest="api_key_env", default="",
                    help="read the bearer key from this env var (gpt-4o-mini: OPENAI_API_KEY)")
    ap.add_argument("--set", dest="set_name", default="main")
    ap.add_argument("--run-tag", dest="run_tag", default="")
    ap.add_argument("--max-docs", dest="max_docs", type=int, default=0)
    R.add_reviewer_args(ap)
    a = ap.parse_args()

    row = S.ROW[a.row]
    if row.needs_weights and a.model in S.API_MODELS:
        raise SystemExit(f"[FATAL] {a.row} retrains the reviewer's weights; {a.model} is "
                         f"API-only. This is the limitation the comparison is about: the "
                         f"row is reported as n/a for {a.model}.")
    if a.row == "SR_safereview" and a.round is None:
        raise SystemExit("[FATAL] SR_safereview needs --round")
    if not a.vllm_url:
        raise SystemExit("[FATAL] --vllm-url is required")
    rec, idx = S.inputs_paths(a.model, a.run_tag, a.set_name)
    te = D.load_test(rec, idx)
    if a.max_docs:
        te = te.head(a.max_docs)
    out_dir = S.run_dir(a.model, S.row_dir_name(a.row, a.round), a.run_tag, a.set_name)
    os.makedirs(out_dir, exist_ok=True)

    api_key = os.environ.get(a.api_key_env, "") if a.api_key_env else ""
    rv = R.make_reviewer(a, model_key=a.model, url=a.vllm_url, served=a.served,
                         adapter_dir=a.adapter_dir, variant=row.prompt, api_key=api_key)
    require_served(rv.cl, a.served)
    print(rv.describe())
    print(f"[eval] {a.row}{'' if a.round is None else f' round {a.round}'} on {len(te)} "
          f"documents of {rec}")
    t0 = time.time()
    res = rv.review_many(list(te["document"]), workers=a.workers, label=a.row)

    det = te[["eval_index", "doc_id", "paper_id", "pair_uid", "label", "condition",
              "input_source", "threat_family", "seen_in_train"]].copy()
    det["rating"] = [r["rating"] for r in res]
    det["n_ratings"] = [len(r.get("ratings") or []) for r in res]
    det["parse_ok"] = [int(bool(r["parse_ok"])) for r in res]
    det["n_llm_calls"] = [r["n_calls"] for r in res]
    det["cached"] = [int(bool(r.get("cached"))) for r in res]
    det["weaknesses"] = [str(r.get("weaknesses") or "")[:600] for r in res]
    det["row"] = a.row if a.round is None else f"{a.row}_r{a.round}"
    det["served"] = a.served
    det["fingerprint"] = rv.fingerprint
    det["prompt_sha1"] = rv.sys_hash
    det.to_csv(os.path.join(out_dir, "review_detail.csv"), index=False)
    with open(os.path.join(out_dir, "reviews.jsonl"), "w", encoding="utf-8") as f:
        for (_, r), x in zip(te.iterrows(), res):
            f.write(json.dumps({"eval_index": int(r["eval_index"]), "doc_id": r["doc_id"],
                                "label": int(r["label"]), "rating": x["rating"],
                                "raw": x.get("raw", ""), "error": x.get("error", "")},
                               ensure_ascii=False) + "\n")

    def m(x):
        x = pd.to_numeric(x, errors="coerce").dropna()
        return float(x.mean()) if len(x) else None

    summ = {"model": a.model, "row": det["row"].iloc[0], "served": a.served,
            "set": a.set_name, "run_tag": a.run_tag, "n_docs": len(det),
            "parse_fail_rate": float(1 - det["parse_ok"].mean()),
            "mean_rating_clean": m(det.loc[det["label"] == 0, "rating"]),
            "mean_rating_attacked": m(det.loc[det["label"] == 1, "rating"]),
            "reviewer": rv.describe(), "minutes": (time.time() - t0) / 60, "per_source": {}}
    for src, g in det.groupby("input_source"):
        summ["per_source"][src] = {"n_docs": len(g),
                                   "clean": m(g.loc[g["label"] == 0, "rating"]),
                                   "attacked": m(g.loc[g["label"] == 1, "rating"])}
    with open(os.path.join(out_dir, "review_summary.json"), "w") as f:
        json.dump(summ, f, indent=2)
    print("\n" + "=" * 80)
    print(f"{summ['row']} — {a.model} — {a.set_name}{' / ' + a.run_tag if a.run_tag else ''}")
    print("=" * 80)
    print(f"  documents {len(det)}   parse failures {100*summ['parse_fail_rate']:.1f}%")
    print(f"  {'source':<16}{'docs':>6}{'R(clean)':>10}{'R(attacked)':>13}{'shift':>8}")
    for src, v in summ["per_source"].items():
        sh = (v["attacked"] - v["clean"]) if (v["clean"] is not None and v["attacked"] is not None) else None
        print(f"  {src:<16}{v['n_docs']:>6}{(v['clean'] or float('nan')):>10.2f}"
              f"{(v['attacked'] or float('nan')):>13.2f}{(sh if sh is not None else float('nan')):>+8.2f}")
    if summ["parse_fail_rate"] > 0.05:
        print("  MORE THAN 5% of reviews did not parse. A parse failure is a missing rating, "
              "and pairs with a missing rating are dropped from the rating panels.")
    print(f"[saved] {out_dir}/review_detail.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
