#!/usr/bin/env python3
"""
make_extra_row_from_excise.py — put your detect -> excise -> re-review results (rl_excise's
excise_detail.csv) in the SAME table as SafeReview, computed the same way.

    python make_extra_row_from_excise.py --model gemma3-27b-it \
        --excise $AD_ROOT/defense/lim_agents_v7_RL/excise/gemma3-27b-it/G_grpo_prov/excise_detail.csv \
        --label "GRPO Prov Reward + excision"
    # then, as it prints:
    python sr_report.py --model gemma3-27b-it --extra-row "<label>|gemma3-27b-it|<out csv>"

WHY THIS IS NEEDED
    rl_excise.summarise() reports restoration / residual / invariance over DETECTED attacks
    only, and takes r_clean / r_atk from the attack benchmark's CSV (base_rating / rating)
    unless --rereview-clean was set. SafeReview's numbers are over ALL attacked papers, with
    R_0 / R_A from the undefended reviewer (rl_excise's own reviewer prompt, re-run). Put
    side by side, the two would compare different quantities.

    This script gives every test document its FINAL rating under your defence:
        excised ("delete")              -> sanitized_rating (rl_excise's re-review)
        untouched / refused / not found -> the undefended reviewer's rating of the same,
                                           unchanged document (SR_undefended; same prompt,
                                           temperature 0, so it is what rl_excise.review
                                           would return)
    plus your detector's decision (pred) and span F1. sr_report.py then scores it with the
    SafeReview rows: same R_0, same R_A, same denominators.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import sr_config as S
import sr_data as D


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--excise", required=True, help="rl_excise.py's excise_detail.csv")
    ap.add_argument("--label", default="Detect + excise + re-review")
    ap.add_argument("--set", dest="set_name", default="main")
    ap.add_argument("--run-tag", dest="run_tag", default="")
    ap.add_argument("--undefended", default="",
                    help="SR_undefended review_detail.csv (default: runs/<model>/SR_undefended...)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    ud_path = a.undefended or os.path.join(
        S.run_dir(a.model, "SR_undefended", a.run_tag, a.set_name), "review_detail.csv")
    for p in (ud_path, a.excise):
        if not os.path.exists(p):
            raise SystemExit(f"[FATAL] {p} not found")
    ud = pd.read_csv(ud_path, low_memory=False)
    ex = pd.read_csv(a.excise, low_memory=False)
    need = {"doc_id", "label", "mode", "sanitized_rating"}
    if need - set(ex.columns):
        raise SystemExit(f"[FATAL] {a.excise} lacks {sorted(need - set(ex.columns))}; is it "
                         f"rl_excise.py's excise_detail.csv?")
    for c in ("pred", "span_f1", "paper_id"):
        if c not in ex.columns:
            ex[c] = np.nan
    ex["doc_id"] = ex["doc_id"].astype(str)
    ud["doc_id"] = ud["doc_id"].astype(str)
    cols = ["mode", "sanitized_rating", "pred", "span_f1"]
    dup = ex["doc_id"].duplicated().sum()
    if dup:
        print(f"[warn] {dup} duplicate doc_id rows in the excision file; keeping the first")
    m = ud.merge(ex.drop_duplicates("doc_id")[["doc_id"] + cols], on="doc_id", how="left")

    ins = m["input_source"] == "insertion"
    cov = m.loc[ins, "mode"].notna().mean() if ins.any() else 0.0
    how = "doc_id"
    if cov < 0.9 and ex["paper_id"].notna().any():
        # different doc_id spellings for the same document: join on (paper, clean/attacked),
        # which is unique on the paired insertion corpus (one clean + one attacked per paper)
        ex["_pid"] = ex["paper_id"].astype(str).map(D.raw_pid)
        e2 = ex.drop_duplicates(["_pid", "label"])[["_pid", "label"] + cols]
        alt = ud.assign(_pid=ud["paper_id"].astype(str).map(D.raw_pid)).merge(
            e2, on=["_pid", "label"], how="left")
        alt.loc[~(alt["input_source"] == "insertion"), cols] = np.nan
        cov2 = alt.loc[ins, "mode"].notna().mean() if ins.any() else 0.0
        if cov2 > cov:
            m, cov, how = alt.drop(columns="_pid"), cov2, "(paper_id, label)"
    print(f"[join] {how}: {int(m['mode'].notna().sum())} of {len(m)} test documents have an "
          f"excision record ({100*cov:.1f}% of insertion documents)")
    if cov < 0.9:
        print("[warn] under 90% of the insertion documents matched. Check that the excision "
              "run used the same test corpus (the 990 published insertion documents).")

    mode = m["mode"].astype(str)
    m["R_D"] = np.where(mode == "delete", pd.to_numeric(m["sanitized_rating"], errors="coerce"),
                        np.where(m["mode"].notna(), m["rating"], np.nan))
    m["R_D_source"] = np.where(mode == "delete", "re-review after excision",
                               np.where(m["mode"].notna(), "unchanged document (undefended)",
                                        "not in the excision run"))
    out = a.out or S.wpath("extra_rows", S.mkey(a.model),
                           "".join(ch if ch.isalnum() else "_" for ch in a.label).strip("_")
                           + (f"__{a.set_name}" if a.set_name != "main" else "")
                           + (f"__{a.run_tag}" if a.run_tag else "") + ".csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    m[["eval_index", "doc_id", "label", "input_source", "R_D", "pred", "span_f1", "mode",
       "R_D_source"]].to_csv(out, index=False)

    att, cln = m[m["label"] == 1], m[m["label"] == 0]
    print(f"[rows]  attacked: {att['R_D'].notna().sum()} with a final rating "
          f"({int((att['mode'] == 'delete').sum())} excised, "
          f"{int(att['pred'].fillna(0).sum())} detected)")
    print(f"        clean:    {cln['R_D'].notna().sum()} with a final rating "
          f"({int((cln['mode'] == 'delete').sum())} excised = false alarms)")
    print(f"        missing sanitized ratings (re-review failed): "
          f"{int(((mode == 'delete') & m['R_D'].isna()).sum())}")
    print(f"[saved] {out}\n\nNext:\n    python sr_report.py --model {a.model}"
          + (f" --set {a.set_name}" if a.set_name != "main" else "")
          + (f" --run-tag {a.run_tag}" if a.run_tag else "")
          + f' \\\n        --extra-row "{a.label}|{a.model}|{out}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())