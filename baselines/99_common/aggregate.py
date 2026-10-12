#!/usr/bin/env python3
"""
aggregate.py — read every baseline's output and write the two tables the paper needs.

    python aggregate.py

OUTPUT
    FINAL_BASELINE_TABLE.txt    the tables, formatted
    baseline_table.csv          the same numbers, machine-readable
    baseline_table.tex          booktabs, ready to paste

THE TABLE IS SPLIT BY FAMILY ON PURPOSE
    Table 1: detection on INSTRUCTION-bearing attacks
    Table 2: detection on RHETORICAL attacks
    Table 3: mitigation (D2) -- different columns entirely, because it has no
             confusion matrix

    A single pooled table would let a reader average a detector that works on one family
    with one that is blind to the other, and report the mean as if it were a capability.
    The gap between Table 1 and Table 2 is the result.

WHAT GETS FLAGGED AUTOMATICALLY
    * any row whose FPR is above 0.10 -- on 500 honest manuscripts that is 50+ accused
      authors, and no F1 makes that acceptable
    * any row whose recall is at or below the flag-everything floor
    * any detector whose AUROC is below 0.5, which usually means a flipped label index
      rather than a weak model
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List

import pandas as pd

import common as C

ORDER = ["D1_flag_nothing", "D1_flag_everything", "D3_regex_only", "D3_ppl_only",
         "D3_combined", "D6_protectai_deberta", "D2_h4_checklist"]

CALLS = {"D1_flag_nothing": 0.0, "D1_flag_everything": 0.0, "D3_regex_only": 0.0,
         "D3_ppl_only": 0.0, "D3_combined": 0.0, "D6_protectai_deberta": 0.0,
         "D2_h4_checklist": 1.0}


def load_all(out_root: str) -> Dict[str, Dict]:
    """Every metrics*.json any baseline wrote, keyed by a readable name."""
    found: Dict[str, Dict] = {}
    for p in sorted(glob.glob(os.path.join(out_root, "*", "metrics*.json"))):
        base = os.path.basename(os.path.dirname(p))
        tag = os.path.basename(p)[len("metrics"):-len(".json")].strip("_")
        name = f"{base.split('_')[0]}_{tag}" if tag else base
        try:
            found[name] = json.load(open(p))
        except Exception as e:
            print(f"[warn] {p}: {type(e).__name__}: {e}")
    return found


def table(R, metrics: Dict[str, Dict], slice_key: str, title: str) -> List[dict]:
    cols = [("n_pos", "n_pos", "{:.0f}"), ("recall", "Recall", "{:.3f}"),
            ("precision", "Prec", "{:.3f}"), ("f1", "F1", "{:.3f}"),
            ("fpr", "FPR", "{:.3f}"), ("harm_recall", "HarmRec", "{:.3f}"),
            ("recall_at_fpr05", "R@FPR5", "{:.3f}"),
            ("recall_at_fpr10", "R@FPR10", "{:.3f}"),
            ("auroc", "AUROC", "{:.3f}"), ("accuracy", "Acc", "{:.3f}"),
            ("escalation", "Esc", "{:.3f}")]
    R("")
    R(f"--- {title} " + "-" * max(0, 80 - len(title)))
    hdr = f"{'baseline':24s}" + "".join(f"{h:>9s}" for _, h, _ in cols) + f"{'Calls':>8s}"
    R(hdr); R("-" * len(hdr))
    rows = []
    names = [n for n in ORDER if n in metrics] + \
            [n for n in sorted(metrics) if n not in ORDER]
    for name in names:
        m = metrics[name].get(slice_key)
        if not m:
            continue
        line = f"{name:24s}"
        rec = dict(baseline=name, slice=slice_key)
        for key, _, fmt in cols:
            v = m.get(key)
            rec[key] = v
            line += (fmt.format(v).rjust(9) if isinstance(v, (int, float)) and v == v
                     else f"{'-':>9s}")
        calls = CALLS.get(name, m.get("calls_per_paper", 0.0))
        rec["calls_per_paper"] = calls
        line += f"{calls:8.2f}"
        R(line)
        rows.append(rec)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=C.OUT_ROOT)
    ap.add_argument("--final-dir", default=C.DEF_ROOT)
    a = ap.parse_args()

    R = C.Report("AGGREGATE", a.final_dir)
    metrics = load_all(a.out_root)
    if not metrics:
        R(f"[FATAL] no metrics*.json under {a.out_root}")
        R("        Run d1/d3/d6 first; each writes its own metrics json.")
        R.close()
        return 1
    R(f"[load] {len(metrics)} baseline result set(s): {sorted(metrics)}")

    rows: List[dict] = []
    rows += table(R, metrics, "instruction",
                  "TABLE 1 — detection, INSTRUCTION-bearing attacks (A5-A9, A11, A12)")
    rows += table(R, metrics, "rhetorical",
                  "TABLE 2 — detection, RHETORICAL attacks (A1-A4, A10; no imperative)")
    rows += table(R, metrics, "control",
                  "TABLE 3 — length-matched filler (A0). Every flag here is a FALSE ALARM")
    rows += table(R, metrics, "ALL", "TABLE 4 — pooled. Reference only; do not cite.")

    # ---- the comparison the paper turns on --------------------------------------------
    R("")
    R("=" * 84)
    R("THE GAP: recall on instruction minus recall on rhetorical")
    R("=" * 84)
    R(f"{'baseline':24s}{'instr':>10s}{'rhet':>10s}{'gap':>10s}")
    R("-" * 54)
    for name, m in metrics.items():
        i = (m.get("instruction") or {}).get("recall")
        r = (m.get("rhetorical") or {}).get("recall")
        if i is None or r is None or i != i or r != r:
            continue
        R(f"{name:24s}{i:10.3f}{r:10.3f}{i-r:+10.3f}")
    R("")
    R("A large positive gap means the baseline covers one of your two threat families and")
    R("not the other. That is the sentence your related-work section is built on: the")
    R("cheap detectors everyone already uses do not see suppression attacks.")

    # ---- automatic problem flags -------------------------------------------------------
    R("")
    R("=" * 84)
    R("FLAGS")
    R("=" * 84)
    probs = []
    for name, m in metrics.items():
        for sl in ("instruction", "rhetorical", "ALL"):
            b = m.get(sl)
            if not b:
                continue
            if isinstance(b.get("fpr"), float) and b["fpr"] > 0.10:
                probs.append(f"{name} / {sl}: FPR {b['fpr']:.3f}. On 500 honest papers "
                             f"that is ~{b['fpr']*500:.0f} accused authors.")
            if isinstance(b.get("auroc"), float) and b["auroc"] == b["auroc"] \
                    and b["auroc"] < 0.5:
                probs.append(f"{name} / {sl}: AUROC {b['auroc']:.3f} is BELOW chance. "
                             f"Usually a flipped positive-label index, not a weak model.")
            if isinstance(b.get("recall"), float) and b["recall"] == 0.0 \
                    and b.get("n_pos", 0) > 0:
                probs.append(f"{name} / {sl}: recall 0.000 — detected nothing at all.")
    if probs:
        for p_ in dict.fromkeys(probs):
            R(f"  - {p_}")
    else:
        R("  none")

    # ---- write ---------------------------------------------------------------------
    df = pd.DataFrame(rows)
    R.save(df, "baseline_table.csv")

    tex = [r"\begin{table}[t]\centering\small",
           r"\caption{Defense baselines. Recall at a fixed false-positive budget; "
           r"detection only. D2 is a mitigation and appears separately.}",
           r"\label{tab:def-baselines}",
           r"\begin{tabular}{llrrrrrr}", r"\toprule",
           r"Family & Baseline & Recall & FPR & HarmRec & R@FPR5 & AUROC & Calls \\",
           r"\midrule"]
    for sl, lab in (("instruction", "Instruction"), ("rhetorical", "Rhetorical")):
        first = True
        for _, r_ in df[df["slice"] == sl].iterrows():
            def f(x):
                return f"{x:.3f}" if isinstance(x, (int, float)) and x == x else "--"
            tex.append(f"{lab if first else ''} & {r_['baseline'].replace('_',' ')} & "
                       f"{f(r_.get('recall'))} & {f(r_.get('fpr'))} & "
                       f"{f(r_.get('harm_recall'))} & {f(r_.get('recall_at_fpr05'))} & "
                       f"{f(r_.get('auroc'))} & {r_.get('calls_per_paper',0):.1f} \\\\")
            first = False
        tex.append(r"\midrule")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    p = os.path.join(a.final_dir, "baseline_table.tex")
    open(p, "w").write("\n".join(tex) + "\n")
    R(f"[saved] {p}")

    R.close()
    # rename the aggregate transcript to the name the paper refers to
    src = os.path.join(a.final_dir, "output.txt")
    dst = os.path.join(a.final_dir, "FINAL_BASELINE_TABLE.txt")
    try:
        os.replace(src, dst)
        print(f"[ok] {dst}")
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())