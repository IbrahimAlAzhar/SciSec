#!/usr/bin/env python3
"""
aggregate_d7.py -- your existing baseline table (D1/D3/D6/D2-h4) with the D7 LlamaFirewall
rows added, plus D7's rating-change table. Nothing in the parent folder is modified.

    python aggregate_d7.py
    python aggregate_d7.py --parent-out-root ../output --d7-out-root ./output

OUTPUT (in llama_firewall/output/)
    FINAL_BASELINE_TABLE_with_D7.txt   tables 1-4 as in aggregate.py, D7 rows included,
                                       THE GAP, FLAGS, and the D7 rating-change table
    baseline_table_with_D7.csv         the same detection numbers, machine-readable
    baseline_table_with_D7.tex         booktabs (D7 replay rows; add --tex-live for live)
    d7_rating_change_table.csv         rating change / ASR before and after, per config

ROW NAMES
    D7_replay_<config>   offline replay of the stored reviews: same papers, same split,
                         same reviewer as D1/D3/D6. These are the rows to compare.
    D7_live_<config>     the guarded agent writing new reviews on a paper subset. Its n
                         differs; compare it with itself (with vs without firewall).
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
for p in (HERE, PARENT):
    if p not in sys.path:
        sys.path.insert(0, p)

import common as C                                                  # noqa: E402
import aggregate as AG                                              # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent-out-root", default=C.OUT_ROOT,
                    help="where D1/D3/D6/D2 wrote their metrics*.json")
    ap.add_argument("--d7-out-root", default=os.path.join(HERE, "output"))
    ap.add_argument("--final-dir", default=os.path.join(HERE, "output"))
    ap.add_argument("--tex-live", action="store_true")
    a = ap.parse_args()

    R = C.Report("AGGREGATE_with_D7", a.final_dir)
    base = AG.load_all(a.parent_out_root) if os.path.isdir(a.parent_out_root) else {}
    d7 = AG.load_all(a.d7_out_root)
    d7 = {k: v for k, v in d7.items() if k.startswith("D7_")}
    R(f"[load] existing baselines from {a.parent_out_root}: {sorted(base) or 'none'}")
    R(f"[load] D7 from {a.d7_out_root}: {sorted(d7) or 'none'}")
    if not d7:
        R("[FATAL] no D7 metrics found; run d7_llamafirewall.py --stage eval first")
        R.close()
        return 1
    metrics = {**base, **d7}
    for k in list(d7):
        if k not in AG.CALLS:
            AG.CALLS[k] = float((d7[k].get("ALL") or {}).get("calls_per_paper", 0.0) or 0.0)

    rows = []
    rows += AG.table(R, metrics, "instruction",
                     "TABLE 1 — detection, INSTRUCTION-bearing attacks (A5-A9, A11, A12)")
    rows += AG.table(R, metrics, "rhetorical",
                     "TABLE 2 — detection, RHETORICAL attacks (A1-A4, A10; no imperative)")
    rows += AG.table(R, metrics, "control",
                     "TABLE 3 — length-matched filler (A0). Every flag here is a FALSE ALARM")
    rows += AG.table(R, metrics, "ALL", "TABLE 4 — pooled. Reference only; do not cite.")

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
    R("=" * 84)
    R("FLAGS")
    R("=" * 84)
    probs = []
    for name, m in metrics.items():
        for sl in ("instruction", "rhetorical", "ALL"):
            b = m.get(sl)
            if not isinstance(b, dict):
                continue
            if isinstance(b.get("fpr"), float) and b["fpr"] > 0.10:
                probs.append(f"{name} / {sl}: FPR {b['fpr']:.3f}. On 500 honest papers "
                             f"that is ~{b['fpr']*500:.0f} accused authors.")
            if isinstance(b.get("auroc"), float) and b["auroc"] == b["auroc"] \
                    and b["auroc"] < 0.5:
                probs.append(f"{name} / {sl}: AUROC {b['auroc']:.3f} is below chance.")
            if isinstance(b.get("recall"), float) and b["recall"] == 0.0 \
                    and b.get("n_pos", 0) > 0:
                probs.append(f"{name} / {sl}: recall 0.000 — detected nothing at all.")
    for p_ in dict.fromkeys(probs):
        R(f"  - {p_}")
    if not probs:
        R("  none")

    # ---- D7 rating change -----------------------------------------------------------
    R("")
    R("=" * 84)
    R("D7 — WHAT THE FIREWALL DOES TO THE RATING (attacked documents)")
    R("=" * 84)
    parts = []
    for mode in ("replay", "live"):
        p = os.path.join(a.d7_out_root, f"D7_llamafirewall_{mode}", f"summary_{mode}.csv")
        if os.path.exists(p):
            parts.append(pd.read_csv(p))
    if parts:
        s = pd.concat(parts, ignore_index=True)
        R(f"{'mode':7s}{'config':18s}{'slice':13s}{'withheld':>9s}{'dR_atk':>8s}"
          f"{'dR_rel':>8s}{'dR_fb':>8s}{'ASR':>7s}{'ASRdef':>8s}{'util':>7s}")
        R("-" * 93)

        def f(x, w, sign=False):
            try:
                x = float(x)
            except Exception:
                return f"{'-':>{w}s}"
            if x != x:
                return f"{'-':>{w}s}"
            return f"{x:+{w}.3f}" if sign else f"{x:{w}.3f}"
        for _, r in s.iterrows():
            R(f"{r['mode']:7s}{r['config']:18s}{r['slice']:13s}{f(r.withheld_rate, 9)}"
              f"{f(r.d_rating_nodef, 8, True)}{f(r.d_rating_def_released, 8, True)}"
              f"{f(r.d_rating_def_fallback, 8, True)}{f(r.asr_nodef, 7)}"
              f"{f(r.asr_def, 8)}{f(r.utility_clean_released, 7)}")
        R("  dR_atk  rating change the attack causes with no defense")
        R("  dR_rel  rating change among reviews the firewall released")
        R("  dR_fb   whole pipeline, if withheld reviews are replaced by a correct human")
        R("          review (an upper bound on the benefit)")
        R("  util    share of CLEAN manuscripts whose review is released (1 - lost utility)")
        R.save(s, "d7_rating_change_table.csv")

    df = pd.DataFrame(rows)
    R.save(df, "baseline_table_with_D7.csv")

    keep = df[~df["baseline"].str.startswith("D7_live_")] if not a.tex_live else df
    tex = [r"\begin{table}[t]\centering\small",
           r"\caption{Defense baselines, including LlamaFirewall (PromptGuard~2 and "
           r"AlignmentCheck; Chennabasappa et al., 2025). Detection only.}",
           r"\label{tab:def-baselines-d7}",
           r"\begin{tabular}{llrrrrrrr}", r"\toprule",
           r"Family & Baseline & Prec & Recall & F1 & FPR & HarmRec & AUROC & Calls \\",
           r"\midrule"]
    for sl, lab in (("instruction", "Instruction"), ("rhetorical", "Rhetorical")):
        first = True
        for _, r_ in keep[keep["slice"] == sl].iterrows():
            def g(x):
                return f"{x:.3f}" if isinstance(x, (int, float)) and x == x else "--"
            tex.append(f"{lab if first else ''} & {r_['baseline'].replace('_', ' ')} & "
                       f"{g(r_.get('precision'))} & {g(r_.get('recall'))} & "
                       f"{g(r_.get('f1'))} & {g(r_.get('fpr'))} & "
                       f"{g(r_.get('harm_recall'))} & {g(r_.get('auroc'))} & "
                       f"{float(r_.get('calls_per_paper') or 0):.2f} \\\\")
            first = False
        tex.append(r"\midrule")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    p = os.path.join(a.final_dir, "baseline_table_with_D7.tex")
    with open(p, "w") as fh:
        fh.write("\n".join(tex) + "\n")
    R(f"[saved] {p}")
    R.close()
    src = os.path.join(a.final_dir, "output.txt")
    dst = os.path.join(a.final_dir, "FINAL_BASELINE_TABLE_with_D7.txt")
    try:
        os.replace(src, dst)
        print(f"[ok] {dst}")
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
