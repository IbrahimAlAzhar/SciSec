#!/usr/bin/env python3
"""
pg_table.py — the PIGuard rows, formatted like YOUR table.

Writes, under defense_baselines/piguard/runs/:
    PIGUARD_TABLE.txt          everything, readable
    piguard_panelA.tex         Method & N & Acc & Prec & Rec & F1 & FPR & HarmRec & Esc & Calls
    piguard_panelB.tex         per threat family: N_pos & Recall & Prec & F1 & FPR & HarmRec &
                               R@FPR5 & R@FPR10 & AUROC   (same conventions as your agent rows)
    piguard_localization.tex   where PIGuard says the attack is, against the gold diff
    piguard_rating.tex         clean / attacked / sanitised ratings and their deviation
    piguard_per_condition.csv  recall per attack condition (A1 ... A12)

--ablation-runs <llim_agents_v4_SFT_DPO>/runs  additionally prints your SFT/DPO rows
(from runs/<model>/<ablation>/)
beside the PIGuard rows, with Panel B recomputed from their stage1_detail.csv by the same code.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from typing import Dict, List, Optional

import pandas as pd

import pg_common as PC           # first: puts the SFT/DPO directory on sys.path
import ft_config as C

ABL_LABEL = {a.key: a.label for a in C.ABLATIONS}
ORDER = ["piguard_released_trunc", "piguard_released_full", "piguard_released_region",
         "piguard_nomof_full", "piguard_mof_full", "piguard_mof_region"]


def f(v, nd=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "---"
    return f"{v:.{nd}f}"


def tex(s: str) -> str:
    return str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")


def load_rows(detail: str) -> List[Dict]:
    d = pd.read_csv(detail, low_memory=False, dtype={"paper_id": str, "doc_id": str})
    return d.to_dict("records")


def collect(runs_dir: str, model: str) -> Dict[str, Dict]:
    out = {}
    for p in glob.glob(os.path.join(runs_dir, model, "*", "summary.json")):
        s = json.load(open(p))
        s["_dir"] = os.path.dirname(p)
        out[s.get("ablation") or os.path.basename(os.path.dirname(p))] = s
    return out


def sort_keys(keys):
    return sorted(keys, key=lambda k: (ORDER.index(k) if k in ORDER else 100, k))


def main() -> int:
    p = argparse.ArgumentParser(description="PIGuard tables in the format of your paper")
    p.add_argument("--runs-dir", dest="runs_dir", default=PC.runs_dir())
    p.add_argument("--ablation-runs", dest="ablation_runs", default="",
                   help="your runs/ directory (SFT/DPO ablations) to print alongside")
    p.add_argument("--models", nargs="*", default=None)
    a = p.parse_args()

    models = a.models or sorted(d for d in os.listdir(a.runs_dir)
                                if os.path.isdir(os.path.join(a.runs_dir, d)))
    T: List[str] = []
    A_tex, B_tex, L_tex, R_tex = [], {"instruction": [], "rhetorical": []}, [], []
    per_cond = []
    W = 128
    for model in models:
        pg = collect(a.runs_dir, model)
        ab = collect(a.ablation_runs, model) if a.ablation_runs else {}
        if not pg and not ab:
            continue
        T += ["=" * W, f"REVIEWER / EVALUATION CORPUS: {model}", "=" * W,
              "PANEL A", f"  {'Method':<52} {'N':>5} {'Acc':>6} {'Prec':>6} {'Rec':>6} "
              f"{'F1':>6} {'FPR':>6} {'HarmRec':>8} {'Esc%':>6} {'Calls':>6} {'Guard/doc':>9}"]
        A_tex.append(f"% ---- {model}")
        known = [x.key for x in C.ABLATIONS]
        entries = [(k, pg[k], "pg") for k in sort_keys(pg)] + \
                  [(k, ab[k], "ab") for k in known if k in ab] + \
                  [(k, ab[k], "ab") for k in sorted(ab) if k not in known]
        for key, s, kind in entries:
            m, h = s.get("all") or {}, s.get("harmful") or {}
            label = s.get("label") or ABL_LABEL.get(key, key)
            T.append(f"  {label[:52]:<52} {m.get('n', 0):>5} {f(m.get('accuracy')):>6} "
                     f"{f(m.get('precision')):>6} {f(m.get('recall')):>6} {f(m.get('f1')):>6} "
                     f"{f(m.get('fpr')):>6} {f(h.get('recall')):>8} "
                     f"{f(m.get('escalated_pct'), 1):>6} {f(m.get('mean_llm_calls'), 2):>6} "
                     f"{f(s.get('mean_guard_passes'), 1):>9}")
            if kind == "pg":
                A_tex.append(f"{tex(label)} & {m.get('n', 0)} & {f(m.get('accuracy'))} & "
                             f"{f(m.get('precision'))} & {f(m.get('recall'))} & "
                             f"{f(m.get('f1'))} & {f(m.get('fpr'))} & {f(h.get('recall'))} & "
                             f"{f(m.get('escalated_pct'), 1)} & "
                             f"{f(m.get('mean_llm_calls'), 2)} \\\\")

        T += ["", "PANEL B (positives = attacked docs of the family; negatives = ALL clean docs)",
              f"  {'Family':<12} {'Method':<44} {'Npos':>5} {'Rec':>6} {'Prec':>6} {'F1':>6} "
              f"{'FPR':>6} {'HarmRec':>8} {'R@FPR5':>7} {'R@FPR10':>8} {'AUROC':>6}"]
        for g in ("instruction", "rhetorical"):
            B_tex[g].append(f"% ---- {model}")
            for key, s, kind in entries:
                label = s.get("label") or ABL_LABEL.get(key, key)
                det = os.path.join(s["_dir"], "stage1_detail.csv")
                if kind == "pg" and "panel_b" in s:
                    b = s["panel_b"][g]
                elif os.path.exists(det):
                    rows = load_rows(det)
                    # the agent rows have no calibrated score: score columns stay '---'
                    b = PC.panel_b(rows, g, score_key="__none__")
                else:
                    continue
                T.append(f"  {g:<12} {label[:44]:<44} {b['n_pos']:>5} {f(b['recall']):>6} "
                         f"{f(b['precision']):>6} {f(b['f1']):>6} {f(b['fpr']):>6} "
                         f"{f(b['harm_recall']):>8} {f(b['r_at_fpr5']):>7} "
                         f"{f(b['r_at_fpr10']):>8} {f(b['auroc']):>6}")
                if kind == "pg":
                    B_tex[g].append(f"{tex(label)} & {b['n_pos']} & {f(b['recall'])} & "
                                    f"{f(b['precision'])} & {f(b['f1'])} & {f(b['fpr'])} & "
                                    f"{f(b['harm_recall'])} & {f(b['r_at_fpr5'])} & "
                                    f"{f(b['r_at_fpr10'])} & {f(b['auroc'])} \\\\")

        # ---- localisation -------------------------------------------------------------
        T += ["", "LOCALISATION (attacked docs PIGuard detected; char-level vs the gold diff)",
              f"  {'Method':<44} {'group':<12} {'det/att':>9} {'hit':>6} {'grndRec':>8} "
              f"{'P':>6} {'R':>6} {'IoU':>6} {'rm chars':>9} {'gold ch':>8} {'resolved':>8}"]
        for key in sort_keys(pg):
            s = pg[key]
            L = s.get("localization") or {}
            for g in ("all", "instruction", "rhetorical"):
                x = L.get(g) or {}
                if not x:
                    continue
                T.append(f"  {(s.get('label') or key)[:44]:<44} {g:<12} "
                         f"{x.get('n_detected', 0):>4}/{x.get('n_attacked', 0):<4} "
                         f"{f(x.get('hit_rate_detected')):>6} {f(x.get('grounded_recall')):>8} "
                         f"{f(x.get('precision')):>6} {f(x.get('recall')):>6} "
                         f"{f(x.get('iou')):>6} {f(x.get('mean_removed_chars'), 0):>9} "
                         f"{f(x.get('mean_gold_chars'), 0):>8} {f(x.get('resolved_rate')):>8}")
            x = L.get("all") or {}
            c = L.get("clean") or {}
            if x:
                L_tex.append(f"{tex(model)} & {tex(s.get('label') or key)} & "
                             f"{x.get('n_detected', 0)}/{x.get('n_attacked', 0)} & "
                             f"{f(x.get('hit_rate_detected'))} & "
                             f"{f(x.get('grounded_recall'))} & {f(x.get('precision'))} & "
                             f"{f(x.get('recall'))} & {f(x.get('iou'))} & "
                             f"{f(x.get('mean_removed_chars'), 0)} & "
                             f"{f(c.get('mean_removed_chars_flagged'), 0)} \\\\")

        # ---- rating deviation ---------------------------------------------------------
        T += ["", "RATING DEVIATION (same reviewer, prompt, T=0, seed for every version)",
              f"  {'Method':<40} {'subset':<12} {'n':>4} {'r_clean':>7} {'r_att':>7} "
              f"{'r_san':>7} {'d_att':>7} {'d_san':>7} {'MAE_att':>7} {'MAE_san':>7} "
              f"{'recov':>6} {'raise_att':>9} {'raise_san':>9}"]
        for key in sort_keys(pg):
            rp = os.path.join(pg[key]["_dir"], "rating_summary.json")
            if not os.path.exists(rp):
                continue
            R = json.load(open(rp))
            lab = pg[key].get("label") or key
            for sub in ("all", "detected", "missed", "harmful", "instruction", "rhetorical"):
                x = R.get(sub) or {}
                if not x.get("n"):
                    continue
                T.append(f"  {lab[:40]:<40} {sub:<12} {x['n']:>4} {f(x['r_clean'], 2):>7} "
                         f"{f(x['r_attacked'], 2):>7} {f(x['r_sanitized'], 2):>7} "
                         f"{f(x['d_attack'], 2):>7} {f(x['d_sanitized'], 2):>7} "
                         f"{f(x['mae_attack'], 2):>7} {f(x['mae_sanitized'], 2):>7} "
                         f"{f(x['recovery'], 2):>6} {f(x['raised_attack'], 3):>9} "
                         f"{f(x['raised_sanitized'], 3):>9}")
            cl = R.get("clean") or {}
            T.append(f"  {lab[:40]:<40} {'clean FP':<12} {cl.get('n_flagged', 0):>4} flagged "
                     f"of {cl.get('n', 0)} clean papers; mean rating change on them "
                     f"{f(cl.get('mean_d_fp_flagged'), 3)} (MAE {f(cl.get('mae_fp_flagged'), 3)})")
            x = R.get("all") or {}
            if x.get("n"):
                R_tex.append(f"{tex(model)} & {tex(lab)} & {x['n']} & {f(x['r_clean'], 2)} & "
                             f"{f(x['r_attacked'], 2)} & {f(x['r_sanitized'], 2)} & "
                             f"{f(x['d_attack'], 2)} & {f(x['d_sanitized'], 2)} & "
                             f"{f(x['mae_attack'], 2)} & {f(x['mae_sanitized'], 2)} & "
                             f"{f(x['recovery'], 2)} & {cl.get('n_flagged', 0)} & "
                             f"{f(cl.get('mean_d_fp_flagged'), 2)} \\\\")

        # ---- per condition ------------------------------------------------------------
        for key in sort_keys(pg):
            det = os.path.join(pg[key]["_dir"], "stage1_detail.csv")
            if not os.path.exists(det):
                continue
            d = pd.read_csv(det, low_memory=False)
            att = d[d["label"] == 1]
            for cond, g in att.groupby("condition"):
                per_cond.append({"model": model, "variant": key, "condition": cond,
                                 "family_group": PC.family_group(cond), "n": len(g),
                                 "recall": g["pred"].mean(),
                                 "mean_score": g["screen_score"].mean(),
                                 "loc_hit": g["loc_hit"].mean() if "loc_hit" in g else None})
        T.append("")

    T += ["=" * W, "NOTES", "=" * W,
          "  * Same 990-document corpus and metric code as the SFT/DPO rows (infer_ladder).",
          "  * PIGuard makes no LLM call: Esc% = 0 and Calls = 0. 'Guard/doc' is the number of",
          "    DeBERTa forward passes per document (windows + localisation).",
          "  * Panel B precision = TP_family / (TP_family + FP on all clean docs), as in your",
          "    agent rows; R@FPR and AUROC use PIGuard's continuous max-window P(injection).",
          "  * Localisation hit = the removed text overlaps the gold attack span; grounded recall",
          "    = attacked docs that were detected AND localised correctly / all attacked docs.",
          "  * A3_concealment and A10 LOWER ratings in the benchmark; read HarmRec for the",
          "    harm-relevant recall."]
    os.makedirs(a.runs_dir, exist_ok=True)
    out = os.path.join(a.runs_dir, "PIGUARD_TABLE.txt")
    open(out, "w").write("\n".join(T) + "\n")
    open(os.path.join(a.runs_dir, "piguard_panelA.tex"), "w").write(
        "% Method & N & Acc & Prec & Rec & F1 & FPR & HarmRec & Esc (%) & Calls\n"
        + "\n".join(A_tex) + "\n")
    open(os.path.join(a.runs_dir, "piguard_panelB.tex"), "w").write(
        "% Method & N_pos & Recall & Prec & F1 & FPR & HarmRec & R@FPR5 & R@FPR10 & AUROC\n"
        "% \\multicolumn{10}{l}{\\textit{Instruction-bearing attacks: A5--A9, A11, A12}} \\\\\n"
        + "\n".join(B_tex["instruction"]) +
        "\n% \\multicolumn{10}{l}{\\textit{Rhetorical suppression attacks: A1--A4, A10}} \\\\\n"
        + "\n".join(B_tex["rhetorical"]) + "\n")
    open(os.path.join(a.runs_dir, "piguard_localization.tex"), "w").write(
        "% Reviewer & Method & Det/Att & Hit & Grounded Rec & Char P & Char R & IoU & "
        "Removed chars (TP) & Removed chars (FP clean)\n" + "\n".join(L_tex) + "\n")
    open(os.path.join(a.runs_dir, "piguard_rating.tex"), "w").write(
        "% Reviewer & Method & N pairs & r_clean & r_att & r_san & d_att & d_san & MAE_att & "
        "MAE_san & Recovery & #FP clean & d on FP clean\n" + "\n".join(R_tex) + "\n")
    if per_cond:
        pd.DataFrame(per_cond).to_csv(os.path.join(a.runs_dir, "piguard_per_condition.csv"),
                                      index=False)
    print("\n".join(T))
    print(f"\n[saved] {out} and piguard_*.tex in {a.runs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
