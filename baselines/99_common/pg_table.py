#!/usr/bin/env python3
"""
pg_table.py — PaperGuard next to the SFT/DPO rows, on the same documents.

WHAT IT PRINTS
=======================================================================================
  Panel A   Acc / Prec / Rec / F1 / FPR / HarmRec / Esc% / Calls for every SFT/DPO
            ablation found under --ft-runs-dir and every PaperGuard run under
            --pg-runs-dir, all recomputed from stage1_detail.csv with the formulas of
            infer_ladder.metrics — one function for every row, so a difference between
            rows is a difference between methods.
  MATCH     whether each PaperGuard file has the SAME ordered (paper_id, condition,
            label, harmful) sequence as each SFT/DPO file. If it does the comparison is
            head-to-head on identical documents, and an exact McNemar test on the paired
            per-document correctness is reported.
  Panel B   by threat family, in the layout of your Panel B: instruction-bearing
            (A5-A9, A11, A12) and rhetorical suppression (A1-A4, A10); negatives are all
            clean documents, shared by both families. R@FPR5, R@FPR10 and AUROC use the
            retrieval score pg_score (max chunk/pattern cosine), NOT the binary verdict.
  ABLATION  PaperGuard registry + our TRAIN attacks, reported in its own block against
            the PaperGuard baseline (deltas, McNemar, per-family recall change). It is
            never placed among the baselines, in the text table or in the LaTeX.
  per family recall, localization and re-rating (if pg_rerate.py has run).
  LaTeX     pg_rows.tex: baseline rows for both panels, then a separate ablation block.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import pg_config as P

FT_ORDER = ["zeroshot", "dpo_only", "sft", "sft_dpo", "sft_dpo_per_agent"]
FT_LABEL = {"zeroshot": "ZS LLM Agents (naive)",
            "dpo_only": "DPO only, no SFT",
            "sft": "SFT, single adapter",
            "sft_dpo": "SFT then DPO, single adapter",
            "sft_dpo_per_agent": "SFT+DPO per agent (5 adapters)"}
PG_PREFIX = [r.ablation for r in P.REGISTRIES.values()]    # baseline first, then ablation
INSTRUCTION = {5, 6, 7, 8, 9, 11, 12}
RHETORICAL = {1, 2, 3, 4, 10}


def fam_num(c) -> Optional[int]:
    m = re.match(r"A(\d+)", str(c))
    return int(m.group(1)) if m else None


def _div(a, b):
    return a / b if b else float("nan")


def panel_a(d: pd.DataFrame) -> Dict:
    y, p = d["label"].astype(int), d["pred"].astype(int)
    tp = int(((y == 1) & (p == 1)).sum()); fp = int(((y == 0) & (p == 1)).sum())
    tn = int(((y == 0) & (p == 0)).sum()); fn = int(((y == 1) & (p == 0)).sum())
    n = tp + fp + tn + fn
    prec, rec = _div(tp, tp + fp), _div(tp, tp + fn)
    f1 = (2 * prec * rec / (prec + rec)) if (prec == prec and rec == rec and prec + rec) \
        else 0.0
    h = d[(y == 0) | (d.get("harmful", 0).astype(int) == 1)]
    hy, hp = h["label"].astype(int), h["pred"].astype(int)
    harm_rec = _div(int(((hy == 1) & (hp == 1)).sum()), int((hy == 1).sum()))
    return {"N": n, "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "Acc": _div(tp + tn, n), "Prec": prec, "Rec": rec, "F1": f1,
            "FPR": _div(fp, fp + tn), "HarmRec": harm_rec,
            "Esc": 100.0 * d["escalated"].astype(float).mean() if "escalated" in d else float("nan"),
            "Calls": d["n_llm_calls"].astype(float).mean() if "n_llm_calls" in d else float("nan")}


def recall_at_fpr(y: np.ndarray, s: np.ndarray, target: float) -> float:
    from sklearn.metrics import roc_curve
    if len(np.unique(y)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= target + 1e-12
    return float(tpr[ok].max()) if ok.any() else 0.0


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def panel_b(d: pd.DataFrame, fams: set, score_col: str = "pg_score") -> Dict:
    fn_ = d["condition"].map(fam_num)
    sub = d[(d["label"] == 0) | fn_.isin(fams)]
    m = panel_a(sub)
    out = {"N_pos": int((sub["label"] == 1).sum()), "Recall": m["Rec"], "Prec": m["Prec"],
           "F1": m["F1"], "FPR": m["FPR"], "HarmRec": m["HarmRec"],
           "R@FPR5": float("nan"), "R@FPR10": float("nan"), "AUROC": float("nan")}
    if score_col in sub and sub[score_col].notna().any():
        y = sub["label"].astype(int).values
        s = sub[score_col].astype(float).fillna(-1.0).values
        out.update({"R@FPR5": recall_at_fpr(y, s, 0.05),
                    "R@FPR10": recall_at_fpr(y, s, 0.10), "AUROC": auroc(y, s)})
    return out


def mcnemar(a_correct: np.ndarray, b_correct: np.ndarray) -> Tuple[int, int, float]:
    b = int((a_correct & ~b_correct).sum())
    c = int((~a_correct & b_correct).sum())
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    try:
        from scipy.stats import binomtest
        pv = binomtest(k, n, 0.5).pvalue
    except Exception:
        pv = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return b, c, float(pv)


def load(dirpath: str) -> Optional[pd.DataFrame]:
    p = os.path.join(dirpath, "stage1_detail.csv")
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p, dtype={"paper_id": str, "doc_id": str}, low_memory=False)
    for c in ("label", "pred"):
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0).astype(int)
    if "harmful" not in d:
        d["harmful"] = 0
    d["harmful"] = pd.to_numeric(d["harmful"], errors="coerce").fillna(0).astype(int)
    return d


PG_LABEL: Dict[str, str] = {}


def pg_meta(pg_runs: str, ab: str) -> Tuple[str, str]:
    """(label, role) from the run's summary.json. Anything not declared a baseline is
    treated as an ablation, so an unlabelled run can never be promoted to a baseline."""
    try:
        s = json.load(open(os.path.join(pg_runs, ab, "summary.json")))
        return s.get("label") or ab, ("baseline" if s.get("role") == "baseline"
                                      else "ablation")
    except Exception:
        return ab, "ablation"


def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f"{v:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus-model", "--model", dest="model", default=P.CORPUS_MODEL,
                    help="the attacked corpus; selects <FT_CODE_DIR>/runs/<this>")
    ap.add_argument("--pg-runs-dir", dest="pg_runs", default="")
    ap.add_argument("--ft-runs-dir", dest="ft_runs", default="")
    ap.add_argument("--out-dir", dest="out_dir", default="")
    a = ap.parse_args()
    mk = a.model.replace("/", "_")
    pg_runs = a.pg_runs or P.runs_root()
    ft_runs = a.ft_runs or os.path.join(P.FT_CODE_DIR, "runs", mk)
    out_dir = a.out_dir or pg_runs
    os.makedirs(out_dir, exist_ok=True)

    # kind: "ft" = your SFT/DPO rows, "pg" = PaperGuard BASELINE, "abl" = ablation
    rows: List[Tuple[str, str, str, pd.DataFrame]] = []
    for ab in FT_ORDER:
        d = load(os.path.join(ft_runs, ab))
        if d is not None:
            rows.append(("ft", ab, FT_LABEL[ab], d))
    pg_dirs = sorted((os.path.basename(x) for x in glob.glob(
        os.path.join(pg_runs, "paperguard_*")) if os.path.isdir(x)),
        key=lambda b: (next((i for i, p in enumerate(PG_PREFIX) if b.startswith(p)), 99), b))
    for ab in pg_dirs:
        d = load(os.path.join(pg_runs, ab))
        if d is not None:
            lab, role = pg_meta(pg_runs, ab)
            PG_LABEL[ab] = lab
            rows.append(("pg" if role == "baseline" else "abl", ab, lab, d))
    ft_rows = [r for r in rows if r[0] == "ft"]
    pg_rows = [r for r in rows if r[0] == "pg"]
    abl_rows = [r for r in rows if r[0] == "abl"]
    if not pg_rows:
        raise SystemExit(f"[FATAL] no PaperGuard BASELINE stage1_detail.csv under {pg_runs}")
    main_rows = ft_rows + pg_rows                 # everything that is NOT an ablation

    W = 124
    L = ["=" * W, f"PAPERGUARD BASELINE vs SFT/DPO AGENTS — corpus {a.model}", "=" * W,
         f"  SFT/DPO runs : {ft_runs}", f"  PaperGuard   : {pg_runs}",
         f"  baseline     : {', '.join(ab for _, ab, _, _ in pg_rows)}",
         f"  ablation     : {', '.join(ab for _, ab, _, _ in abl_rows) or '(none run)'}"
         f"   <- NOT a baseline; reported only in the ABLATION block", ""]

    def a_line(lab, m):
        return (f"  {lab:<66} {m['N']:>5} {fmt(m['Acc']):>6} {fmt(m['Prec']):>6} "
                f"{fmt(m['Rec']):>6} {fmt(m['F1']):>6} {fmt(m['FPR']):>6} "
                f"{fmt(m['HarmRec']):>7} {fmt(m['Esc'], 1):>6} {fmt(m['Calls'], 2):>6}")
    A_HEAD = (f"  {'Method':<66} {'N':>5} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} "
              f"{'FPR':>6} {'HarmRec':>7} {'Esc%':>6} {'Calls':>6}")

    # ------------------------------------------------------------------ Panel A
    pa = {ab: panel_a(d) for _, ab, _, d in rows}
    L += ["PANEL A — detection on the full test corpus (SFT/DPO rows + PaperGuard baseline)",
          A_HEAD, "  " + "-" * (W - 4)]
    for kind, ab, lab, d in main_rows:
        L.append(a_line(lab, pa[ab]))
    for _, ab, lab, _ in pg_rows + abl_rows:
        sp = os.path.join(pg_runs, ab, "summary.json")
        if os.path.exists(sp):
            s = json.load(open(sp)).get("pg", {})
            L.append(f"  [{ab}] registry={s.get('registry')} ({s.get('registry_size')} "
                     f"patterns), top-k={s.get('top_k')}, embedder={s.get('embedder')}, "
                     f"localization calls={s.get('n_loc_calls_total')} (not in Calls), "
                     f"parse lenient/fail={s.get('parse_lenient')}/{s.get('parse_fail')}, "
                     f"verifier errors={s.get('verifier_errors')}")

    # ------------------------------------------------------------------ matching
    L += ["", "HEAD-TO-HEAD vs the PaperGuard BASELINE — same documents, same order?  "
              "(McNemar exact test on paired per-document correctness)"]
    if not ft_rows:
        L.append(f"  no SFT/DPO stage1_detail.csv under {ft_runs} — pass --ft-runs-dir")
    key = ["paper_id", "condition", "label", "harmful"]

    def same_corpus(x: pd.DataFrame, y: pd.DataFrame) -> bool:
        return len(x) == len(y) and bool(
            (x[key].astype(str).values == y[key].astype(str).values).all())
    for _, pab, plab, pdf in pg_rows:
        for _, fab, flab, fdf in ft_rows:
            if not same_corpus(pdf, fdf):
                L.append(f"  {pab} vs {fab}: NOT the same ordered corpus "
                         f"({len(pdf)} vs {len(fdf)} rows) — compare as separate samples")
                continue
            pc = (pdf["pred"].values == pdf["label"].values)
            fc = (fdf["pred"].values == fdf["label"].values)
            b, c, pv = mcnemar(fc, pc)
            dF1 = pa[fab]["F1"] - pa[pab]["F1"]
            dFPR = pa[fab]["FPR"] - pa[pab]["FPR"]
            L.append(f"  {flab:<32} vs PaperGuard  matched | {fab} right & PG wrong {b:>4}"
                     f" | PG right & {fab} wrong {c:>4} | p={pv:.2e} | dF1 {dF1:+.3f} "
                     f"dFPR {dFPR:+.3f}")

    # ------------------------------------------------------------------ Panel B
    B_HEAD = (f"  {'Method':<66} {'Npos':>5} {'Recall':>7} {'Prec':>6} {'F1':>6} {'FPR':>6} "
              f"{'HarmRec':>7} {'R@5':>6} {'R@10':>6} {'AUROC':>6}")
    FAMS = (("Instruction-bearing attacks: A5-A9, A11, A12", INSTRUCTION),
            ("Rhetorical suppression attacks: A1-A4, A10", RHETORICAL))

    def b_line(lab, m):
        return (f"  {lab:<66} {m['N_pos']:>5} {fmt(m['Recall']):>7} "
                f"{fmt(m['Prec']):>6} {fmt(m['F1']):>6} {fmt(m['FPR']):>6} "
                f"{fmt(m['HarmRec']):>7} {fmt(m['R@FPR5']):>6} {fmt(m['R@FPR10']):>6} "
                f"{fmt(m['AUROC']):>6}")
    pb = {(t, ab): panel_b(d, f) for t, f in FAMS for _, ab, _, d in rows}
    L += ["", "PANEL B — by threat family (negatives = all clean documents)", B_HEAD]
    for title, _ in FAMS:
        L += [f"  {title}", "  " + "-" * (W - 4)]
        for _, ab, lab, _ in main_rows:
            L.append(b_line(lab, pb[(title, ab)]))

    # ------------------------------------------------------------------ per family
    def family_table(rr_: List[Tuple[str, str, str, pd.DataFrame]]) -> List[str]:
        conds = sorted({c for *_, d in rr_ for c in d.loc[d["label"] == 1, "condition"]},
                       key=lambda c: (fam_num(c) or 999, c))
        def short(kind, ab):
            return {"pg": "PaperGuard", "abl": "PG+train[abl]"}.get(kind, ab)[:16]
        out = [f"  {'family':<22} {'n':>5} " + " ".join(f"{short(k, ab):>16}"
                                                     for k, ab, _, _ in rr_)]
        for c in conds:
            cells, n = [], None
            for *_, d in rr_:
                x = d[(d["label"] == 1) & (d["condition"] == c)]
                n = len(x) if n is None else n
                cells.append(fmt(x["pred"].mean()) if len(x) else "--")
            out.append(f"  {c:<22} {n:>5} " + " ".join(f"{s_:>16}" for s_ in cells))
        return out
    L += ["", "RECALL PER ATTACK FAMILY (SFT/DPO rows + PaperGuard baseline)"]
    L += family_table(main_rows)

    # ------------------------------------------------------------------ ABLATION
    abl_tex: List[str] = []
    if abl_rows:
        base_ab, base_lab, base_df = pg_rows[0][1], pg_rows[0][2], pg_rows[0][3]
        L += ["", "=" * W,
              "ABLATION (NOT A BASELINE) — does adding our TRAIN attacks to PaperGuard's "
              "known-attack list help?",
              f"  reference: the PaperGuard baseline ({base_ab})", "=" * W, A_HEAD,
              "  " + "-" * (W - 4), a_line(base_lab + " [baseline]", pa[base_ab])]
        for _, ab, lab, d in abl_rows:
            m, mb = pa[ab], pa[base_ab]
            L.append(a_line(lab, m))
            dl = {k: m[k] - mb[k] for k in ("Acc", "Prec", "Rec", "F1", "FPR", "HarmRec",
                                              "Calls")}
            L.append(f"  {'  change vs baseline':<66} {'':>5} {dl['Acc']:>+6.3f} "
                     f"{dl['Prec']:>+6.3f} {dl['Rec']:>+6.3f} {dl['F1']:>+6.3f} "
                     f"{dl['FPR']:>+6.3f} {dl['HarmRec']:>+7.3f} {'':>6} {dl['Calls']:>+6.2f}")
            if same_corpus(d, base_df):
                bc = base_df["pred"].values == base_df["label"].values
                ac = d["pred"].values == d["label"].values
                b, c, pv = mcnemar(bc, ac)
                new_tp = int(((d["pred"] == 1) & (base_df["pred"].values == 0)
                              & (d["label"] == 1)).sum())
                new_fp = int(((d["pred"] == 1) & (base_df["pred"].values == 0)
                              & (d["label"] == 0)).sum())
                lost = int(((d["pred"] == 0) & (base_df["pred"].values == 1)).sum())
                L += [f"  paired on the same {len(d)} documents: {new_tp} attacks newly "
                      f"caught, {new_fp} clean papers newly flagged, {lost} baseline flags "
                      f"lost;",
                      f"  McNemar ablation vs baseline: ablation right & baseline wrong {c}, "
                      f"baseline right & ablation wrong {b}, p={pv:.2e}"]
                if lost:
                    L.append("  (lost flags > 0 means the shared 10 patterns were not answered "
                             "identically — run the baseline FIRST so the ablation reuses its "
                             "cached verifier answers)")
            else:
                L.append("  NOT the same ordered corpus as the baseline — deltas are between "
                         "different samples")
            for title, _ in FAMS:
                p0, p1 = pb[(title, base_ab)], pb[(title, ab)]
                L.append(f"  {title.split(':')[0]:<31} recall {fmt(p0['Recall'])} -> "
                         f"{fmt(p1['Recall'])}   FPR {fmt(p0['FPR'])} -> {fmt(p1['FPR'])}   "
                         f"HarmRec {fmt(p0['HarmRec'])} -> {fmt(p1['HarmRec'])}")
            abl_tex += [f"{lab} % folder: defense_baselines/paperguard_defense/runs/{ab}",
                        f"& {m['N']} & {fmt(m['Acc'])} & {fmt(m['Prec'])} & {fmt(m['Rec'])} & "
                        f"{fmt(m['F1'])} & {fmt(m['FPR'])} & {fmt(m['HarmRec'])} & "
                        f"{fmt(m['Esc'], 1)} & {fmt(m['Calls'], 2)} \\\\",
                        r"\quad $\Delta$ vs.\ PaperGuard baseline",
                        f"& & {dl['Acc']:+.3f} & {dl['Prec']:+.3f} & {dl['Rec']:+.3f} & "
                        f"{dl['F1']:+.3f} & {dl['FPR']:+.3f} & {dl['HarmRec']:+.3f} & & "
                        f"{dl['Calls']:+.2f} \\\\"]
        L += ["", "  recall per attack family, baseline vs ablation:"]
        L += family_table(pg_rows[:1] + abl_rows)

    # ------------------------------------------------------------------ rerate
    rr = {}
    for _, ab, lab, _ in pg_rows + abl_rows:
        sp = os.path.join(pg_runs, ab, "rerate_summary.json")
        if os.path.exists(sp):
            rr[ab] = json.load(open(sp))
    if rr:
        L += ["", "LOCATE -> REMOVE -> RE-REVIEW  (OVERALL 1-10; B benign, A attacked, "
                  "S attacked minus located chunks; S = A when not flagged)",
              f"  {'Method / group':<74} {'n':>4} {'B':>5} {'A':>5} {'S':>5} {'A-B':>6} "
              f"{'S-B':>6} {'recov':>6} {'ASR A':>6} {'ASR S':>6} {'locHit':>6} "
              f"{'locP':>5} {'locR':>5}"]
        for ab, s in rr.items():
            tag = PG_LABEL.get(ab, ab)
            for g, m in s["groups"].items():
                L.append(f"  {(tag[:46] + ' | ' + g)[:74]:<74} "
                         f"{m['n_scored']:>4} {fmt(m['B_overall'], 2):>5} "
                         f"{fmt(m['A_overall'], 2):>5} {fmt(m['S_overall'], 2):>5} "
                         f"{fmt(m['d_attack'], 2):>6} {fmt(m['d_after'], 2):>6} "
                         f"{fmt(m.get('recovery'), 2):>6} {fmt(m['asr_attack'], 2):>6} "
                         f"{fmt(m['asr_after'], 2):>6} {fmt(m.get('loc_hit'), 2):>6} "
                         f"{fmt(m.get('loc_precision'), 2):>5} "
                         f"{fmt(m.get('loc_recall'), 2):>5}")
            c = s["false_positive_collateral"]
            L.append(f"  {'':<4}false-positive collateral: {c['n_flagged_clean']} clean "
                     f"papers flagged, OVERALL {fmt(c['B_overall'], 2)} -> "
                     f"{fmt(c['SB_overall'], 2)} after removal")

    L += ["", "NOTES",
          "  * The PaperGuard BASELINE uses only PaperGuard's own known-attack list (A1, seed "
          "42, 10 prompts).",
          "    The '+ our TRAIN attacks' row is an ABLATION and is never reported as a "
          "baseline.",
          "  * Esc% is 100 for PaperGuard: every document goes to the verifier (no cheaper "
          "tier in front).",
          "  * Calls = verifier calls for detection (<= registry size, stops at the first "
          "verified pattern).",
          "    Localization calls are reported separately and are not in Calls.",
          "  * PaperGuard R@FPR5/R@FPR10/AUROC use the retrieval score, not the binary "
          "verdict; '--' for rows with no score.",
          "=" * W]
    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(out_dir, "PG_TABLE.txt"), "w") as f:
        f.write(txt + "\n")

    # ------------------------------------------------------------------ LaTeX
    T = ["% ---- Panel A: PaperGuard BASELINE row (paste under the SFT/DPO rows) ----"]
    for _, ab, lab, _ in pg_rows:
        m = pa[ab]
        T.append(f"{lab} % folder: defense_baselines/paperguard_defense/runs/{ab}")
        T.append(f"& {m['N']} & {fmt(m['Acc'])} & {fmt(m['Prec'])} & {fmt(m['Rec'])} & "
                 f"{fmt(m['F1'])} & {fmt(m['FPR'])} & {fmt(m['HarmRec'])} & "
                 f"{fmt(m['Esc'], 1)} & {fmt(m['Calls'], 2)} \\\\")
    for title, _ in FAMS:
        T.append(f"% ---- Panel B: PaperGuard BASELINE row, {title} ----")
        for _, ab, lab, _ in pg_rows:
            m = pb[(title, ab)]
            T.append(f"D8 {lab}")
            T.append(f"& {m['N_pos']} & {fmt(m['Recall'])} & {fmt(m['Prec'])} & "
                     f"{fmt(m['F1'])} & {fmt(m['FPR'])} & {fmt(m['HarmRec'])} & "
                     f"{fmt(m['R@FPR5'])} & {fmt(m['R@FPR10'])} & {fmt(m['AUROC'])} \\\\")
    if abl_tex:
        T += ["", "% ==== ABLATION — NOT a baseline. Report separately (e.g. an ablation "
                  "table or a footnote), never among the baselines. ====",
              "% columns: N & Acc & Prec & Rec & F1 & FPR & HarmRec & Esc & Calls"] + abl_tex
    if rr:
        T += ["", "% ---- Re-rating table ----",
              r"\begin{tabular}{lrrrrrrrrrr}", r"\toprule",
              r"Method & $n$ & $B$ & $A$ & $S$ & $A{-}B$ & $S{-}B$ & Recov. & "
              r"ASR$_A$ & ASR$_S$ & LocHit \\", r"\midrule"]
        for ab, s in rr.items():
            m = s["groups"].get("ALL attacked pairs")
            if not m:
                continue
            T.append(f"{PG_LABEL.get(ab, ab)} & {m['n_scored']} & {fmt(m['B_overall'], 2)} & "
                     f"{fmt(m['A_overall'], 2)} & {fmt(m['S_overall'], 2)} & "
                     f"{fmt(m['d_attack'], 2)} & {fmt(m['d_after'], 2)} & "
                     f"{fmt(m.get('recovery'), 2)} & {fmt(m['asr_attack'], 2)} & "
                     f"{fmt(m['asr_after'], 2)} & {fmt(m.get('loc_hit'), 2)} \\\\")
        T += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(out_dir, "pg_rows.tex"), "w") as f:
        f.write("\n".join(T) + "\n")
    role_of = {"ft": "ours", "pg": "baseline", "abl": "ablation"}
    pd.DataFrame([{"role": role_of[k], "ablation": ab, "label": lab, **pa[ab]}
                  for k, ab, lab, _ in rows]).to_csv(
        os.path.join(out_dir, "pg_panel_a.csv"), index=False)
    print(f"\n[saved] {os.path.join(out_dir, 'PG_TABLE.txt')}")
    print(f"[saved] {os.path.join(out_dir, 'pg_rows.tex')}")
    print(f"[saved] {os.path.join(out_dir, 'pg_panel_a.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
