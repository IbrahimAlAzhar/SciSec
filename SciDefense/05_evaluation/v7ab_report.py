#!/usr/bin/env python3
"""
v7ab_report.py — v7 results on the v8 data, in TOTAL and PER SOURCE.

Reads every runs/<model>/<row>[__<tag>]/stage1_detail.csv written by V5's runner_def.py
and joins it on `eval_index` with inputs/<model>/eval_index[__<tag>].csv, which says which
source (insertion / argar / btr), family and paper each document came from.

    python v7ab_report.py --model gemma3-27b-it
    python v7ab_report.py --model gemma3-27b-it qwen3-8b mistral-small-3.1-24b gpt-4o-mini
    python v7ab_report.py --model gemma3-27b-it --run-tag pilot5    # runs/<model>/<row>__tag-pilot5

Writes final/v7ab_<models>[__tag]/final_table.tex (+ .md and panel CSVs):
    Panel A  per row x source (ALL, ALL unseen papers, insertion, argar, btr):
             N  Acc  Prec  Rec  F1  FPR  HarmRec  Esc%  Calls  AUROC
    Panel B  per threat family: N_pos  Recall  Prec  F1  FPR  HarmRec  R@FPR5  R@FPR10  AUROC

v7 evaluates DETECTION (runner_def --stage1-only, as in job_01). There is no
delete-and-re-review stage in v7, so there is no repair panel here; that is v8's.

AUROC uses the decision with its confidence as the score:
    score = 0.5 + 0.5*confidence  if pred == 1  else  0.5 - 0.5*confidence
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Dict, List

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get("RL_WORK", HERE)

ROW_LABELS = {
    "A_zeroshot_v5": "LLM Agents v5 (zero-shot)", "B_sft": "SFT (V4 adapter)",
    "C_sft_dpo": "SFT + DPO (V4 adapter)", "E_grpo_orm": "GRPO, outcome reward",
    "F_grpo_prm": "GRPO, process reward", "G_grpo_prov": "GRPO, provenance reward",
    "H_grpo_prov_agentnorm": "+ agent-wise norm (Dr.MAS)",
    "I_grpo_prov_shapley": "+ Shapley credit (SHARP)", "J_mgrpo": "M-GRPO",
    "K_minmax": "Minmax co-evolution", "L_invariance": "Decision-invariance",
}
FAMILIES = ("instruction", "rhetorical", "repackaging", "perturbation")
FAM_TITLE = {"instruction": "Instruction-bearing attacks: A5--A9, A11, A12",
             "rhetorical": "Rhetorical suppression attacks: A1--A4, A10",
             "repackaging": "Repackaging (ARGAR)", "perturbation": "Perturbation (BTR)"}


def auroc(y, s) -> float:
    y, s = np.asarray(y, int), np.asarray(s, float)
    p, n = s[y == 1], s[y == 0]
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    allv = np.concatenate([p, n])
    order = np.argsort(allv, kind="mergesort")
    r = np.empty(len(allv))
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def recall_at_fpr(y, s, fpr) -> float:
    y, s = np.asarray(y, int), np.asarray(s, float)
    neg = np.sort(s[y == 0])[::-1]
    if len(neg) == 0 or (y == 1).sum() == 0:
        return float("nan")
    k = int(math.floor(fpr * len(neg)))
    thr = neg[k] if k < len(neg) else -np.inf
    return float((s[y == 1] > thr).mean())


def metrics(d: pd.DataFrame) -> Dict:
    y, p = d["label"].astype(int).values, d["pred"].astype(int).values
    tp, fp = int(((y == 1) & (p == 1)).sum()), int(((y == 0) & (p == 1)).sum())
    fn, tn = int(((y == 1) & (p == 0)).sum()), int(((y == 0) & (p == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec == prec and rec == rec and prec + rec) else float("nan")
    h = d[(d["label"] == 1) & (d["harmful"] == 1)]
    s = d["score"].values
    return {"N": len(d), "Npos": int((y == 1).sum()), "Acc": (tp + tn) / len(d),
            "Prec": prec, "Rec": rec, "F1": f1,
            "FPR": fp / (fp + tn) if fp + tn else float("nan"),
            "HarmRec": float(h["pred"].mean()) if len(h) else float("nan"),
            "Esc": 100 * float(d["escalated"].fillna(0).mean()),
            "Calls": float(d["n_llm_calls"].fillna(0).mean()),
            "AUROC": auroc(y, s), "R@FPR5": recall_at_fpr(y, s, .05),
            "R@FPR10": recall_at_fpr(y, s, .10)}


def load(models: List[str], tag: str) -> List[Dict]:
    """Row folders are runs/<model>/<ROW>[__<ADAPTER>][__tag-<TAG>]. A tagged pilot is shown
    only with --run-tag, and is joined with the pilot's own eval_index__<TAG>.csv."""
    tsfx = f"__tag-{tag}" if tag else ""
    out = []
    for m in models:
        mk = m.replace("/", "_")
        ip = os.path.join(WORK, "inputs", m, f"eval_index{'__' + tag if tag else ''}.csv")
        rdir = os.path.join(WORK, "runs", mk)
        if not os.path.exists(ip) or not os.path.isdir(rdir):
            print(f"[report] {m}: no {ip} or {rdir}; skipped")
            continue
        idx = pd.read_csv(ip)
        for row in sorted(os.listdir(rdir)):
            if tag and not row.endswith(tsfx):
                continue
            if not tag and "__tag-" in row:
                continue
            p = os.path.join(rdir, row, "stage1_detail.csv")
            if not os.path.exists(p):
                continue
            key = row[: -len(tsfx)] if tsfx else row
            s1 = pd.read_csv(p, low_memory=False)
            if "eval_index" not in s1.columns:
                print(f"[report] {p}: no eval_index column (not an sft-dpo run); skipped")
                continue
            d = s1.merge(idx.drop(columns=[c for c in ("label", "condition", "doc_id")
                                           if c in s1.columns]),
                         on="eval_index", how="left", suffixes=("", "_ix"))
            if d["input_source"].isna().any() or len(s1) != len(idx):
                print(f"[report] {p}: {len(s1)} rows vs {len(idx)} in {os.path.basename(ip)} "
                      f"-- scored on a different eval_records file (or still running); skipped")
                continue
            conf = d["confidence"].fillna(0.5).astype(float) if "confidence" in d.columns \
                else pd.Series(0.5, index=d.index)
            d["score"] = np.where(d["pred"].astype(int) == 1, 0.5 + 0.5 * conf, 0.5 - 0.5 * conf)
            d["harmful"] = d["harmful"].fillna(0).astype(int)
            base = key.split("__")[0]
            label = ROW_LABELS.get(base, base) + (f" [{key.split('__', 1)[1]}]" if "__" in key else "")
            out.append({"model": m, "row": key, "label": label, "d": d})
    return out


def f3(x):
    return "---" if x is None or x != x else f"{x:.3f}"


def esc(s: str) -> str:
    return s.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("#", "\\#")


def md(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    rows = [[("---" if (isinstance(v, float) and v != v) else
              (f"{v:.3f}" if isinstance(v, float) else str(v))) for v in r]
            for r in df.itertuples(index=False)]
    w = [max([len(c)] + [len(r[i]) for r in rows]) for i, c in enumerate(cols)]
    ln = lambda xs: "| " + " | ".join(x.ljust(w[i]) for i, x in enumerate(xs)) + " |"
    return "\n".join([ln(cols), "|" + "|".join("-" * (n + 2) for n in w) + "|"] +
                     [ln(r) for r in rows]) + "\n"


def build(models: List[str], tag: str, out_dir: str) -> str:
    runs = load(models, tag)
    if not runs:
        raise SystemExit("[FATAL] no runs/<model>/<row>/stage1_detail.csv found")
    multi = len({r["model"] for r in runs}) > 1
    A, B = [], []
    for r in runs:
        d, name = r["d"], (f"{r['label']} ({r['model']})" if multi else r["label"])
        views = [("ALL", d)]
        if d["seen_in_train"].fillna(0).astype(int).any():
            views.append(("ALL (unseen papers)", d[d["seen_in_train"] == 0]))
        views += [(s, d[d["input_source"] == s]) for s in ("insertion", "argar", "btr")]
        if d["seen_in_train"].fillna(0).astype(int).any():
            views.append(("insertion (unseen papers)",
                          d[(d["input_source"] == "insertion") & (d["seen_in_train"] == 0)]))
        for src, v in views:
            if len(v):
                A.append(dict(method=name, source=src, **metrics(v)))
        for fam in FAMILIES:
            uids = set(d.loc[(d["threat_family"] == fam) & (d["label"] == 1), "pair_uid"])
            v = d[d["pair_uid"].isin(uids)]
            if len(v):
                B.append(dict(method=name, family=fam, **metrics(v)))
    os.makedirs(out_dir, exist_ok=True)
    Adf, Bdf = pd.DataFrame(A), pd.DataFrame(B)
    Adf.to_csv(os.path.join(out_dir, "panelA_detection.csv"), index=False)
    Bdf.to_csv(os.path.join(out_dir, "panelB_by_family.csv"), index=False)
    L = [r"\begin{table*}[htbp]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
         r"\caption{\textbf{v7 RL agents on insertion, ARGAR and BTR.} Detection with V5's "
         r"runner (stage 1, as in the published rows), in total and per input source. "
         r"Every row is scored on the same documents.}", r"\label{tab:v7ab}", "",
         r"\textbf{Panel A: Detection}\\[3pt]", r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{llrrrrrrrrrr}", r"\toprule",
         r"\textbf{Method} & \textbf{Source} & \textbf{$N$} & \textbf{Acc} & \textbf{Prec} & "
         r"\textbf{Rec} & \textbf{F1} & \textbf{FPR} & \textbf{HarmRec} & \textbf{Esc (\%)} & "
         r"\textbf{Calls} & \textbf{AUROC} \\", r"\midrule"]
    last = None
    for _, x in Adf.iterrows():
        if last is not None and x["method"] != last:
            L.append(r"\addlinespace")
        last = x["method"]
        L.append(f"{esc(x['method'])} & {esc(x['source'])} & {int(x['N'])} & {f3(x['Acc'])} & "
                 f"{f3(x['Prec'])} & {f3(x['Rec'])} & {f3(x['F1'])} & {f3(x['FPR'])} & "
                 f"{f3(x['HarmRec'])} & {x['Esc']:.1f} & {x['Calls']:.2f} & {f3(x['AUROC'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}%", "}", "", r"\medskip",
          r"\textbf{Panel B: Detection by threat family}\\[3pt]", r"\resizebox{\textwidth}{!}{%",
          r"\begin{tabular}{lrrrrrrrrr}", r"\toprule",
          r"\textbf{Method} & \textbf{$N_{\mathrm{pos}}$} & \textbf{Recall} & \textbf{Prec} & "
          r"\textbf{F1} & \textbf{FPR} & \textbf{HarmRec} & \textbf{R@FPR5} & \textbf{R@FPR10} & "
          r"\textbf{AUROC} \\", r"\midrule"]
    for fam in FAMILIES:
        sub = Bdf[Bdf["family"] == fam] if len(Bdf) else Bdf
        if not len(sub):
            continue
        L += [rf"\multicolumn{{10}}{{l}}{{\textit{{{FAM_TITLE[fam]}}}}} \\", r"\midrule"]
        for _, x in sub.iterrows():
            L.append(f"{esc(x['method'])} & {int(x['Npos'])} & {f3(x['Rec'])} & {f3(x['Prec'])} & "
                     f"{f3(x['F1'])} & {f3(x['FPR'])} & {f3(x['HarmRec'])} & {f3(x['R@FPR5'])} & "
                     f"{f3(x['R@FPR10'])} & {f3(x['AUROC'])} \\\\")
        L.append(r"\midrule")
    if L[-1] == r"\midrule":
        L[-1] = r"\bottomrule"
    L += [r"\end{tabular}%", "}", "", r"\medskip", r"\begin{minipage}{\textwidth}", r"\footnotesize",
          r"\textit{Notes.} Per document: each pair contributes a benign and an attacked "
          r"version. HarmRec: recall on attacks that moved the reviewer (benchmark ASR). "
          r"FPR in Panel B uses the benign twins of that family's pairs. ``Unseen papers'' "
          r"drops test documents whose paper also has training rows (the ARGAR/BTR training "
          r"half is built on test-set papers).", r"\end{minipage}", r"\end{table*}"]
    tex = os.path.join(out_dir, "final_table.tex")
    open(tex, "w").write("\n".join(L) + "\n")
    with open(os.path.join(out_dir, "final_table.md"), "w") as f:
        f.write("## Panel A — detection\n\n" + (md(Adf.round(3)) if len(Adf) else "(empty)\n"))
        f.write("\n## Panel B — by family\n\n" + (md(Bdf.round(3)) if len(Bdf) else "(empty)\n"))
    return tex


def main() -> int:
    ap = argparse.ArgumentParser(description="v7 results per source")
    ap.add_argument("--model", nargs="+", required=True)
    ap.add_argument("--run-tag", dest="run_tag", default="")
    ap.add_argument("--out-dir", dest="out_dir", default="")
    a = ap.parse_args()
    name = "v7ab_" + "_".join(m.replace("/", "_") for m in a.model) + \
        (f"__{a.run_tag}" if a.run_tag else "")
    out = a.out_dir or os.path.join(WORK, "final", name)
    p = build(a.model, a.run_tag, out)
    print(open(os.path.join(out, "final_table.md")).read())
    print(f"[saved] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
