#!/usr/bin/env python3
"""
v7_table.py — one table, with the published rows IMPORTED and the new rows measured by
the same evaluator.

THE RULE THIS FILE ENFORCES
=======================================================================================
An imported row is READ FROM DISK and never regenerated. Re-running an experiment whose
numbers are already in your paper can only move them, and a baseline that moved is not
the baseline the paper reports.

A new row is produced by v5's OWN runner_def.py with a different --vllm-model. vLLM's
LoRA selection IS the model field, so pointing it at an adapter name swaps the policy and
changes nothing else: same defense.detect(), same gates, same calibration, same memory,
same corpus. The row then differs from the baseline by the adapter and by nothing else,
which is the only way an ablation means anything.

THE COMPARABILITY CHECK, AND WHY IT REFUSES RATHER THAN WARNS
    Before any two rows are put in the same table, `check_comparable` verifies they were
    scored on the same documents: same doc_id set, same label per doc_id. If they were
    not, the rows are measuring different corpora and the difference between them is not
    a method effect. That is a hard error. A warning would be read once and forgotten,
    and the table would go into a paper anyway.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import rl_config as C

BASELINE_F1 = 0.807
BASELINE_FPR = 0.107


def _f(x, n: int = 3) -> str:
    if x is None:
        return "--"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "--"
    return "--" if v != v else f"{v:.{n}f}"


def esc(s: str) -> str:
    out = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                 ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                 ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        out = out.replace(a, b)
    return out


# ======================================================================================
# loading a row
# ======================================================================================
def find_detail(d: str) -> str:
    """stage1_detail.csv anywhere under d. v5 and v4 nest it differently."""
    if not d or not os.path.isdir(d):
        return ""
    direct = os.path.join(d, "stage1_detail.csv")
    if os.path.exists(direct):
        return direct
    hits = sorted(glob.glob(os.path.join(d, "**", "stage1_detail.csv"), recursive=True))
    return hits[0] if hits else ""


def load_row(path: str):
    import pandas as pd
    df = pd.read_csv(path, low_memory=False)
    need = {"doc_id", "label", "pred"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"[FATAL] {path} is missing {sorted(missing)}")
    df["doc_id"] = df["doc_id"].astype(str)
    return df


def metrics(df, subset: str = "all") -> Dict:
    d = df
    if subset == "harmful" and "harmful" in df.columns:
        d = df[(df.label == 0) | (df.harmful == 1)]
    elif subset in ("text_only", "query", "white_box") and "access" in df.columns:
        d = df[(df.label == 0) | (df.access == subset)]
    elif subset == "control" and "is_control" in df.columns:
        d = df[df.is_control == 1]
    tp = int(((d.label == 1) & (d.pred == 1)).sum())
    fp = int(((d.label == 0) & (d.pred == 1)).sum())
    tn = int(((d.label == 0) & (d.pred == 0)).sum())
    fn = int(((d.label == 1) & (d.pred == 0)).sum())
    n = tp + fp + tn + fn
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    harm = d[(d.label == 1) & (d.get("harmful", 0) == 1)] if "harmful" in d.columns \
        else d.iloc[0:0]
    calls = d["n_llm_calls"].mean() if "n_llm_calls" in d.columns else None
    esc_ = 100.0 * d["escalated"].mean() if "escalated" in d.columns else None
    return {"n": n, "accuracy": (tp + tn) / n if n else None,
            "precision": prec, "recall": rec,
            "f1": (2 * prec * rec / (prec + rec)) if (prec and rec) else None,
            "fpr": fp / (fp + tn) if fp + tn else None,
            "harm_recall": (float((harm.pred == 1).mean()) if len(harm) else None),
            "escalated_pct": esc_, "mean_calls": calls,
            "flag_rate": (tp + fp) / n if n else None}


# ======================================================================================
# the check that decides whether a table is allowed to exist
# ======================================================================================
def check_comparable(rows: List[Tuple[str, "object"]]) -> List[str]:
    L: List[str] = []
    if len(rows) < 2:
        return L
    ref_name, ref = rows[0]
    ref_ids = set(ref["doc_id"])
    ref_lab = dict(zip(ref["doc_id"], ref["label"]))
    bad: List[str] = []
    L.append("")
    L.append("--- comparability " + "-" * 72)
    L.append(f"  reference: {ref_name}  ({len(ref_ids)} documents)")
    for name, df in rows[1:]:
        ids = set(df["doc_id"])
        shared = ref_ids & ids
        lab = dict(zip(df["doc_id"], df["label"]))
        flipped = sum(1 for i in shared if lab[i] != ref_lab[i])
        cov = len(shared) / max(1, len(ref_ids))
        mark = ""
        if cov < 0.98:
            mark = "   <-- DIFFERENT CORPUS"
            bad.append(f"{name}: only {cov:.1%} of the reference documents")
        if flipped:
            mark = "   <-- LABELS DISAGREE"
            bad.append(f"{name}: {flipped} documents carry a different label")
        L.append(f"  {name:<28}{len(ids):>6} docs   shared {len(shared):>6} "
                 f"({cov:.1%}){mark}")
    if bad:
        L.append("")
        for b in bad:
            L.append(f"  {b}")
        raise SystemExit(
            "[FATAL] these rows were scored on different documents:\n    "
            + "\n    ".join(bad)
            + "\n  Putting them in one table would present a corpus difference as a\n"
              "  method effect. Re-run the new rows on the SAME corpus the imported\n"
              "  baseline used — with v5's runner_def.py and the same --benign/--attacked\n"
              "  arguments — or drop the mismatched row.")
    L.append("  all rows scored on the same documents with the same labels")
    return L


# ======================================================================================
def collect(work: str, model: str) -> List[Dict]:
    mk = model.replace("/", "_")
    out: List[Dict] = []
    for ab in C.ABLATIONS:
        if ab.imported:
            d = C.BASELINE_RUNS.get(ab.imported, "")
            path = find_detail(d)
            src = "imported"
        else:
            path = find_detail(os.path.join(work, C.D_RUNS, mk, ab.key))
            src = "measured"
        if not path:
            continue
        try:
            df = load_row(path)
        except SystemExit:
            raise
        except Exception as e:
            print(f"[warn] {path}: {type(e).__name__}: {e}")
            continue
        rec = {"key": ab.key, "label": ab.label, "source": src, "path": path,
               "df": df, "mechint": ab.mechint, "imported": bool(ab.imported)}
        for sub in ("all", "harmful", "text_only", "white_box", "control"):
            rec[sub] = metrics(df, sub)
        out.append(rec)
    return out


def table(rows: Sequence[Dict], L: List[str], subset: str, title: str) -> None:
    L.append("")
    L.append(f"--- {title} " + "-" * max(0, 74 - len(title)))
    hdr = (f"  {'ablation':<44}{'src':<10}{'N':>6}{'Acc':>8}{'Prec':>8}{'Rec':>8}"
           f"{'F1':>8}{'FPR':>8}{'HarmRec':>9}{'Calls':>8}")
    L.append(hdr)
    L.append("  " + "-" * (len(hdr) - 2))
    for r in rows:
        m = r[subset]
        if not m["n"]:
            continue
        L.append(f"  {r['label'][:44]:<44}{r['source']:<10}{m['n']:>6}"
                 f"{_f(m['accuracy']):>8}{_f(m['precision']):>8}{_f(m['recall']):>8}"
                 f"{_f(m['f1']):>8}{_f(m['fpr']):>8}{_f(m['harm_recall']):>9}"
                 f"{_f(m['mean_calls'], 2):>8}")


def main() -> int:
    p = argparse.ArgumentParser(description="the v7 ablation table")
    p.add_argument("--model", required=True, choices=sorted(C.MODELS))
    p.add_argument("--work", default=C.WORK)
    p.add_argument("--out-dir", dest="out_dir", default="")
    p.add_argument("--no-strict", dest="strict", action="store_false",
                   help="report corpus mismatches instead of refusing. For a pilot only.")
    p.set_defaults(strict=True)
    a = p.parse_args()

    rows = collect(a.work, a.model)
    if not rows:
        raise SystemExit(
            "[FATAL] no rows found.\n"
            "  IMPORTED rows are read from C.BASELINE_RUNS — check those paths point at\n"
            "  the directories holding your published stage1_detail.csv files.\n"
            "  MEASURED rows come from runs/<model>/<ablation>/stage1_detail.csv, which\n"
            "  v5's runner_def.py writes when you pass --out-dir there.")
    out_dir = a.out_dir or os.path.join(a.work, C.D_FINAL)
    os.makedirs(out_dir, exist_ok=True)

    L: List[str] = []
    L.append("=" * 100)
    L.append(f"V7 ABLATION TABLE — {a.model}")
    L.append("=" * 100)
    n_imp = sum(1 for r in rows if r["imported"])
    L.append(f"  {len(rows)} rows: {n_imp} imported, {len(rows)-n_imp} measured here")
    L.append("  Imported rows are READ FROM DISK and were never re-run. Measured rows")
    L.append("  come from v5's own runner_def.py with a different --vllm-model, so they")
    L.append("  differ from the baseline by the adapter and by nothing else.")
    for r in rows:
        L.append(f"    {r['key']:<24}{r['source']:<10}{r['path']}")

    try:
        L += check_comparable([(r["key"], r["df"]) for r in rows])
    except SystemExit as e:
        if a.strict:
            print("\n".join(L))
            raise
        L.append(f"  [not strict] {e}")

    table(rows, L, "all", "DETECTION — all conditions")
    table(rows, L, "harmful", "DETECTION — attacks that actually moved this reviewer")
    table(rows, L, "text_only", "DETECTION — deployable threat model (text in a PDF)")
    if any(r["white_box"]["n"] for r in rows):
        table(rows, L, "white_box", "WHITE-BOX CEILING — requires the reviewing stack")

    mech = [r for r in rows if r["mechint"]]
    if mech:
        L.append("")
        L.append("--- MECHANISTICALLY GROUNDED ROWS " + "-" * 57)
        L.append("  These need the reviewer's attention, so they are available only to a")
        L.append("  venue hosting an open-weights reviewer. Report them in their own")
        L.append("  table with that caption; pooling them with the deployable rows would")
        L.append("  overstate what a venue using a closed API can run.")
        for r in mech:
            L.append(f"    {r['label']}")

    # the control rows, which decide whether any of this is real
    L.append("")
    L.append("--- FILLER CONTROL — every flag here is a false alarm " + "-" * 37)
    L.append(f"  {'ablation':<44}{'n':>6}{'flag rate':>11}")
    for r in rows:
        m = r["control"]
        if m["n"]:
            L.append(f"  {r['label'][:44]:<44}{m['n']:>6}{_f(m['flag_rate']):>11}")
    L.append("  Length-matched neutral inserts. If this rate approaches the attacked")
    L.append("  recall, the detector is finding 'a block of text was added here'.")

    L.append("")
    L.append("=" * 100)
    L.append("FLAGS")
    L.append("=" * 100)
    probs: List[str] = []
    for r in rows:
        m = r["all"]
        if isinstance(m["fpr"], float) and m["fpr"] > BASELINE_FPR + 1e-9:
            probs.append(f"{r['label']}: FPR {m['fpr']:.3f} above the baseline's "
                         f"{BASELINE_FPR:.3f} — ~{m['fpr']*500:.0f} accused authors per "
                         f"500 honest manuscripts.")
        if isinstance(m["flag_rate"], float) and m["flag_rate"] > 0.8:
            probs.append(f"{r['label']}: flags {m['flag_rate']:.1%} of ALL documents. "
                         f"That is the flag-everything corner, not a detector.")
    for p_ in dict.fromkeys(probs) or ["  none"]:
        L.append(f"  - {p_}" if probs else p_)

    txt = "\n".join(L)
    tp = os.path.join(out_dir, "V7_TABLE.txt")
    open(tp, "w").write(txt + "\n")
    print(txt)

    tex = [r"\begin{table*}[t]\centering\small",
           r"\caption{Defence ablations. Rows marked \emph{imported} are the published "
           r"results, read from disk and not re-run; measured rows come from the same "
           r"evaluator with a different adapter.}",
           r"\label{tab:v7}", r"\begin{tabular}{llcccccccc}", r"\toprule",
           r"\textbf{Ablation} & \textbf{Src} & \textbf{$N$} & \textbf{Acc} & "
           r"\textbf{Prec} & \textbf{Rec} & \textbf{F1} & \textbf{FPR} & "
           r"\textbf{HarmRec} & \textbf{Calls} \\", r"\midrule"]
    for r in rows:
        m = r["text_only"] if r["text_only"]["n"] else r["all"]
        tex.append(f"{esc(r['label'])} & {r['source']} & {m['n']} & "
                   f"{_f(m['accuracy'])} & {_f(m['precision'])} & {_f(m['recall'])} & "
                   f"{_f(m['f1'])} & {_f(m['fpr'])} & {_f(m['harm_recall'])} & "
                   f"{_f(m['mean_calls'], 2)} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    xp = os.path.join(out_dir, "v7_table.tex")
    open(xp, "w").write("\n".join(tex) + "\n")

    import pandas as pd
    flat = [{"key": r["key"], "label": r["label"], "source": r["source"],
             **{f"{s}.{k}": r[s][k] for s in ("all", "harmful", "text_only")
                for k in ("n", "accuracy", "precision", "recall", "f1", "fpr")}}
            for r in rows]
    cp = os.path.join(out_dir, "v7_table.csv")
    pd.DataFrame(flat).to_csv(cp, index=False)
    print(f"\n[saved] {tp}\n[saved] {xp}\n[saved] {cp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
