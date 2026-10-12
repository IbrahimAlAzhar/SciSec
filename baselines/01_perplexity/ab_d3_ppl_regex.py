#!/usr/bin/env python3
"""
ab_d3_ppl_regex.py — D3 on ARGAR/BTR: regex only, GPT-2 perplexity only, and the two
combined (Jain et al. 2309.00614).

    python ab_d3_ppl_regex.py --ppl-model path/to/models/gpt2
    python ab_d3_ppl_regex.py --no-ppl                # regex only, CPU

THE SAME DETECTOR AS THE INSERTION TABLE
    The patterns (`regex_score`) and the perplexity engine (`PPL`, 128-token windows,
    stride 64, max over windows) are IMPORTED from defense_baselines/d3_ppl_regex.py, not
    copied, so this row and the insertion row are one detector.

OPERATING POINT (--thresholds insertion, the default)
    regex_only / ppl_only   the threshold d3_ppl_regex.py chose on its validation papers
                            (best recall at FPR <= 5%), re-derived from
                            output/D3_ppl_regex/predictions_all.csv.
    combined                d3's combined score is  regex / max(regex)  +  rank-percentile
                            of PPL, both computed over the scored set. To keep it the SAME
                            function here, the normaliser (max regex) and the PPL
                            percentile come from the insertion test rows, and the
                            threshold is d3's own val-derived one.

WHAT TO EXPECT, AND WHY IT IS STILL WORTH A ROW
    ARGAR inserts no imperative and BTR inserts nothing at all, so the regex has nothing
    to match. BTR's character edits (DeepWordBug, PuncAttack) DO raise perplexity, which
    is the one place this cheap screen may beat chance -- that is the contrast to report.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import ab_common as A


def ecdf(ref: np.ndarray):
    ref = np.sort(ref[~np.isnan(ref)])

    def f(x):
        x = np.asarray(x, float)
        if len(ref) == 0:
            return np.zeros_like(x)
        v = np.searchsorted(ref, x, side="right") / len(ref)
        return np.where(np.isnan(x), 0.0, v)
    return f


def main() -> int:
    ap = A.add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--ppl-model", default=A.first_dir(f"{A.MODELS_DIR}/gpt2", f"{A.HF_HOME}/gpt2"))
    ap.add_argument("--no-ppl", action="store_true")
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--stride", type=int, default=64)
    a = ap.parse_args()

    D3 = A.parent_import("d3_ppl_regex")
    out = A.Out("D3_ppl_regex", a.run_tag)
    out(f"[reuse] regex patterns + PPL engine from {D3.__file__}")
    pairs = A.load_pairs(a.sources, a.subset, a.max_rows, a.pairs_csv)
    docs = A.docs_frame(pairs)
    texts, inv = A.unique_texts(docs)
    out(f"[data] {len(docs)} documents, {len(texts)} distinct texts scored")

    # ---- scores ---------------------------------------------------------------------
    rx = np.array([D3.regex_score(t) for t in texts], float)
    docs["regex_score"] = rx[inv]
    ppl_ok = False
    docs["ppl_score"] = np.nan
    if not a.no_ppl:
        try:
            eng = D3.PPL(a.ppl_model, a.window, a.stride)
            vals = []
            for i, t in enumerate(texts):
                if i % 100 == 0:
                    out(f"[ppl] {i}/{len(texts)}")
                try:
                    vals.append(eng.max_window_ppl(t))
                except Exception:
                    vals.append(float("nan"))
            docs["ppl_score"] = np.asarray(vals, float)[inv]
            ppl_ok = True
        except Exception as e:
            out(f"[ppl] UNAVAILABLE ({type(e).__name__}: {e}); regex only. "
                f"Pass --ppl-model <local gpt2 dir>.")

    pred_csv = A.find_insertion_predictions("D3_ppl_regex")
    out(f"[thresh] mode={a.thresholds}   insertion predictions: {pred_csv or 'NOT FOUND'}")

    variants = [("regex_only", "D3 Regex Only", "regex_score")]
    if ppl_ok:
        variants.append(("ppl_only", "D3 PPL Only (GPT-2)", "ppl_score"))
    for variant, label, col in variants:
        s = docs[col].values.astype(float)
        t, info, mask = A.resolve_threshold(a.thresholds, docs, s, pred_csv, col, 0.5)
        d = docs.copy()
        d["score"] = s
        d["pred"] = (np.nan_to_num(s, nan=-np.inf) >= t).astype(int)
        out(f"[thresh] {variant}: {t:.6g}   {info}")
        out.write_scores(d, variant, label, {"threshold": t, "threshold_info": info},
                         mask, extra_cols=("regex_score", "ppl_score"))

    if ppl_ok:
        info = {"mode": a.thresholds}
        mask = np.ones(len(docs), bool)
        if a.thresholds == "insertion" and pred_csv:
            ins = pd.read_csv(pred_csv, low_memory=False,
                              usecols=lambda c: c in ("regex_score", "ppl_score", "label", "split"))
            te, va = ins[ins["split"] == "test"], ins[ins["split"] == "val"]
            rmax = max(1e-9, float(te["regex_score"].max()))
            F = ecdf(te["ppl_score"].values.astype(float))
            s = docs["regex_score"].values / rmax + F(docs["ppl_score"].values)
            vs = (va["regex_score"] / max(1e-9, va["regex_score"].max())
                  + va["ppl_score"].rank(pct=True).fillna(0)).values
            C = A.parent_import("common")
            _, t = C.recall_at_fpr(va["label"].values, vs, 0.05)
            info.update({"from": pred_csv, "regex_max_insertion_test": rmax,
                         "ppl_percentile": "ECDF of insertion test PPL"})
        else:
            if a.thresholds == "insertion":
                info["mode"] = "own (fallback)"
            s = (docs["regex_score"] / max(1e-9, docs["regex_score"].max())
                 + docs["ppl_score"].rank(pct=True).fillna(0)).values
            t = A.threshold_own(docs, s)
            mask = docs["split"].values != "val"
        if not (t == t and np.isfinite(t)):
            t = float("inf")
            info["note"] = "no finite threshold met FPR<=5% on val -> flags nothing"
        d = docs.copy()
        d["score"] = s
        d["pred"] = (s >= t).astype(int)
        out(f"[thresh] combined: {t:.6g}   {info}")
        out.write_scores(d, "combined", "D3 Regex + PPL Combined",
                         {"threshold": t, "threshold_info": info}, mask,
                         extra_cols=("regex_score", "ppl_score"))

    # what fired, so a nonzero regex recall on ARGAR can be read rather than guessed
    from collections import Counter
    out("\n--- regex patterns that fired, by source and label " + "-" * 30)
    for (src, lab), g in docs.groupby(["source", "label"]):
        cnt = Counter(h for t in g["text"] for h in D3.regex_hits(t))
        out(f"  {src} label={lab} (n={len(g)}): " +
            (", ".join(f"{n}x {p}" for p, n in cnt.most_common(4)) or "nothing"))
    out.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
