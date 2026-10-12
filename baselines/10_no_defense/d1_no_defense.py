#!/usr/bin/env python3
"""
d1_no_defense.py — the reference row. No detector, no mitigation, nothing flagged.

    python d1_no_defense.py

WHY A ROW THAT DOES NOTHING IS WORTH RUNNING
    Every other baseline is read against this one. It fixes, on YOUR corpus with YOUR
    threshold protocol:

      * what the attacks do to the rating, per family
      * how much of the clean review's criticism survives, per family      <- the number
        your paper turns on
      * the trivial-detector floor: flag nothing (recall 0, FPR 0) and flag everything
        (recall 1, FPR 1). Any baseline that does not beat BOTH of those is not a
        detector. Your earlier single-stage screen scored F1 0.909 at FPR 1.000, which
        is exactly the flag-everything row -- printed here so the comparison is explicit
        rather than something a reviewer discovers.
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

import common as C


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--wide-csv", default=C.WIDE_CSV)
    p.add_argument("--detail-csv", default=C.DETAIL_CSV)
    p.add_argument("--doc-col", default="")
    p.add_argument("--max-papers", type=int, default=0)
    p.add_argument("--embed-model", default="",
                   help="local sentence-transformers path; TF-IDF cosine if omitted")
    p.add_argument("--out-dir", default="")
    a = p.parse_args()

    R = C.Report("D1_no_defense", a.out_dir)
    df = C.build_eval_frame(a.wide_csv, a.detail_csv, a.doc_col, a.max_papers)
    R(f"[data] {len(df)} rows, {df.paper_id.nunique()} papers")

    # ---- the two trivial detectors -------------------------------------------------
    for name, pred in (("flag_nothing", 0), ("flag_everything", 1)):
        d = df.copy()
        d["pred"] = pred
        d["score"] = float(pred)
        m = C.evaluate(d, calls_per_paper=0.0)
        R.table({k: v for k, v in m.items() if not k.startswith("cond:")},
                f"D1 / {name}")
        # aggregate.py reads these; the two trivial detectors are the floor every other
        # baseline has to clear, so they belong in the same table as the real ones.
        R.save_json(m, f"metrics_{name}.json")
        if name == "flag_everything":
            R("")
            R("  ^ note the FPR column: 1.000. An F1 of 0.9 here means nothing, because")
            R("    every honest manuscript is accused. This is the row every detector")
            R("    below must beat.")

    # ---- what the attacks did ------------------------------------------------------
    atk = df[df.label == 1]
    R("")
    R("--- attack effect, no defense " + "-" * 62)
    g = atk.groupby(["family", "condition"], as_index=False).agg(
        n=("paper_id", "size"),
        d_rating=("d_rating", "mean"),
        asr=("asr", "mean"),
        rating=("rating", "mean"),
        n_chars=("n_chars", "mean"))
    base = df.loc[df.label == 0, "base_rating"].mean()
    R(f"{'family':13s}{'condition':20s}{'n':>5s}{'d_rating':>10s}{'ASR':>8s}"
      f"{'rating':>9s}{'chars':>9s}")
    R("-" * 74)
    for _, r in g.iterrows():
        R(f"{r['family']:13s}{r['condition']:20s}{r['n']:5.0f}{r['d_rating']:10.3f}"
          f"{r['asr']:8.3f}{r['rating']:9.3f}{r['n_chars']:9.0f}")
    R(f"\n  clean baseline rating: {base:.3f}")

    # ---- did the criticism survive? ------------------------------------------------
    emb = C.Embedder(a.embed_model)
    w = C.weakness_report(df, emb)
    R("")
    R("--- weakness survival (1.0 = review says the same as on the clean paper) " + "-" * 18)
    R(f"{'family':13s}{'condition':20s}{'n':>5s}{'ngram':>9s}{'embed':>9s}"
      f"{'gt_ngram':>10s}{'d_rating':>10s}")
    R("-" * 76)
    for _, r in w.iterrows():
        R(f"{r['family']:13s}{r['condition']:20s}{r['n']:5.0f}"
          f"{r.get('w_ngram', float('nan')):9.3f}{r.get('w_embed', float('nan')):9.3f}"
          f"{r.get('gt_ngram', float('nan')):10.3f}"
          f"{r.get('d_rating', float('nan')):10.3f}")

    R("")
    R("HOW TO READ THE TWO BLOCKS TOGETHER")
    R("  If a family has d_rating near zero but LOW weakness overlap, the attack changed")
    R("  no score and still removed the criticism. That is suppression, and it is")
    R("  invisible to every rating-based metric in the literature.")
    fam = w.groupby("family").agg(d_rating=("d_rating", "mean"),
                                  w_ngram=("w_ngram", "mean"))
    R("")
    for f_, r in fam.iterrows():
        R(f"  {f_:13s} mean d_rating {r['d_rating']:+.3f}   "
          f"mean weakness overlap {r['w_ngram']:.3f}")

    # the miss/gtrec columns your run already computed, if they came through the join
    for c in ("miss", "gtrec", "miss_llm", "gtrec_llm"):
        if c in df.columns and df[c].notna().any():
            R("")
            R(f"--- {c} from metrics_detail, by family " + "-" * 40)
            for f_, v in df[df.label == 1].groupby("family")[c].mean().items():
                base_v = df.loc[df.label == 0, c].mean()
                R(f"  {f_:13s} {v:.3f}   (clean baseline {base_v:.3f}, "
                  f"delta {v-base_v:+.3f})")
            R("  ^ read as a DELTA against the clean baseline, never against zero.")

    R.save(df.drop(columns=["text", "doc_clean"]), "rows_nodefense.csv")
    R.save(w, "weakness_by_condition.csv")
    R.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())