#!/usr/bin/env python3
"""
ab_d1_trivial.py — D1 on ARGAR/BTR: the flag-nothing and flag-everything floors.

    python ab_d1_trivial.py
    python ab_d1_trivial.py --max-rows 5 --run-tag pilot5

Every detector below has to beat BOTH rows. Flag-everything has recall 1.000 and FPR 1.000;
its F1 looks respectable on a balanced pair set and means nothing, which is why it is
printed in the same table as the real detectors.
"""

from __future__ import annotations

import argparse

import ab_common as A


def main() -> int:
    ap = A.add_common_args(argparse.ArgumentParser(description=__doc__))
    a = ap.parse_args()
    out = A.Out("D1_trivial", a.run_tag)
    pairs = A.load_pairs(a.sources, a.subset, a.max_rows, a.pairs_csv)
    docs = A.docs_frame(pairs)
    out(f"[data] {len(docs)} documents ({int(docs.label.sum())} attacked)")
    for variant, label, v in (("flag_nothing", "D1 Flag Nothing", 0),
                              ("flag_everything", "D1 Flag Everything", 1)):
        d = docs.copy()
        d["pred"] = v
        d["score"] = float(v)
        out.write_scores(d, variant, label, {"thresholds": "none (constant detector)",
                                             "source_note": pairs.attrs.get("source_note")})
    out.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
