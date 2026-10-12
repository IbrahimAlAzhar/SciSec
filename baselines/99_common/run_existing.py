#!/usr/bin/env python3
"""
run_existing.py — run an ORIGINAL defense_baselines script, unchanged, on ARGAR/BTR.

    python run_existing.py d7_llamafirewall.py -- --some-flag x
    python run_existing.py d3_ppl_regex.py -- --ppl-model /path/gpt2

WHY THIS EXISTS
    Every script in defense_baselines/ loads its documents with common.build_eval_frame()
    and scores them with common.evaluate(). This wrapper imports that same `common`,
    replaces those two functions (and the output folder) so they serve the ARGAR/BTR test
    pairs, and then executes the original script as __main__. The detector code itself is
    not touched, so for a script I have not seen (your LlamaFirewall, PIGuard or
    PaperGuard runners) this is the most faithful way to get its ARGAR/BTR row.

    The frame has common's schema. One CLEAN row per paper (common's convention: FPR is
    measured on the unmodified manuscripts), and one attacked row per pair with
    family = "repackaging" (ARGAR) or "perturbation" (BTR) and condition = the variant.
    ins_start / ins_end are the gold diff offsets when the pair file has them, else -1.

    Note the protocol is the ORIGINAL script's: it re-tunes its threshold on 20% of the
    papers (common.stable_split) and reports the other 80%. The ab_*.py scripts default to
    the insertion-frozen threshold instead; say which one a table row used.

OUTPUT
    output/<RUN_TAG or default>/existing__<script name>/   (whatever the script writes)
"""

from __future__ import annotations

import json
import os
import runpy
import sys

import numpy as np
import pandas as pd

import ab_common as A


def frame(sources, subset, max_rows) -> pd.DataFrame:
    pairs = A.load_pairs(sources, subset, max_rows)
    rows, seen = [], set()
    for _, r in pairs.iterrows():
        pid = str(r["paper_id"])
        ben = str(r.get("benign_document") or "")
        base = dict(paper_id=pid, split=A.stable_split(pid),
                    base_rating=A._f(r.get("historical_base_rating")),
                    clean_weaknesses="", gt_limitation=str(r.get("reference_weaknesses") or ""),
                    doc_clean=ben)
        if pid not in seen:
            seen.add(pid)
            rows.append(dict(base, condition="CLEAN", family="clean", label=0, text=ben,
                             payload="", slot="none", ins_start=-1, ins_end=-1,
                             rating=base["base_rating"], d_rating=0.0, asr=0, weaknesses=""))
        try:
            g = json.loads(r.get("gold_offsets") or "[]")
        except Exception:
            g = []
        s0 = min((x[0] for x in g), default=-1)
        e0 = max((x[1] for x in g), default=-1)
        ra, rb = A._f(r.get("historical_attacked_rating")), base["base_rating"]
        asr = A._f(r.get("historical_asr"))
        rows.append(dict(base, condition=A.method_of(r), family=str(r["threat_family"]),
                         label=1, text=str(r.get("attacked_document") or ""),
                         payload=str(r.get("insert") or ""), slot="rewrite",
                         ins_start=s0, ins_end=e0, rating=ra,
                         d_rating=(ra - rb) if (ra is not None and rb is not None) else np.nan,
                         asr=int(asr > 0) if asr is not None else 0, weaknesses=""))
    df = pd.DataFrame(rows)
    df["n_chars"] = df["text"].str.len()
    print(f"[shim] {len(df)} rows: " + ", ".join(f"{k}={v}" for k, v in
                                                 df["family"].value_counts().items()))
    return df


def main() -> int:
    argv = sys.argv[1:]
    if not argv or argv[0].startswith("-"):
        print(__doc__)
        return 2
    script = argv[0]
    rest = argv[argv.index("--") + 1:] if "--" in argv else argv[1:]
    path = script if os.path.isabs(script) else os.path.join(A.DEF_ROOT, script)
    if not os.path.exists(path):
        print(f"[FATAL] {path} not found")
        return 1
    sources = os.environ.get("SOURCES", "argar btr").split()
    subset = os.environ.get("SUBSET", "test")
    max_rows = int(os.environ.get("MAX_ROWS", "0") or 0)
    run = os.environ.get("RUN_TAG", "") or "default"

    C = A.parent_import("common")
    cache = {}

    def build_eval_frame(*_a, **_k):
        if "df" not in cache:
            cache["df"] = frame(sources, subset, max_rows)
        return cache["df"].copy()

    orig_eval = C.evaluate

    def evaluate(df, *a, **k):
        res = orig_eval(df, *a, **k)
        neg = df[df.label == 0]
        for fam in ("repackaging", "perturbation"):
            sub = df[df.family == fam]
            if len(sub):
                both = sub.copy()
                both["family"] = "unmapped"          # reuse orig block for this slice
                res[fam] = orig_eval(pd.concat([both, neg]), *a, **k).get("unmapped", {})
        return res

    C.build_eval_frame = build_eval_frame
    C.evaluate = evaluate
    C.OUT_ROOT = os.path.join(A.OUT_ROOT, run)
    os.makedirs(C.OUT_ROOT, exist_ok=True)
    print(f"[shim] running {path} on ARGAR/BTR ({sources}, subset={subset}); "
          f"outputs under {C.OUT_ROOT}/<name the script chooses>")
    sys.argv = [path] + rest
    sys.path.insert(0, os.path.dirname(path))
    try:
        runpy.run_path(path, run_name="__main__")
    except SystemExit as e:
        return int(e.code or 0) if not isinstance(e.code, str) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
