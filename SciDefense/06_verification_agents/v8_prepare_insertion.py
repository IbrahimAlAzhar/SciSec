#!/usr/bin/env python3
"""
v8_prepare_insertion.py — insertion TEST pairs for a reviewer that has NO published
reference run (gpt-4o-mini), built with V5's OWN corpus code.

For gemma / qwen / mistral the PBS runs V5's prepare_comparison.py UNCHANGED: it builds
the corpus and proves, against the V4 zero-shot stage1_detail.csv, that the documents are
the ones behind your published rows. gpt-4o-mini has no such reference file, so that
proof is impossible and prepare_comparison.py refuses. This file runs the SAME builder
(comparison_input.build -> def_dataset.build_eval_set, the all-papers branch) and writes
the same eval_records.jsonl + eval_manifest.json, with `reference: null` recorded in the
manifest so nobody later mistakes the gpt corpus for a verified one.

    python v8_prepare_insertion.py --attacked-csv .../gpt-4o-mini/seed2/metrics_detail.csv \
        --benign-csv .../dataset_primary_500.csv --out-dir inputs/gpt-4o-mini/prepared
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project-root", default=os.environ.get(
        "AD_ROOT", "."))
    p.add_argument("--v5-dir", default=os.environ.get("V5_DIR", ""))
    p.add_argument("--benign-csv", required=True)
    p.add_argument("--attacked-csv", required=True)
    p.add_argument("--reference-detail", default="",
                   help="checked when given and present; otherwise recorded as absent")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--id-column", default="id")
    p.add_argument("--max-papers", type=int, default=1000000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    # identical settings to prepare_comparison.py
    a.text_column = "input_text_cleaned"
    a.sections_column = "pdf_text"
    a.author_lim_column = "author_men_lim"
    a.title_column = "title"
    a.slot = "limitations_end"
    a.max_chars = 30000
    v5 = a.v5_dir or str(Path(a.project_root) / "defense/llm_agents_v5_final")
    if v5 not in sys.path:
        sys.path.insert(0, v5)
    os.environ["AD_ROOT"] = a.project_root
    import comparison_input as I
    I.setup_paths(a.project_root)
    import def_dataset as DS
    if not os.path.realpath(DS.__file__).startswith(os.path.realpath(v5) + os.sep):
        raise SystemExit(f"[FATAL] def_dataset resolved to {DS.__file__}, not {v5}. "
                         f"Put V5 first on PYTHONPATH.")
    records = I.build(a, DS)
    ref = None
    if a.reference_detail and os.path.exists(a.reference_detail):
        ref = I.check_reference(records, a.reference_detail)
        print(f"[input] reference matched: {a.reference_detail}")
    else:
        print("[input] NO reference run for this reviewer: the corpus is built by V5's code "
              "but cannot be verified against a published row (recorded in the manifest)")
    src = {k: {"path": str(Path(v).resolve()), "sha256": I.sha256(v)} for k, v in
           [("benign_csv", a.benign_csv), ("metrics_detail", a.attacked_csv),
            ("def_dataset", DS.__file__)]}
    rep = I.save_records(records, a.out_dir, src, ref, dict(vars(a), built_by="v8_prepare_insertion.py"))
    rep["reference_verified"] = ref is not None
    (Path(a.out_dir) / "eval_manifest.json").write_text(json.dumps(rep, indent=2, default=str) + "\n")
    print(f"[saved] {a.out_dir}/eval_records.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
