#!/usr/bin/env python3
"""
sr_inputs.py — stage I: the documents SafeReview trains and is tested on, which are the
SAME files your v7ab rows use.

    python sr_inputs.py --models gemma3-27b-it qwen3-8b mistral-small-3.1-24b gpt-4o-mini
    python sr_inputs.py --models gemma3-27b-it --max-rows 5 --run-tag pilot5

--mode auto (default)
    For each model: if lim_agents_v7_RL_with_argar_btr/inputs/<model>/ already has
    eval_records[__TAG].jsonl + eval_index[__TAG].csv, COPY them here and check the copy's
    sha256 against the source. Same for inputs/train_documents.csv. Otherwise BUILD them by
    running v7ab_inputs.py itself with RL_WORK pointed at this directory.
--mode copy / --mode build force one path.

Nothing is re-rendered and no document is edited: the documents SafeReview is scored on
are byte-identical to the ones every other row is scored on, and the manifest records the
hashes so a later rebuild of the v7ab inputs cannot silently change this table.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from typing import Dict, List

import sr_config as S
from sr_data import sha256

ALL_MODELS = ["gemma3-27b-it", "qwen3-8b", "mistral-small-3.1-24b", "gpt-4o-mini"]


def _copy(src: str, dst: str) -> Dict:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".tmp"
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)
    a, b = sha256(src), sha256(dst)
    if a != b:
        raise SystemExit(f"[FATAL] copy of {src} does not match its source")
    return {"file": dst, "source": src, "sha256": b}


def v7ab_paths(model: str, tag: str) -> List[str]:
    d = os.path.join(S.V7AB_DIR, "inputs", model)
    sfx = f"__{tag}" if tag else ""
    return [os.path.join(d, f"eval_records{sfx}.jsonl"), os.path.join(d, f"eval_index{sfx}.csv")]


def build(models: List[str], a) -> None:
    script = os.path.join(S.V7AB_DIR, "v7ab_inputs.py")
    if not os.path.exists(script):
        raise SystemExit(f"[FATAL] {script} not found; cannot build the inputs")
    cmd = [sys.executable, "-u", script, "--models", *models, "--max-rows", str(a.max_rows),
           "--train-cap", str(a.train_cap)]
    if a.sources:
        cmd += ["--sources", *a.sources]
    if a.run_tag:
        cmd += ["--run-tag", a.run_tag, "--no-train"]
    env = dict(os.environ, RL_WORK=S.WORK, V8_DATA_DIR=S.V8_DATA_DIR)
    print("[build] " + " ".join(cmd) + f"   (RL_WORK={S.WORK})")
    rc = subprocess.call(cmd, env=env, cwd=S.V7AB_DIR)
    if rc != 0:
        raise SystemExit(f"[FATAL] v7ab_inputs.py exited {rc}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=ALL_MODELS)
    ap.add_argument("--mode", choices=["auto", "copy", "build"], default="auto")
    ap.add_argument("--max-rows", dest="max_rows", type=int, default=0)
    ap.add_argument("--sources", nargs="*", default=[])
    ap.add_argument("--run-tag", dest="run_tag", default="")
    ap.add_argument("--train-cap", dest="train_cap", type=int, default=0)
    a = ap.parse_args()

    if a.max_rows and not a.run_tag:
        raise SystemExit("[FATAL] --max-rows without --run-tag would overwrite the full test "
                         "set with a pilot. Pass --run-tag (e.g. pilot5).")
    man: Dict = {"work": S.WORK, "v7ab": S.V7AB_DIR, "v8_data": S.V8_DATA_DIR,
                 "run_tag": a.run_tag, "files": []}
    to_build: List[str] = []
    for m in a.models:
        src = v7ab_paths(m, a.run_tag)
        have = all(os.path.exists(p) for p in src)
        if a.mode == "build" or (a.mode == "auto" and not have) or a.max_rows:
            to_build.append(m)
            continue
        if not have:
            raise SystemExit(f"[FATAL] --mode copy but {src} missing")
        for p in src:
            man["files"].append(_copy(p, os.path.join(S.WORK, "inputs", m,
                                                      os.path.basename(p))))
        print(f"[copy] {m}: {src[0]} (+ index)")
    tr_src = os.path.join(S.V7AB_DIR, "inputs", "train_documents.csv")
    tr_dst = S.wpath(S.D_INPUTS, "train_documents.csv")
    if not a.run_tag:
        if a.mode != "build" and os.path.exists(tr_src):
            man["files"].append(_copy(tr_src, tr_dst))
            print(f"[copy] train_documents.csv <- {tr_src}")
        elif not to_build:
            to_build.append(a.models[0])      # the build writes train_documents.csv too
    if to_build:
        build(to_build, a)
        for m in to_build:
            r, i = S.inputs_paths(m, a.run_tag)
            for p in (r, i):
                if os.path.exists(p):
                    man["files"].append({"file": p, "source": "v7ab_inputs.py (built here)",
                                         "sha256": sha256(p)})
        if os.path.exists(tr_dst) and not any(f["file"] == tr_dst for f in man["files"]):
            man["files"].append({"file": tr_dst, "source": "v7ab_inputs.py (built here)",
                                 "sha256": sha256(tr_dst)})
    for m in a.models:
        r, i = S.inputs_paths(m, a.run_tag)
        if os.path.exists(r):
            n = sum(1 for l in open(r, encoding="utf-8") if l.strip())
            print(f"[test]  {m}: {n} documents  {r}")
        else:
            print(f"[warn]  {m}: no test records at {r}")
    if os.path.exists(tr_dst):
        import pandas as pd
        tr = pd.read_csv(tr_dst, usecols=lambda c: c in ("paper_id", "label", "input_source"))
        print(f"[train] {tr_dst}: {len(tr)} documents, {tr['paper_id'].nunique()} papers")
        if "input_source" in tr.columns:
            print(tr.groupby(["input_source", "label"]).size().unstack(fill_value=0).to_string())
    sfx = f"__{a.run_tag}" if a.run_tag else ""
    os.makedirs(S.wpath(S.D_INPUTS), exist_ok=True)
    with open(S.wpath(S.D_INPUTS, f"manifest_sr{sfx}.json"), "w") as f:
        json.dump(man, f, indent=2)
    print(f"[saved] {S.wpath(S.D_INPUTS, f'manifest_sr{sfx}.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
