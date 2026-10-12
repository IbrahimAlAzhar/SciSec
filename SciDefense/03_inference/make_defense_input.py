#!/usr/bin/env python3
"""
make_defense_input.py — attack data IN, defense input OUT. Nothing else.

This file contains no defense logic, no prompts, no thresholds, no model calls and no
network access. It reads the attack benchmark's own per-row output and writes the exact
input formats your three defense codebases already accept:

    llm_agents_v5_final/runner_def.py  --input-mode sft-dpo --eval-records <out>/eval_records.jsonl
    llim_agents_v4_SFT_DPO/ft_trajectories.py --documents <out>/train_documents.csv
    lim_agents_v7_RL/v7_score_rollouts.py      (consumes V4's samples.jsonl, built from the above)

WHY THIS FILE EXISTS
=======================================================================================
The document a defense sees is not the `insert` column and it is not the manuscript. It
is `anchor.render(insert, slot)` — the manuscript rebuilt with the payload spliced into
the limitations slot, then truncated at max_chars. If two codebases rebuild that document
with two different code paths, their numbers are not comparable, and no amount of care
elsewhere fixes it. v6 learned this the expensive way.

So this script does NOT re-implement rendering. It imports `def_dataset` from
llm_agents_v5_final and calls that module's own `read_record` / `read_anchor` /
`align.wrap_for_slot(...).render(...)`. The documents it writes are byte-identical to
v5's by construction, not by inspection. If those modules cannot be imported the script
FAILS — it never falls back to a local renderer, because a silent fallback that produces
documents differing by a newline is exactly the bug that is impossible to find later.

ONE ATTACK PER ROW
=======================================================================================
Attack data carries one family per row: `A5_injection` or `A2_reframing`, never both.
A paper contributes several rows — one clean plus one per family it was attacked with —
and each of those rows becomes its own document. The defender is handed one document at a
time, carrying at most one attack, with no condition label. Nothing in any output of this
script stacks two families into one document. (A7_stack / A9_pressure_stack are single
families that internally layer techniques; they are still one row, one condition.)

TWO LAYOUTS, AND THE DIFFERENCE MATTERS
=======================================================================================
--layout paired   one clean + ONE attacked document per paper, the attacked condition
                  assigned round-robin over families. This is the design your published
                  v5 / v4 numbers use. Balanced 50/50 by construction, so accuracy is not
                  inflated by class ratio. Use it when the new row has to sit in the same
                  table as a published row. doc_id is reproduced exactly ("<pid>|<cond>"),
                  so --reference-detail can PROVE the corpus is the published one.

--layout all      EVERY attack row becomes a document: 12 families x N papers. Use it for
                  per-family recall, for the RL rollout corpus, and for anything where you
                  need coverage rather than comparability. The clean arm is repeated once
                  per attacked row (v5's paired validator requires strict clean/attack
                  pairs), which costs clean inference calls and means the FPR denominator
                  is rows, not papers — dedupe by doc_id at report time. train_documents.csv
                  emits each clean document ONCE, matching ft_attack.py.

THE SPLIT GUARD
=======================================================================================
The attack memory is built from INDEX papers; the defense is evaluated on EVAL papers.
If a paper appears in both, the detector looks up the string it is being tested on and
reports a lookup as a detection. --papers eval (the default for eval_records) applies
v5's own split_papers with the same seed, so the guard holds across codebases. Setting
--papers all is legitimate for the RL training corpus and illegitimate for a test row;
the script prints which one you asked for and records it in the manifest.

THE CAMPAIGN LAYOUT THIS READS
=======================================================================================
    lim_bb_multimodel_500_data/output_primary500/
        gemma3-27b-it/seed0/metrics_detail.csv      <- one reviewer, one payload draw
        gemma3-27b-it/seed1/metrics_detail.csv
        gemma3-27b-it/seed2/metrics_detail.csv
        mistral-small-3.1-24b/seed{0,1,2}/metrics_detail.csv
        qwen3-8b/seed{0,1,2}/metrics_detail.csv
        gpt-4o-mini/...

Discovery is recursive, so `--bench-dir output_primary500` finds all of them, and older
`output_<model>/metrics_detail.csv` campaigns too. Two axes then have to be chosen, and
neither is chosen for you silently:

    --reviewer-model   ONE model. base_rating / rating / asr belong to that reviewer;
                       pooling two models compares one model's attacked rating against
                       another's baseline, which measures nothing. (The attack MEMORY does
                       pool models, correctly — that is runner_def's job, not this file's.)
    --attack-seed      ONE seed, or --all-seeds. Three seeds are three independent payload
                       draws for the same (paper, condition): extra documents, not extra
                       papers, and not something to average. With --all-seeds the doc_id
                       carries @seed so nothing collides; --layout paired refuses it.

USAGE
=======================================================================================
    # 0. look first, write nothing — prints every campaign, its model, its seeds, its schema
    python make_defense_input.py --scan

    # 1. the comparable test corpus for v5 / v7 stage C
    python make_defense_input.py \
        --bench-dir   $AD_ROOT/lim_bb_multimodel_500_data/output_primary500 \
        --benign-csv  $AD_ROOT/data/final_data/dataset_primary_500.csv \
        --reviewer-model gemma3-27b-it --attack-seed seed0 \
        --out-dir     $AD_ROOT/defense/lim_agents_v7_RL/inputs/primary500_paired \
        --layout paired --papers eval \
        --reference-detail $AD_ROOT/defense/llm_agents_v5_final/runs/gemma3-27b-it/stage1_detail.csv

    # 2. full-coverage corpus: every family, every paper, all three seeds
    python make_defense_input.py \
        --bench-dir $AD_ROOT/lim_bb_multimodel_500_data/output_primary500 \
        --benign-csv $AD_ROOT/data/final_data/dataset_primary_500.csv \
        --reviewer-model gemma3-27b-it --all-seeds \
        --out-dir .../primary500_all --layout all --papers eval

    # 3. an RL / SFT training corpus from an attack campaign over the TRAIN papers.
    #    This needs a campaign that was actually RUN on those papers. If --scan shows
    #    only output_primary500, those 500 are your TEST papers and building a training
    #    corpus from them would train on the test set. Either run the attack over the
    #    secondary/tertiary papers first, or keep using ft_attack.py (which GENERATES
    #    payloads rather than reading them) for the train split.
    python make_defense_input.py \
        --bench-dir  $AD_ROOT/lim_bb_multimodel_500_data/output_secondary280 \
        --benign-csv $AD_ROOT/data/final_data/dataset_secondary_280.csv \
        --reviewer-model gemma3-27b-it --all-seeds \
        --out-dir    $AD_ROOT/defense/llim_agents_v4_SFT_DPO/traindata_shared \
        --emit train_documents --layout all --papers all

Then hand the output straight to the defense:

    python $V5_DIR/runner_def.py --mode defend --input-mode sft-dpo \
        --eval-records <out-dir>/eval_records.jsonl --out-dir runs/... [model flags]

    python $V4_DIR/ft_trajectories.py --documents <out-dir>/train_documents.csv ...
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from collections import Counter, OrderedDict
from typing import Dict, List, Optional, Sequence

# ======================================================================================
# import bootstrap — the one part that is allowed to be picky
# ======================================================================================
HERE = os.path.dirname(os.path.abspath(__file__))


def _ad_root() -> str:
    r = os.environ.get("AD_ROOT")
    if r and os.path.isdir(r):
        return os.path.abspath(r)
    # .../attack_def_lim_peer_rev/defense/<this dir>/make_defense_input.py
    cand = os.path.dirname(os.path.dirname(HERE))
    if os.path.isdir(os.path.join(cand, "defense")):
        return cand
    raise SystemExit("[fatal] set AD_ROOT to .../attack_def_lim_peer_rev")


AD_ROOT = _ad_root()


def _v5_dir() -> str:
    for c in (os.environ.get("V5_DIR"),
              os.path.join(AD_ROOT, "defense", "llm_agents_v5_final"),
              os.path.join(os.path.dirname(HERE), "llm_agents_v5_final")):
        if c and os.path.isfile(os.path.join(c, "def_dataset.py")):
            return os.path.abspath(c)
    raise SystemExit(
        "[fatal] llm_agents_v5_final/def_dataset.py not found.\n"
        "        Searched: $V5_DIR, $AD_ROOT/defense/llm_agents_v5_final,\n"
        "                  <this dir>/../llm_agents_v5_final\n"
        "        This script renders documents with V5's own code path on purpose. It has\n"
        "        no fallback renderer: a local copy would drift from V5 by a newline and\n"
        "        every comparison in the paper would quietly stop meaning anything.")


V5_DIR = _v5_dir()
V4_DIR = os.environ.get("V4_FT_DIR") or os.path.join(AD_ROOT, "defense",
                                                     "llim_agents_v4_SFT_DPO")

if V5_DIR not in sys.path:
    sys.path.insert(0, V5_DIR)

try:
    import comparison_input as CI          # no heavy imports at module level
except Exception as e:                     # pragma: no cover
    raise SystemExit(f"[fatal] cannot import comparison_input from {V5_DIR}: {e!r}")

CI.setup_paths(AD_ROOT)                    # puts the attack repos on sys.path

try:
    import def_dataset as DS               # this pulls in align + runner from the attack dirs
    import align
except Exception as e:
    raise SystemExit(
        f"[fatal] cannot import def_dataset/align: {type(e).__name__}: {e}\n"
        f"        sys.path entries that were tried:\n        " +
        "\n        ".join(p for p in sys.path if AD_ROOT in p) + "\n"
        "        Point WB_BB_DIR / WB_COMP_DIR / WB_FEAS_DIR at the attack directories, or\n"
        "        run this from a node where they are mounted. Do not work around this.")

import pandas as pd                        # noqa: E402  (after bootstrap, deliberately)

CLEAN = DS.CLEAN_CONDITION                 # "A0_clean" — never hardcoded twice


# ======================================================================================
# discovery / scan
# ======================================================================================
INSERT_ALIASES = ("insert", "payload", "payload_text", "inserted_text", "attack_text")
COND_ALIASES = ("condition", "family", "attack", "strategy_family")

SEED_RE = __import__("re").compile(r"^seed[-_]?(\d+)$", __import__("re").I)


def discover(bench_dirs: Sequence[str]) -> List[str]:
    """
    Every metrics_detail.csv at ANY depth under the given dirs. If none are given, sweep
    AD_ROOT for campaign directories.

    Recursive and layout-agnostic on purpose. Your campaigns are not consistently shaped:
        output_primary500/<model>/seed<N>/metrics_detail.csv     (the 500-paper run)
        output_<model>/metrics_detail.csv                        (older runs)
    A glob that encodes either one silently returns nothing on the other, and "nothing
    found" reads as "no data" when it means "wrong glob". A missing --bench-dir is now a
    hard error naming the path, not an empty list.
    """
    out: List[str] = []
    if bench_dirs:
        missing = [d for d in bench_dirs
                   if not (os.path.isdir(d) or os.path.isfile(d))]
        if missing:
            raise SystemExit(
                "[fatal] --bench-dir path(s) do not exist:\n        " +
                "\n        ".join(os.path.abspath(m) for m in missing) +
                "\n        An attack campaign has to have been RUN over those papers "
                "before\n        its payloads can be turned into defense input. Run "
                "--scan to see\n        which campaigns you actually have.")
        for d in bench_dirs:
            d = os.path.abspath(d)
            if os.path.isfile(d):
                out.append(d)
                continue
            out += sorted(glob.glob(os.path.join(d, "**", "metrics_detail.csv"),
                                    recursive=True))
    else:
        out += sorted(glob.glob(os.path.join(AD_ROOT, "lim_*", "**",
                                            "metrics_detail.csv"), recursive=True))
    seen, uniq = set(), []
    for p in out:
        rp = os.path.realpath(p)
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


def source_key(path: str) -> "tuple[str, str]":
    """
    (reviewer_model, seed) from the path, for both layouts this project uses:

        .../output_primary500/gemma3-27b-it/seed0/metrics_detail.csv -> (gemma3-27b-it, seed0)
        .../output_gemma3-27b-it/metrics_detail.csv                  -> (gemma3-27b-it, "")

    The seed is not cosmetic. Three seeds re-generate the payload for the SAME
    (paper, condition), so pooling them without saying so puts three different documents
    under one doc_id and the join downstream drops two of them.
    """
    parts = os.path.abspath(path).split(os.sep)
    d1 = parts[-2] if len(parts) >= 2 else ""
    d2 = parts[-3] if len(parts) >= 3 else ""
    if SEED_RE.match(d1):
        model = d2[len("output_"):] if d2.startswith("output_") else d2
        return model, d1.lower()
    model = d1[len("output_"):] if d1.startswith("output_") else d1
    return model, ""


def tag_of(path: str) -> str:
    m, s = source_key(path)
    return f"{m}@{s}" if s else m


def scan(paths: Sequence[str]) -> None:
    if not paths:
        print(f"[scan] no metrics_detail.csv found under {AD_ROOT}/lim_*.")
        return
    print(f"[scan] {len(paths)} attack file(s)\n")
    by_model: Dict[str, List[str]] = {}
    for p in paths:
        try:
            d = pd.read_csv(p, low_memory=False)
        except Exception as e:
            print(f"  {p}\n    UNREADABLE: {type(e).__name__}: {e}\n")
            continue
        model, seed = source_key(p)
        by_model.setdefault(model, []).append(seed or "-")
        ins = next((c for c in INSERT_ALIASES if c in d.columns), None)
        con = next((c for c in COND_ALIASES if c in d.columns), None)
        n_att = n_use = n_pap = 0
        conds = "?"
        if con:
            att = d[d[con].astype(str) != CLEAN]
            n_att = len(att)
            if ins:
                usable = att[att[ins].notna() &
                             (att[ins].astype(str).str.strip() != "")]
                n_use = len(usable)
                conds = ", ".join(sorted(set(usable[con].astype(str)))[:14])
                if "paper_id" in usable:
                    n_pap = usable["paper_id"].nunique()
        print(f"  {os.path.relpath(p, AD_ROOT)}")
        print(f"    model={model}  seed={seed or '(none)'}  rows={len(d)}  "
              f"attack_rows={n_att}  usable_insert={n_use}  papers={n_pap}")
        print(f"    insert column : {ins or 'MISSING -> unusable'}")
        print(f"    condition col : {con or 'MISSING -> unusable'}")
        miss = [c for c in ("paper_id", "base_rating", "rating", "d_rating", "asr",
                            "group", "slot", "strategy", "tier") if c not in d.columns]
        if miss:
            print(f"    missing (optional): {', '.join(miss)}")
        print(f"    conditions    : {conds}\n")
    print("[scan] summary — reviewer model -> seeds present")
    for m, seeds in sorted(by_model.items()):
        print(f"    {m:<26} {', '.join(sorted(set(seeds)))}")
    print("\n[scan] pick ONE model for the documents (--reviewer-model) and ONE seed\n"
          "       (--attack-seed seed0), or --all-seeds to use every seed of that model\n"
          "       as separate documents. --layout paired requires a single seed.")


# ======================================================================================
# loading the attack side
# ======================================================================================
def load_attack_frames(paths: Sequence[str], conditions: Optional[Sequence[str]]
                       ) -> "OrderedDict[str, pd.DataFrame]":
    """One frame per (reviewer model, seed). Attack rows only, usable insert only."""
    frames: "OrderedDict[str, pd.DataFrame]" = OrderedDict()
    for p in paths:
        tag = tag_of(p)
        # v5's load_attacked prints the basename of the parent dir, which is "seed0" in
        # this layout and useless with nine files. Name the source first.
        print(f"[data] {tag}  <- {os.path.relpath(p, AD_ROOT)}")
        d = DS.load_attacked(p)            # v5's own loader: drops clean + empty inserts
        if conditions:
            keep = set(conditions)
            before = len(d)
            d = d[d["condition"].astype(str).isin(keep)]
            print(f"[data] {tag}: --conditions kept {len(d)}/{before} rows")
        if d.empty:
            print(f"[data] {tag}: no usable rows — skipped")
            continue
        model, seed = source_key(p)
        d = d.copy()
        d["_model"], d["_seed"], d["_src"] = model, (seed or "-"), p
        frames[tag] = d
    if not frames:
        raise SystemExit("[fatal] no usable attack rows in any file found. Run --scan.")
    return frames


def select_frame(frames: "OrderedDict[str, pd.DataFrame]", prefer_model: Optional[str],
                 prefer_seed: Optional[str], all_seeds: bool, layout: str
                 ) -> "tuple[str, pd.DataFrame]":
    """
    The document corpus comes from ONE reviewer model, not from a pool of models.

    Pooling models would be wrong here and is right in the memory. The payload TEXT is
    shared across reviewer models by construction, so pooling it into the memory adds
    coverage without leakage (that is what build_memory does). But `base_rating`, `rating`,
    `d_rating` and `asr` are that reviewer's own numbers. Pool those and stage 2's
    restoration metric compares a rating from one model against a baseline from another,
    which measures nothing.

    Seeds are a second axis. Within one model, seed0/1/2 are three independent payload
    draws for the same (paper, condition), so they are legitimate extra DOCUMENTS but not
    extra papers — and they cannot be averaged into one row. Default: one seed.
    """
    models: "OrderedDict[str, List[str]]" = OrderedDict()
    for tag, d in frames.items():
        models.setdefault(str(d["_model"].iloc[0]), []).append(tag)

    if prefer_model:
        if prefer_model not in models:
            raise SystemExit(f"[fatal] --reviewer-model {prefer_model!r} not found. "
                             f"Available: {sorted(models)}")
        model = prefer_model
    else:
        model = next(iter(models))
        if len(models) > 1:
            print(f"[data] {len(models)} reviewer models found {sorted(models)}; "
                  f"documents built from {model!r} — its base_rating/rating/asr are what "
                  f"stage 2 scores restoration against. Pass --reviewer-model to choose.")
    tags = models[model]
    seeds = sorted({str(frames[t]["_seed"].iloc[0]) for t in tags})

    if all_seeds:
        if layout == "paired":
            raise SystemExit(
                "[fatal] --all-seeds cannot be combined with --layout paired.\n"
                "        Paired assigns ONE condition per paper round-robin; with three "
                "seeds the\n        same (paper, condition) appears three times and two "
                "of them would be\n        silently discarded. Use --layout all, or pick "
                "one seed.")
        frame = pd.concat([frames[t] for t in tags], ignore_index=True)
        print(f"[data] model={model} seeds={seeds}: {len(frame)} attack rows pooled as "
              f"separate documents (doc_id carries @seed)")
        return f"{model}@{'+'.join(seeds)}", frame

    if prefer_seed:
        want = prefer_seed.lower()
        hit = [t for t in tags if str(frames[t]["_seed"].iloc[0]) == want]
        if not hit:
            raise SystemExit(f"[fatal] --attack-seed {prefer_seed!r} not present for "
                             f"{model!r}. Seeds found: {seeds}")
        tag = hit[0]
    else:
        tag = sorted(tags)[0]
        if len(tags) > 1:
            print(f"[data] {model} has seeds {seeds}; using {str(frames[tag]['_seed'].iloc[0])!r}. "
                  f"Pass --attack-seed to choose, or --all-seeds (layout=all only) to use "
                  f"every draw as its own document.")
    return tag, frames[tag]


# ======================================================================================
# args namespace for V5's dataset code
# ======================================================================================
def dataset_args(a) -> argparse.Namespace:
    """
    The exact field set def_dataset / runner.row_record / runner.build_anchor read, with
    runner_def.py's defaults. Different values here mean different documents, so they are
    all surfaced as flags and all recorded in the manifest.
    """
    return argparse.Namespace(
        text_column=a.text_column, sections_column=a.sections_column,
        author_lim_column=a.author_lim_column, title_column=a.title_column,
        id_column=a.id_column, slot=a.slot, max_chars=a.max_chars,
        max_papers=a.max_papers, seed=a.seed, threat="insertion", conditions=None,
    )


# ======================================================================================
# layout: all — every attack row its own document
# ======================================================================================
def build_all(benign_csvs: Sequence[str], attacked: pd.DataFrame,
              keep_papers: Sequence[str], dargs, a) -> List["DS.DocRecord"]:
    """
    One document per (paper, condition), plus a clean twin per pair.

    Each record carries exactly one attack family. The clean document is rendered from the
    same anchor with an empty insert, so the ONLY difference between a pair is the inserted
    span — not tokenisation, truncation or formatting.
    """
    keep = {str(p) for p in keep_papers}
    multi_seed = attacked["_seed"].nunique() > 1 if "_seed" in attacked else False
    by_paper: Dict[str, Dict[str, pd.Series]] = {}
    for _, r in attacked.iterrows():
        pid = str(r["paper_id"])
        if pid not in keep:
            continue
        cond = str(r["condition"])
        seed = str(r.get("_seed") or "")
        # The key is what makes doc_id unique. With several seeds the same (paper,
        # condition) is a DIFFERENT payload draw and therefore a different document; keying
        # on condition alone would keep only the last draw and silently drop the rest.
        key = f"{cond}@{seed}" if (multi_seed and seed and seed != "-") else cond
        if key in by_paper.get(pid, {}):
            print(f"  [dup] {pid}/{key}: duplicate row in the attack CSV — keeping the "
                  f"first")
            continue
        by_paper.setdefault(pid, {})[key] = r

    ctl = align.control_slot(dargs.slot)
    records: List["DS.DocRecord"] = []
    n_papers = n_skip_anchor = n_skip_trunc = 0
    slot_mismatch = 0

    for csv_path in benign_csvs:
        df = _load_benign(csv_path, dargs)
        for i in range(len(df)):
            rec = DS.read_record(df, i, dargs)
            pid = str(rec["paper_id"])
            if pid not in by_paper:
                continue
            anchor = DS.read_anchor(rec, dargs)
            if anchor is None:
                n_skip_anchor += 1
                continue
            benign_doc = align.wrap_for_slot(anchor, ctl).render("", ctl, dargs.max_chars)
            if not benign_doc.strip():
                n_skip_anchor += 1
                continue

            emitted = 0
            for j, key in enumerate(sorted(by_paper[pid])):
                row = by_paper[pid][key]
                cond = str(row["condition"])       # the family, never the @seed suffix
                insert = str(row["insert"])
                slot = dargs.slot
                row_slot = str(row.get("slot") or "").strip()
                if row_slot and row_slot != dargs.slot:
                    slot_mismatch += 1
                    if a.use_row_slot:
                        slot = row_slot
                doc = align.wrap_for_slot(anchor, slot).render(insert, slot,
                                                              dargs.max_chars)
                if insert.strip() and insert.strip()[:60] not in doc:
                    # gold span truncated away: a positive with no span to localise. It
                    # would teach a model to flag a paper for text it cannot see.
                    print(f"  [drop] {pid}/{key}: insert not in the rendered document "
                          f"(max_chars={dargs.max_chars} truncation)")
                    n_skip_trunc += 1
                    continue
                records.append(DS.DocRecord(
                    doc_id=f"{pid}|{CLEAN}#{j}", paper_id=pid, row_index=i, label=0,
                    condition=CLEAN, group="control", document=benign_doc,
                    base_rating=_f(row.get("base_rating")), slot=slot,
                    max_chars=dargs.max_chars))
                records.append(DS.DocRecord(
                    doc_id=f"{pid}|{key}", paper_id=pid, row_index=i, label=1,
                    condition=cond, group=str(row.get("group", "")), document=doc,
                    insert=insert, base_rating=_f(row.get("base_rating")),
                    attacked_rating=_f(row.get("rating")),
                    d_rating=_f(row.get("d_rating")),
                    asr=(None if _missing(row.get("asr")) else int(row["asr"])),
                    slot=slot, max_chars=dargs.max_chars))
                emitted += 1
            if emitted:
                n_papers += 1
            if n_papers >= dargs.max_papers:
                break

    if slot_mismatch:
        print(f"[data] {slot_mismatch} row(s) record a slot different from --slot "
              f"{dargs.slot!r}; "
              + ("honoured (--use-row-slot)" if a.use_row_slot else
                 "IGNORED. The attack scored those rows at their own slot, so the "
                 "rendered document differs from the one the rating belongs to. Pass "
                 "--use-row-slot to follow the row, at the cost of comparability with "
                 "published rows."))
    print(f"[data] layout=all: {n_papers} papers -> {len(records)} documents "
          f"({len(records)//2} attacked, each carrying exactly one family)")
    if n_skip_anchor or n_skip_trunc:
        print(f"[data] skipped: {n_skip_anchor} paper(s) with no usable anchor, "
              f"{n_skip_trunc} row(s) truncated past their gold span")
    return records


def _load_benign(path: str, dargs) -> pd.DataFrame:
    import runner as RUN
    return RUN.load_df(path).reset_index(drop=True)


def _missing(v) -> bool:
    return DS._is_missing(v)


def _f(x):
    try:
        v = float(x)
        return None if v != v else v
    except (TypeError, ValueError):
        return None


# ======================================================================================
# layout: paired — reproduce the published design exactly
# ======================================================================================
def build_paired(benign_csvs: Sequence[str], attacked: pd.DataFrame,
                 keep_papers: Sequence[str], dargs, a) -> List["DS.DocRecord"]:
    """
    Straight through V5's build_eval_set, one call per benign CSV. No reimplementation:
    doc_id, ordering, round-robin condition assignment and the truncation guard are v5's.
    """
    records: List["DS.DocRecord"] = []
    for csv_path in benign_csvs:
        recs = DS.build_eval_set(csv_path, attacked, keep_papers, dargs,
                                 conditions=(list(a.conditions) if a.conditions else None),
                                 max_papers=dargs.max_papers, seed=dargs.seed)
        records += recs
    # row_index is per-CSV; re-key so pairs stay matched and indices stay unique
    for k in range(0, len(records), 2):
        records[k].row_index = records[k + 1].row_index = k // 2
    print(f"[data] layout=paired: {len(records)} documents "
          f"({len(records)//2} papers, 50/50 benign/attacked)")
    return records


# ======================================================================================
# train_documents.csv — V4's schema, clean rows once per paper
# ======================================================================================
def write_train_documents(records: Sequence["DS.DocRecord"], out_dir: str,
                          weak_by_paper: Dict[str, str]) -> str:
    """
    ft_trajectories.py reads label / document / insert / condition / paper_id (and an
    optional screen_score). The rest of ft_attack.py's columns are written too so
    ft_table.py and count_defense_labels.py keep working.

    Clean documents are emitted ONCE per paper here, unlike eval_records.jsonl. V5's
    paired validator demands strict clean/attack pairs; the rollout has no such constraint,
    and four identical clean documents in a training corpus is four times the sampling cost
    for one behaviour.
    """
    rows, seen_clean = [], set()

    def weak(pid: str) -> str:
        return (weak_by_paper.get(pid) or "")[:8000]

    for r in records:
        if r.label == 0:
            if r.paper_id in seen_clean:
                continue
            seen_clean.add(r.paper_id)
            rows.append(dict(paper_id=r.paper_id, source="attack_csv",
                             row_index=r.row_index, label=0, condition=CLEAN,
                             family=CLEAN, strategy="clean", tier="T0", harmful=0,
                             insert="", document=r.document,
                             n_weaknesses=_n_weak(weak(r.paper_id)),
                             weaknesses=weak(r.paper_id)))
        else:
            rows.append(dict(paper_id=r.paper_id, source="attack_csv",
                             row_index=r.row_index, label=1, condition=r.condition,
                             family=r.condition, strategy=r.condition,
                             tier="", harmful=int(bool(r.asr)),
                             insert=r.insert, document=r.document,
                             n_weaknesses=_n_weak(weak(r.paper_id)),
                             weaknesses=weak(r.paper_id)))
    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "train_documents.csv")
    df.to_csv(path, index=False)
    n_att = int((df["label"] == 1).sum())
    print(f"[out] train_documents.csv: {len(df)} documents "
          f"({n_att} attacked / {len(df)-n_att} clean) over "
          f"{df['paper_id'].nunique()} papers -> {path}")
    if df["paper_id"].nunique() < 50:
        print(f"[warn] only {df['paper_id'].nunique()} papers. A rollout over this is a "
              f"smoke test, not a training corpus — GRPO groups will be too few for the "
              f"advantage to mean anything.")
    return path


def _n_weak(s: str) -> int:
    if not s:
        return 0
    import re
    parts = [p for p in re.split(r"(?:\n\s*[-*\d]+[.)]?\s*)|(?:\n{2,})", s) if p.strip()]
    return len(parts)


def weaknesses_map(benign_csvs: Sequence[str], dargs, col: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not col:
        return out
    for p in benign_csvs:
        try:
            df = _load_benign(p, dargs)
        except Exception:
            continue
        if col not in df.columns:
            print(f"[data] {os.path.basename(p)}: no {col!r} column — weaknesses empty")
            continue
        for _, r in df.iterrows():
            pid = str(r.get(dargs.id_column) or r.get("paper_id") or "")
            if pid:
                out[pid] = str(r.get(col) or "")
    return out


# ======================================================================================
# flat corpus — for anything that wants one CSV
# ======================================================================================
def write_corpus(records: Sequence["DS.DocRecord"], out_dir: str) -> str:
    rows = []
    for r in records:
        rows.append(dict(doc_id=r.doc_id, paper_id=r.paper_id, row_index=r.row_index,
                         label=r.label, condition=r.condition, group=r.group,
                         harmful=int(bool(r.asr)), slot=r.slot,
                         n_chars=len(r.document), n_chars_insert=len(r.insert),
                         base_rating=r.base_rating, attacked_rating=r.attacked_rating,
                         d_rating=r.d_rating, asr=r.asr,
                         insert=r.insert, document=r.document))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "defense_corpus.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[out] defense_corpus.csv: {len(rows)} rows -> {path}")
    return path


# ======================================================================================
# verification
# ======================================================================================
def verify(records: Sequence["DS.DocRecord"]) -> Dict:
    """
    Four properties, each one a bug that has actually happened in this project.
    Failures raise: a warning gets read once and the table goes in the paper anyway.
    """
    CI.validate(records)                                     # pairing, labels, non-empty

    bad_span = [r.doc_id for r in records
                if r.label == 1 and r.insert.strip()
                and r.insert.strip() not in r.document]
    if bad_span:
        raise SystemExit(f"[fatal] {len(bad_span)} attacked document(s) do not contain "
                         f"their gold span; span_f1 would be undefined. "
                         f"First: {bad_span[:3]}")

    leaked = [r.doc_id for r in records if r.label == 0 and r.insert.strip()]
    if leaked:
        raise SystemExit(f"[fatal] clean record(s) carry a non-empty insert: {leaked[:3]}")

    ids = Counter(r.doc_id for r in records)
    dup = [k for k, v in ids.items() if v > 1]
    if dup:
        raise SystemExit(f"[fatal] duplicate doc_id(s): {dup[:5]}. Downstream joins key "
                         f"on doc_id, so a duplicate silently drops a row.")

    per_doc_fams = Counter()
    for r in records:
        if r.label == 1:
            per_doc_fams[r.doc_id] = 1
    info = dict(n=len(records), n_attacked=sum(r.label == 1 for r in records),
                n_papers=len({r.paper_id for r in records}),
                conditions=dict(Counter(r.condition for r in records if r.label == 1)),
                max_families_per_document=max(per_doc_fams.values() or [0]))
    print(f"[verify] ok — {info['n']} documents, {info['n_papers']} papers, "
          f"{len(info['conditions'])} families, "
          f"{info['max_families_per_document']} attack family per attacked document")
    return info


# ======================================================================================
# main
# ======================================================================================
def main() -> None:
    p = argparse.ArgumentParser(
        description="attack data -> defense input (v5 / v4 / v6-v7 RL). Nothing else.")
    p.add_argument("--scan", action="store_true",
                   help="print the schema of every attack CSV found and exit")
    p.add_argument("--bench-dir", dest="bench_dirs", nargs="+", default=[],
                   help="dir(s) holding output_<model>/metrics_detail.csv, or CSV paths")
    p.add_argument("--benign-csv", dest="benign_csvs", nargs="+", default=[],
                   help="the manuscript CSV(s) the attack ran on")
    p.add_argument("--out-dir", dest="out_dir", default=None)
    p.add_argument("--emit", default="eval_records",
                   help="comma list: eval_records,train_documents,corpus  (or 'all')")
    p.add_argument("--layout", choices=["paired", "all"], default="paired")
    p.add_argument("--papers", choices=["eval", "index", "all"], default=None,
                   help="which side of v5's paper split. default: eval for eval_records, "
                        "all for train_documents")
    p.add_argument("--index-frac", dest="index_frac", type=float, default=0.5)
    p.add_argument("--reviewer-model", dest="reviewer_model", default=None,
                   help="which reviewer's rows build the documents (its base_rating/asr "
                        "are what restoration is scored against). --scan lists them.")
    p.add_argument("--attack-seed", dest="attack_seed", default=None,
                   help="seed subdirectory of that model, e.g. seed0. Default: the "
                        "lowest one present.")
    p.add_argument("--all-seeds", dest="all_seeds", action="store_true",
                   help="use EVERY seed of the chosen model as separate documents "
                        "(--layout all only); doc_id carries @seed")
    p.add_argument("--conditions", nargs="+", default=None,
                   help="restrict to these families (e.g. A5_injection A2_reframing)")
    p.add_argument("--reference-detail", dest="reference_detail", default=None,
                   help="published stage1_detail.csv: PROVE the corpus is identical")
    p.add_argument("--use-row-slot", dest="use_row_slot", action="store_true",
                   help="render each row at the slot the attack recorded (breaks "
                        "comparability with published rows)")
    p.add_argument("--weaknesses-column", dest="weaknesses_column", default="weaknesses")
    p.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="build and verify, write nothing")
    # document-identity knobs — changing any of these changes the documents
    p.add_argument("--text-column", dest="text_column", default="input_text_cleaned")
    p.add_argument("--sections-column", dest="sections_column", default="pdf_text")
    p.add_argument("--author-lim-column", dest="author_lim_column",
                   default="author_limitations")
    p.add_argument("--title-column", dest="title_column", default="title")
    p.add_argument("--id-column", dest="id_column", default="paper_id")
    p.add_argument("--slot", default="limitations_end")
    p.add_argument("--max-chars", dest="max_chars", type=int, default=30000)
    p.add_argument("--max-papers", dest="max_papers", type=int, default=10 ** 9)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    paths = discover(a.bench_dirs)
    if a.scan:
        scan(paths)
        return
    if not paths:
        raise SystemExit("[fatal] no attack CSV found. Run --scan, then pass --bench-dir.")
    if not a.benign_csvs:
        raise SystemExit("[fatal] --benign-csv is required: the attack CSV holds the "
                         "payload, the manuscript CSV holds the paper, and the document "
                         "the defense sees is the two rendered together.")
    for c in a.benign_csvs:
        if not os.path.isfile(c):
            raise SystemExit(f"[fatal] benign CSV not found: {c}")
    if not a.out_dir and not a.dry_run:
        raise SystemExit("[fatal] --out-dir is required")

    emit = {"eval_records", "train_documents", "corpus"} if a.emit.strip() == "all" else \
        {t.strip() for t in a.emit.split(",") if t.strip()}
    unknown = emit - {"eval_records", "train_documents", "corpus"}
    if unknown:
        raise SystemExit(f"[fatal] unknown --emit target(s): {sorted(unknown)}")
    papers_side = a.papers or ("all" if emit == {"train_documents"} else "eval")
    if a.reference_detail:
        if a.layout != "paired":
            raise SystemExit("[fatal] --reference-detail only means something with "
                             "--layout paired; the published rows are the paired corpus.")
        if not os.path.isfile(a.reference_detail):
            raise SystemExit(f"[fatal] reference not found: {a.reference_detail}")

    dargs = dataset_args(a)
    print(f"[cfg] AD_ROOT={AD_ROOT}\n[cfg] V5_DIR={V5_DIR}\n"
          f"[cfg] layout={a.layout} papers={papers_side} emit={sorted(emit)} "
          f"slot={a.slot} max_chars={a.max_chars} seed={a.seed}")

    frames = load_attack_frames(paths, a.conditions)
    reviewer, attacked = select_frame(frames, a.reviewer_model, a.attack_seed,
                                      a.all_seeds, a.layout)

    all_papers = sorted({str(x) for x in attacked["paper_id"]})
    index_ids, eval_ids = DS.split_papers(all_papers, index_frac=a.index_frac,
                                          seed=a.seed)
    keep = {"eval": eval_ids, "index": index_ids, "all": all_papers}[papers_side]
    if papers_side == "all":
        print("[split] --papers all: the corpus includes papers whose payloads may also "
              "populate the attack memory. Correct for a TRAINING corpus, wrong for a "
              "test row — a detector would then look up the string it is being tested on.")

    builder = build_paired if a.layout == "paired" else build_all
    records = builder(a.benign_csvs, attacked, keep, dargs, a)
    if not records:
        raise SystemExit("[fatal] built 0 documents. Do the paper_ids in the attack CSV "
                         "match the --id-column of the benign CSV? Run --scan.")

    info = verify(records)

    if a.dry_run:
        print(f"[dry-run] nothing written. records_hash={CI.record_hash(records)}")
        return

    os.makedirs(a.out_dir, exist_ok=True)
    used_srcs = sorted({str(x) for x in attacked["_src"]}) if "_src" in attacked else paths
    sources = {f"attacked_csv[{tag_of(pp)}]": {"path": os.path.abspath(pp),
                                               "sha256": CI.sha256(pp)}
               for pp in used_srcs}
    sources.update({f"benign_csv[{i}]": {"path": os.path.abspath(c),
                                         "sha256": CI.sha256(c)}
                    for i, c in enumerate(a.benign_csvs)})

    reference = None
    if a.reference_detail:
        try:
            reference = CI.check_reference(records, a.reference_detail)
        except ValueError as e:
            raise SystemExit(
                f"[fatal] this corpus is NOT the published one: {e}\n"
                f"        Nothing was written. The published row and a row measured on\n"
                f"        this corpus cannot go in the same table.\n"
                f"        Usual causes, in order of likelihood:\n"
                f"          --seed differs from the published run (it sets the paper split\n"
                f"            AND the round-robin condition assignment)\n"
                f"          --index-frac differs ({a.index_frac} here)\n"
                f"          --benign-csv is a different manuscript CSV, or its rows moved\n"
                f"          the attack CSV gained or lost rows since the published run\n"
                f"          --papers {papers_side} (the published rows are the eval side)")
        print(f"[verify] corpus IS the published one — {reference['rows']} rows match on "
              f"ordered doc_id/paper_id/condition/label. A new row measured here is "
              f"comparable to the published rows.")

    settings = dict(input_mode="sft-dpo", layout=a.layout, papers=papers_side,
                    reviewer_model=reviewer,
                    attack_seeds=(sorted({str(x) for x in attacked["_seed"]})
                                  if "_seed" in attacked else []),
                    index_frac=a.index_frac,
                    conditions=(list(a.conditions) if a.conditions else "all"),
                    slot=a.slot, use_row_slot=bool(a.use_row_slot),
                    max_chars=a.max_chars, seed=a.seed,
                    columns=dict(text=a.text_column, sections=a.sections_column,
                                 author_lim=a.author_lim_column, title=a.title_column,
                                 id=a.id_column),
                    attacks_per_document=1,
                    rendered_by=f"{V5_DIR}/def_dataset.py + align.render",
                    verify=info)

    if "eval_records" in emit:
        CI.save_records(records, a.out_dir, sources, reference, settings)
        print(f"[out] eval_records.jsonl / defense_inputs.jsonl / eval_manifest.json "
              f"-> {a.out_dir}")
    if "train_documents" in emit:
        write_train_documents(records, a.out_dir,
                              weaknesses_map(a.benign_csvs, dargs, a.weaknesses_column))
    if "corpus" in emit:
        write_corpus(records, a.out_dir)

    (open(os.path.join(a.out_dir, "input_settings.json"), "w")
     .write(json.dumps(dict(settings=settings, sources=sources), indent=2) + "\n"))

    print("\n[next] hand it to the defense — the corpus is now the only thing held fixed:")
    if "eval_records" in emit:
        print(f"  python {V5_DIR}/runner_def.py --mode defend --input-mode sft-dpo \\\n"
              f"      --eval-records {os.path.join(a.out_dir, 'eval_records.jsonl')} \\\n"
              f"      --out-dir <run dir> --backend vllm --vllm-url $VLLM_URL "
              f"--vllm-model <model-or-adapter>")
    if "train_documents" in emit:
        print(f"  python {V4_DIR}/ft_trajectories.py --model <key> --n-samples 4 \\\n"
              f"      --documents {os.path.join(a.out_dir, 'train_documents.csv')}")


if __name__ == "__main__":
    main()
