#!/usr/bin/env python3
"""
sr_data.py — the training corpus and the test sets, read in the formats v7ab writes.

    TRAIN  inputs/train_documents.csv   (V4 schema; v7ab_inputs.build_train)
           one clean row per paper_id (label 0) + its attacked rows (label 1, gold in
           `insert`). ARGAR/BTR paper ids are namespaced "argar:<id>" / "btr:<id>".
    TEST   inputs/<model>/eval_records[__TAG].jsonl  (V5 DocRecord; consecutive clean /
           attacked pairs) + eval_index[__TAG].csv (eval_index -> pair_uid, input_source,
           data_origin, threat_family, harmful, seen_in_train).

The test documents are never re-rendered here: they are read, and the row number in the
records file IS eval_index, exactly as runner_def writes it into stage1_detail.csv.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from typing import Dict, List, Sequence, Set, Tuple

import pandas as pd


def _s(x) -> str:
    return "" if x is None or (isinstance(x, float) and x != x) else str(x)


def raw_pid(pid: str) -> str:
    """'argar:123' -> '123'. v7ab_inputs namespaces ARGAR/BTR training papers."""
    s = _s(pid)
    for p in ("argar:", "btr:"):
        if s.startswith(p):
            return s[len(p):]
    return s


# ======================================================================================
# TRAIN
# ======================================================================================
def load_train(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise SystemExit(f"[FATAL] {path} not found. Run the inputs stage first:\n"
                         f"            qsub jobs/job_sr_i_inputs.pbs")
    df = pd.read_csv(path, low_memory=False)
    need = {"paper_id", "label", "condition", "document"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"[FATAL] {path} is missing {sorted(miss)}")
    df["paper_id"] = df["paper_id"].astype(str)
    df["label"] = df["label"].astype(int)
    for c in ("insert", "input_source", "threat_family", "pair_uid"):
        if c not in df.columns:
            df[c] = ""
    df["document"] = df["document"].fillna("").astype(str)
    df["insert"] = df["insert"].fillna("").astype(str)
    df["input_source"] = df["input_source"].fillna("insertion").astype(str)
    return df


def clean_docs(train: pd.DataFrame) -> Dict[str, Dict]:
    """paper_id -> {document, input_source} for the clean (label 0) documents: the
    paper set P that the SafeReview attacker attacks."""
    out: Dict[str, Dict] = {}
    for _, r in train[train["label"] == 0].iterrows():
        d = _s(r["document"])
        if d.strip() and r["paper_id"] not in out:
            out[r["paper_id"]] = {"document": d, "input_source": _s(r["input_source"])}
    return out


def attacked_pairs(train: pd.DataFrame) -> List[Dict]:
    """The EXISTING attack corpus as (clean twin, attacked document) pairs: what the cheap
    Static-DPO row trains on."""
    clean = clean_docs(train)
    out = []
    for _, r in train[train["label"] == 1].iterrows():
        pid = r["paper_id"]
        if pid not in clean or not _s(r["document"]).strip():
            continue
        out.append({"paper_key": pid, "clean": clean[pid]["document"],
                    "attacked": _s(r["document"]), "insert": _s(r["insert"]),
                    "condition": _s(r["condition"]),
                    "input_source": _s(r["input_source"]),
                    "threat_family": _s(r.get("threat_family"))})
    return out


def disjoint(train_keys: Sequence[str], test_pids: Set[str], mode: str
             ) -> Tuple[List[str], int]:
    """mode=all: keep every training paper (the corpus your other rows trained on).
    mode=disjoint: drop training papers whose RAW id is a test paper (the ARGAR/BTR train
    halves sit on test papers; a reviewer retrained on them has seen the test manuscripts)."""
    if mode == "all":
        return list(train_keys), 0
    keep = [k for k in train_keys if raw_pid(k) not in test_pids]
    return keep, len(train_keys) - len(keep)


def sample_keys(keys: Sequence[str], n: int, *seed_parts) -> List[str]:
    h = hashlib.sha1("|".join(str(p) for p in seed_parts).encode()).hexdigest()
    ks = sorted(keys)
    random.Random(int(h[:12], 16)).shuffle(ks)
    return ks[:n] if n > 0 else ks


def stratified(items: List[Dict], n: int, key: str, seed: int) -> List[Dict]:
    """Up to n items, round-robin over `key` values so a cap keeps every source."""
    if n <= 0 or len(items) <= n:
        return list(items)
    rng = random.Random(seed)
    by: Dict[str, List[Dict]] = {}
    for it in items:
        by.setdefault(_s(it.get(key)), []).append(it)
    for v in by.values():
        rng.shuffle(v)
    out: List[Dict] = []
    keys = sorted(by)
    while len(out) < n and any(by.values()):
        for k in keys:
            if by[k] and len(out) < n:
                out.append(by[k].pop())
    return out


# ======================================================================================
# TEST
# ======================================================================================
def load_test(records: str, index: str) -> pd.DataFrame:
    """One row per document, in eval_index order, with the index columns joined."""
    for p in (records, index):
        if not os.path.exists(p):
            raise SystemExit(f"[FATAL] {p} not found. Run the inputs stage first "
                             f"(qsub jobs/job_sr_i_inputs.pbs), with the same RUN_TAG.")
    recs = [json.loads(l) for l in open(records, encoding="utf-8") if l.strip()]
    df = pd.DataFrame(recs)
    df.insert(0, "eval_index", range(len(df)))
    ix = pd.read_csv(index)
    if len(ix) != len(df):
        raise SystemExit(f"[FATAL] {records} has {len(df)} documents but {index} has "
                         f"{len(ix)} rows -- they were built by different stage-I runs")
    drop = [c for c in ("doc_id", "label", "condition") if c in ix.columns]
    d = df.merge(ix.drop(columns=drop), on="eval_index", how="left")
    if not (d["doc_id"].astype(str).values == ix["doc_id"].astype(str).values).all():
        raise SystemExit(f"[FATAL] {records} and {index} disagree on doc_id order")
    d["label"] = d["label"].astype(int)
    d["paper_id"] = d["paper_id"].astype(str)
    for c in ("pair_uid", "input_source", "threat_family", "data_origin"):
        if c not in d.columns:
            d[c] = ""
        d[c] = d[c].fillna("").astype(str)
    for c in ("harmful", "seen_in_train"):
        if c not in d.columns:
            d[c] = 0
        d[c] = d[c].fillna(0).astype(int)
    return d


def test_paper_ids(*indexes_or_records: str) -> Set[str]:
    out: Set[str] = set()
    for p in indexes_or_records:
        if not p or not os.path.exists(p):
            continue
        if p.endswith(".jsonl"):
            for l in open(p, encoding="utf-8"):
                if l.strip():
                    out.add(raw_pid(json.loads(l).get("paper_id")))
    return out


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
