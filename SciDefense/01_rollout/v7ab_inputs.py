#!/usr/bin/env python3
"""
v7ab_inputs.py — feed v7 the SAME data as llim_agents_v8_dvb, in the formats v7 already reads.

Nothing in v7 changes. v7 reads two things, and this file writes both from the v8 data
files ($AD_ROOT/data/final_data/v8_defense_data/), so v7 and v8 are scored on identical
documents:

    TRAIN   inputs/train_documents.csv
            V4's format (ft_attack.py): one row per (paper_id, condition), clean rows with
            label 0 and attacked rows with label 1, gold in `insert`. V4's
            ft_trajectories.py samples rollouts from it and v7_score_rollouts.py scores
            them -- exactly as with V4's own traindata_shared/train_documents.csv.
            Rows: secondary_280 + tertiary_200 (attacked with the benchmark's code) and the
            ARGAR / BTR TRAIN halves.

    TEST    inputs/<model>/eval_records.jsonl
            V5's DocRecord format (prepare_comparison.py): consecutive clean/attacked pairs.
            V5's runner_def.py --input-mode sft-dpo --eval-records reads it, as in job_01.
            The insertion pairs are COPIED from v8's prepared_insertion__<tag>/
            eval_records.jsonl (the published 990 documents, byte-identical); the ARGAR and
            BTR test pairs are appended after them.
            inputs/<model>/eval_index.csv   eval_index -> pair_uid, input_source,
                                            data_origin, threat_family, seen_in_train
            (runner_def writes eval_index into stage1_detail.csv; v7ab_report.py joins on it
            to report every metric in total AND per source.)

    python v7ab_inputs.py --models gemma3-27b-it qwen3-8b mistral-small-3.1-24b gpt-4o-mini
    python v7ab_inputs.py --models gemma3-27b-it --max-rows 5 --run-tag pilot5   # a subset

--max-rows N picks N pairs per source in the same stable order as v8_eval.py --max-rows,
so a v7 pilot and a v8 pilot score the same pairs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from typing import Dict, List, Optional

import pandas as pd

AD_ROOT = os.environ.get(
    "AD_ROOT", ".")
HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get("RL_WORK", HERE)
V8_DATA = os.environ.get("V8_DATA_DIR",
                         os.path.join(AD_ROOT, "data/final_data/v8_defense_data"))

# reviewer key -> the tag v8 uses in its file names
V8_TAG = {"gemma3-27b-it": "gemma3_27b", "qwen3-8b": "qwen3_8b",
          "mistral-small-3.1-24b": "mistral_small_3_1_24b", "gpt-4o-mini": "gpt_4o_mini"}


DOCREC_FIELDS = ("doc_id", "paper_id", "row_index", "label", "condition", "group", "document",
                 "insert", "base_rating", "attacked_rating", "d_rating", "asr", "slot",
                 "max_chars")


def _s(x) -> str:
    return "" if x is None or (isinstance(x, float) and x != x) else str(x)


def _f(x) -> Optional[float]:
    try:
        v = float(x)
        return None if v != v else v
    except (TypeError, ValueError):
        return None


def gold_text(r: pd.Series) -> str:
    """The attacked text the gold span covers (longest span) -- V4/V5 call this `insert`."""
    ins = _s(r.get("insert")).strip()
    if ins and ins.lower() != "nan":
        return ins
    doc = _s(r.get("attacked_document"))
    try:
        spans = json.loads(_s(r.get("gold_offsets")) or "[]")
    except Exception:
        spans = []
    best = ""
    for a, b in spans:
        t = doc[int(a):int(b)]
        if len(t.strip()) > len(best):
            best = t.strip()
    return best


# ======================================================================================
# TRAIN -> V4 train_documents.csv
# ======================================================================================
def build_train(train_pairs: str, out_csv: str, cap: int, seed: int) -> pd.DataFrame:
    tp = pd.read_csv(train_pairs, low_memory=False)
    rows: List[Dict] = []
    seen_clean = set()
    used = set()
    for _, r in tp.iterrows():
        src = _s(r["input_source"])
        origin = _s(r.get("data_origin")) or src
        # ARGAR and BTR rows sit on the SAME papers and carry their own renderings of the
        # benign text, so their paper ids are namespaced: each keeps its own clean twin.
        pid = _s(r["paper_id"]) if src == "insertion" else f"{src}:{_s(r['paper_id'])}"
        cond = _s(r["condition"])
        if src != "insertion":
            v = _s(r.get("variant_label")) or _s(r.get("strategy"))
            cond = f"{'R' if src == 'argar' else 'P'}_{src}" + (f"|{v}" if v else "")
        base, k = cond, 1
        while (pid, cond) in used:                # keep (paper_id, condition) unique
            k += 1
            cond = f"{base}#{k}"
        used.add((pid, cond))
        harm = _f(r.get("historical_asr"))
        harm = int(harm > 0) if harm is not None else int(_f(r.get("harmful_prior")) or 0)
        weak = _s(r.get("reference_weaknesses"))[:8000]
        common = dict(source=origin, input_source=src, data_origin=origin,
                      attack_generator=_s(r.get("attack_generator")),
                      row_index=_s(r.get("row_index")), n_weaknesses=weak.count("\n") + 1,
                      weaknesses=weak, threat_family=_s(r.get("threat_family")),
                      pair_uid=_s(r.get("pair_uid")))
        if pid not in seen_clean:
            seen_clean.add(pid)
            rows.append(dict(paper_id=pid, label=0, condition="A0_clean", family="A0_clean",
                             strategy="clean", tier="T0", harmful=0, insert="",
                             document=_s(r["benign_document"]), **common))
        rows.append(dict(paper_id=pid, label=1, condition=cond, family=cond,
                         strategy=_s(r.get("strategy")) or cond, tier="", harmful=harm,
                         insert=gold_text(r), document=_s(r["attacked_document"]), **common))
    df = pd.DataFrame(rows)
    # Interleave papers in a fixed random order, so `--max-docs` in V4's rollout (which
    # takes the FIRST n rows) gets a mix of sources rather than the first file only.
    papers = sorted(df["paper_id"].unique())
    random.Random(seed).shuffle(papers)
    rank = {p: i for i, p in enumerate(papers)}
    df = df.assign(_r=df["paper_id"].map(rank)).sort_values(["_r", "label"]).drop(columns="_r")
    if cap > 0:
        # keep every ARGAR/BTR paper (they are small); insertion fills the rest
        small = df[df["input_source"] != "insertion"]
        need = max(0, cap - len(small))
        ins_p = [p for p in papers if not str(p).startswith(("argar:", "btr:"))]
        keep, n = set(), 0
        for p in ins_p:
            m = int((df["paper_id"] == p).sum())
            if n + m > need:
                break
            keep.add(p)
            n += m
        df = df[(df["input_source"] != "insertion") | df["paper_id"].isin(keep)]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    tmp = out_csv + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, out_csv)
    return df


# ======================================================================================
# TEST -> V5 eval_records.jsonl + eval_index.csv
# ======================================================================================
def select(te: pd.DataFrame, max_rows: int, sources: List[str]) -> pd.DataFrame:
    """Same rule as v8_eval.select: per source, stable sha1(pair_uid) order."""
    d = te.copy()
    if sources:
        d = d[d["input_source"].isin(sources)]
    if max_rows > 0:
        d["_k"] = [hashlib.sha1(str(u).encode()).hexdigest() for u in d["pair_uid"]]
        d = (d.sort_values(["input_source", "_k"]).groupby("input_source", group_keys=False)
             .head(max_rows).drop(columns="_k"))
    return d


def record(doc_id, pid, row_index, label, cond, group, document, insert, r) -> Dict:
    base, att = _f(r.get("historical_base_rating")), _f(r.get("historical_attacked_rating"))
    asr = _f(r.get("historical_asr"))
    return {"doc_id": doc_id, "paper_id": pid, "row_index": int(row_index), "label": label,
            "condition": cond, "group": group, "document": document,
            "insert": insert if label else "",
            "base_rating": base, "attacked_rating": att if label else None,
            "d_rating": (att - base) if (label and base is not None and att is not None) else None,
            "asr": (int(asr > 0) if asr is not None else None) if label else None,
            "slot": _s(r.get("slot")) or "limitations_end",
            "max_chars": int(_f(r.get("render_max_chars")) or 30000)}


def build_test(model: str, out_dir: str, max_rows: int, sources: List[str], tag: str,
               train_papers: set) -> Dict:
    t = V8_TAG[model]
    test_pairs = os.path.join(V8_DATA, f"test_pairs__{t}.csv.gz")
    prepared = os.path.join(V8_DATA, f"prepared_insertion__{t}", "eval_records.jsonl")
    for p in (test_pairs, prepared):
        if not os.path.exists(p):
            raise SystemExit(f"[FATAL] {p} not found. Build the v8 test data first:\n"
                             f"    cd $AD_ROOT/defense/llim_agents_v8_dvb && "
                             f"qsub -v STAGES=data_test,inputs jobs/job_v8_<model>.pbs")
    te = pd.read_csv(test_pairs, low_memory=False)
    ins_recs = [json.loads(l) for l in open(prepared, encoding="utf-8") if l.strip()]
    # V5's DocRecord fields; a record written by prepare_comparison already has all of them,
    # so this only fills gaps (it never changes a value that is present)
    for k, r in enumerate(ins_recs):
        r.setdefault("row_index", k // 2)
        r.setdefault("group", "")
        for f, dflt in (("insert", ""), ("base_rating", None), ("attacked_rating", None),
                        ("d_rating", None), ("asr", None), ("slot", "limitations_end"),
                        ("max_chars", 30000)):
            r.setdefault(f, dflt)
        for extra in [x for x in r if x not in DOCREC_FIELDS]:
            r.pop(extra)
    ins_te = te[te["input_source"] == "insertion"].reset_index(drop=True)
    if len(ins_recs) != 2 * len(ins_te):
        raise SystemExit(f"[FATAL] {prepared}: {len(ins_recs)} records vs {len(ins_te)} "
                         f"insertion pairs in {test_pairs} -- rebuild the v8 test data")
    # insertion pair i of test_pairs IS records 2i, 2i+1 (v8_inputs.load_insertion_test
    # builds the pairs in record order); checked, not assumed:
    for i in (0, len(ins_te) - 1):
        a = ins_recs[2 * i + 1]
        if _s(a["paper_id"]).strip() != _s(ins_te.loc[i, "paper_id"]).strip() or \
                _s(a["condition"]) != _s(ins_te.loc[i, "condition"]):
            raise SystemExit(f"[FATAL] record/pair order mismatch at insertion pair {i}")
    uid2i = {u: i for i, u in enumerate(ins_te["pair_uid"])}

    sel = select(te, max_rows, sources)
    recs: List[Dict] = []
    idx: List[Dict] = []
    next_row = 1 + max(int(r.get("row_index") or 0) for r in ins_recs)

    def add(rec_pair, r):
        for rec in rec_pair:
            idx.append({"eval_index": len(recs), "doc_id": rec["doc_id"],
                        "pair_uid": r["pair_uid"], "input_source": r["input_source"],
                        "data_origin": _s(r.get("data_origin")),
                        "threat_family": _s(r.get("threat_family")),
                        "attack_generator": _s(r.get("attack_generator")),
                        "label": rec["label"], "condition": rec["condition"],
                        "harmful": int(bool(rec.get("asr"))) if rec["label"] else 0,
                        "seen_in_train": int(_s(r["paper_id"]) in train_papers)})
            recs.append(rec)

    for src in ("insertion", "argar", "btr"):
        for _, r in sel[sel["input_source"] == src].iterrows():
            if src == "insertion":
                i = uid2i[r["pair_uid"]]
                add((ins_recs[2 * i], ins_recs[2 * i + 1]), r)
                continue
            pid = _s(r["paper_id"])
            cond = f"{'R' if src == 'argar' else 'P'}_{src}"
            v = _s(r.get("variant_label"))
            if v:
                cond += f"|{v}"
            uid = _s(r["pair_uid"])
            group = "repackaging" if src == "argar" else "perturbation"
            add((record(f"{uid}|clean", pid, next_row, 0, "A0_clean", group,
                        _s(r["benign_document"]), "", r),
                 record(f"{uid}|attacked", pid, next_row, 1, cond, group,
                        _s(r["attacked_document"]), gold_text(r), r)), r)
            next_row += 1

    os.makedirs(out_dir, exist_ok=True)
    sfx = f"__{tag}" if tag else ""
    rp = os.path.join(out_dir, f"eval_records{sfx}.jsonl")
    ip = os.path.join(out_dir, f"eval_index{sfx}.csv")
    with open(rp + ".tmp", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(rp + ".tmp", rp)
    pd.DataFrame(idx).to_csv(ip, index=False)
    ix = pd.DataFrame(idx)
    counts = ix.groupby(["input_source", "label"]).size().unstack(fill_value=0)
    return {"records": rp, "index": ip, "n_docs": len(recs),
            "counts": counts.to_dict(), "seen_in_train_docs": int(ix["seen_in_train"].sum())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(V8_TAG))
    ap.add_argument("--max-rows", dest="max_rows", type=int, default=0,
                    help="test pairs PER SOURCE (0 = all)")
    ap.add_argument("--sources", nargs="*", default=[])
    ap.add_argument("--run-tag", dest="run_tag", default="",
                    help="write eval_records__<tag>.jsonl (a pilot never replaces the full set)")
    ap.add_argument("--train-cap", dest="train_cap", type=int, default=0,
                    help="max training DOCUMENTS (ARGAR/BTR kept whole; 0 = all)")
    ap.add_argument("--no-train", dest="do_train", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    man = {"v8_data": V8_DATA, "work": WORK, "models": {}}
    train_pairs = os.path.join(V8_DATA, "train_pairs.csv.gz")
    train_papers: set = set()
    if os.path.exists(train_pairs):
        train_papers = set(pd.read_csv(train_pairs, usecols=["paper_id"])["paper_id"].astype(str))
    if a.do_train:
        if not os.path.exists(train_pairs):
            print(f"[train] {train_pairs} not found -- build it with the v8 data stage "
                  f"(qsub -v STAGES=data_train jobs/job_v8_gemma.pbs). Test inputs only.")
        else:
            out = os.path.join(WORK, "inputs", "train_documents.csv")
            df = build_train(train_pairs, out, a.train_cap, a.seed)
            c = df.groupby(["input_source", "label"]).size().unstack(fill_value=0)
            print(f"[train] {out}: {len(df)} documents, {df['paper_id'].nunique()} papers")
            print(c.to_string())
            man["train"] = {"file": out, "n_docs": len(df), "cap": a.train_cap,
                            "counts": c.to_dict()}
    for m in a.models:
        if m not in V8_TAG:
            raise SystemExit(f"[FATAL] unknown model {m}; one of {list(V8_TAG)}")
        try:
            r = build_test(m, os.path.join(WORK, "inputs", m), a.max_rows, a.sources,
                           a.run_tag, train_papers)
        except SystemExit as e:                  # one model's v8 data not built yet
            print(f"[warn] {m}: skipped -- {e}")
            continue
        man["models"][m] = r
        print(f"[test]  {m}: {r['n_docs']} documents -> {r['records']}")
        print(f"        per source/label {r['counts']}   "
              f"({r['seen_in_train_docs']} docs on papers that also have training rows)")
    if not man["models"]:
        raise SystemExit("[FATAL] no model had v8 test data; build it with the v8 data stage")
    sfx = f"__{a.run_tag}" if a.run_tag else ""
    os.makedirs(os.path.join(WORK, "inputs"), exist_ok=True)
    with open(os.path.join(WORK, "inputs", f"manifest{sfx}.json"), "w") as f:
        json.dump(man, f, indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
