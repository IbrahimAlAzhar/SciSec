#!/usr/bin/env python3
"""
v8_inputs.py — build ONE paired corpus from three sources, tagged by origin.

    insertion   the V4/V5 corpus (what sft_dpo_code.zip evaluated): TEST pairs from V5's
                prepare_comparison.py eval_records.jsonl -- the same 990 documents as your
                published table -- and TRAIN pairs from V4's train_documents.csv
    argar       llm_agents_DVB/input_baselines/argar__<model>.csv.gz   (repackaging)
    btr         llm_agents_DVB/input_baselines/btr__<model>.csv.gz     (perturbation)

Every row is a PAIR: the benign and attacked versions of one paper. Output:

    inputs/<model>/pairs_all.csv.gz        every pair, `input_source` + `split` columns
    inputs/<model>/pairs_<source>.csv.gz   one file per source
    inputs/<model>/manifest.json           counts per source x split x threat family,
                                           file hashes, and every warning raised here

THE SPLIT, AND WHY IT IS DONE ON PAPERS
    A paper's benign text is the same in every attacked variant, so a paper on both sides
    lets the model memorise its authors' prose. TEST papers are the insertion TEST papers
    (the 500 set). --split-mode:
      transfer  (default) ARGAR/BTR rows on TEST papers are test-only; rows on other papers
                are training data. If ARGAR/BTR cover only TEST papers, they measure how
                far detection TRANSFERS to threats never trained on -- a result worth
                reporting as such.
      holdout   a hashed fraction (--holdout-frac) of TEST papers moves to TRAIN for ALL
                sources, and those papers leave every test set, insertion included. The
                insertion N then shrinks below 990, and the manifest says by how much.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import os
import sys
from typing import Dict, List, Optional, Sequence

import pandas as pd

import ft_config as C
import v8_tools as T

DVB_COLS = ["pair_id", "dataset", "paper_id", "row_index", "condition", "source_model",
            "attack_kind", "variant_label", "benign_document", "attacked_document",
            "insert", "gold_spans", "gold_span_kind", "reference_weaknesses",
            "reference_source", "source_file", "source_column", "render_method",
            "render_max_chars", "slot", "historical_base_rating",
            "historical_attacked_rating", "historical_asr"]


def sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()[:16]


def _f(x) -> Optional[float]:
    try:
        v = float(x)
        return None if v != v else v
    except (TypeError, ValueError):
        return None


def _hash_frac(pid: str, seed: int) -> float:
    return int(hashlib.sha1(f"{seed}|{pid}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


# ======================================================================================
# insertion (V4/V5)
# ======================================================================================
def load_insertion_test(records_path: str, benign_csv: str = "") -> pd.DataFrame:
    """V5 eval_records.jsonl: consecutive (clean, attacked) DocRecords of one paper."""
    recs = [json.loads(l) for l in open(records_path, encoding="utf-8") if l.strip()]
    if len(recs) % 2:
        raise SystemExit(f"[FATAL] {records_path}: odd record count; not a paired corpus")
    weak = {}
    if benign_csv and os.path.exists(benign_csv):
        b = pd.read_csv(benign_csv, low_memory=False)
        idc = next((c for c in ("id", "paper_id") if c in b.columns), None)
        wc = C.COLUMNS["weaknesses"]
        if idc and wc in b.columns:
            weak = {str(k).strip(): str(v) for k, v in zip(b[idc], b[wc])}
    rows = []
    for i in range(0, len(recs), 2):
        c, a = recs[i], recs[i + 1]
        if int(c["label"]) != 0 or int(a["label"]) != 1 or str(c["paper_id"]) != str(a["paper_id"]):
            raise SystemExit(f"[FATAL] {records_path}: record {i} is not a clean/attacked pair")
        pid = str(a["paper_id"]).strip()
        rows.append({
            "pair_id": a.get("doc_id") or f"{pid}|{a['condition']}", "dataset": "primary_500",
            "paper_id": pid, "row_index": a.get("row_index", i // 2),
            "condition": a["condition"], "source_model": "", "attack_kind": "insertion",
            "variant_label": a.get("group", ""), "benign_document": c["document"],
            "attacked_document": a["document"], "insert": a.get("insert", ""),
            "gold_spans": "", "gold_span_kind": "insert",
            "reference_weaknesses": weak.get(pid, ""), "reference_source": "benign_csv",
            "source_file": os.path.basename(records_path), "source_column": "document",
            "render_method": "align.render", "render_max_chars": a.get("max_chars", ""),
            "slot": a.get("slot", ""), "historical_base_rating": a.get("base_rating"),
            "historical_attacked_rating": a.get("attacked_rating"),
            "historical_asr": a.get("asr")})
    return pd.DataFrame(rows)


def load_insertion_train(path: str) -> pd.DataFrame:
    """V4 train_documents.csv: one row per (paper, condition); clean rows pair with each."""
    d = pd.read_csv(path, low_memory=False)
    need = {"paper_id", "label", "condition", "document"}
    if not need <= set(d.columns):
        raise SystemExit(f"[FATAL] {path} lacks {sorted(need - set(d.columns))}")
    d["paper_id"] = d["paper_id"].astype(str).str.strip()
    clean = d[d["label"] == 0].drop_duplicates("paper_id").set_index("paper_id")
    rows = []
    for _, r in d[d["label"] == 1].iterrows():
        pid = r["paper_id"]
        if pid not in clean.index:
            continue
        cb = clean.loc[pid]
        rows.append({
            "pair_id": f"{pid}|{r['condition']}|{r.get('strategy', '')}",
            "dataset": str(r.get("source", "train")), "paper_id": pid,
            "row_index": r.get("row_index", ""), "condition": r["condition"],
            "source_model": "", "attack_kind": "insertion",
            "variant_label": str(r.get("strategy", "")),
            "benign_document": cb["document"], "attacked_document": r["document"],
            "insert": r.get("insert", ""), "gold_spans": "", "gold_span_kind": "insert",
            "reference_weaknesses": r.get("weaknesses", ""),
            "reference_source": "train_documents", "source_file": os.path.basename(path),
            "source_column": "document", "render_method": "align.render",
            "render_max_chars": "", "slot": "", "historical_base_rating": None,
            "historical_attacked_rating": None, "historical_asr": r.get("harmful")})
    return pd.DataFrame(rows)


# ======================================================================================
# DVB files
# ======================================================================================
def find_dvb(kind: str, model_key: str, explicit: str = "") -> Optional[str]:
    if explicit:
        return explicit if os.path.exists(explicit) else None
    tag = C.MODELS[model_key].tag
    exact = os.path.join(C.DVB_DIR, f"{kind}__{tag}.csv.gz")
    if os.path.exists(exact):
        return exact
    cands = sorted(glob.glob(os.path.join(C.DVB_DIR, f"{kind}__*.csv*")))
    head = tag.split("_")[0]
    for c in cands:
        if head in os.path.basename(c):
            return c
    if cands:
        print(f"[inputs] WARNING: no {kind} file for {model_key} ({tag}); using "
              f"{os.path.basename(cands[0])}. Its historical ratings come from a DIFFERENT "
              f"reviewer -- v8 re-reviews every stage itself, so the table is unaffected, "
              f"but historical_* columns are not this model's.")
        return cands[0]
    return None


def load_dvb(path: str) -> pd.DataFrame:
    d = pd.read_csv(path, low_memory=False, compression="infer")
    miss = [c for c in ("paper_id", "benign_document", "attacked_document") if c not in d.columns]
    if miss:
        raise SystemExit(f"[FATAL] {path} lacks {miss}. Columns: {list(d.columns)[:30]}")
    for c in DVB_COLS:
        if c not in d.columns:
            d[c] = None
    d["paper_id"] = d["paper_id"].astype(str).str.strip()
    return d


# ======================================================================================
# normalisation to the one schema
# ======================================================================================
def finalize(df: pd.DataFrame, source: str, origin: str = "") -> pd.DataFrame:
    out = df.copy()
    out["input_source"] = source
    fam, gold, gfrom, harm = [], [], [], []
    for _, r in out.iterrows():
        att = str(r.get("attacked_document") or "")
        ben = str(r.get("benign_document") or "")
        ins = str(r.get("insert") or "")
        if ins.lower() == "nan":
            ins = ""
        g = T.parse_gold_spans(r.get("gold_spans"), att, "")
        src = "column" if g else ""
        if not g and ins:
            g = T.parse_gold_spans("", att, ins)
            src = "insert" if g else ""
        if not g and ben and att and ben != att:
            g = T.diff_spans(ben, att)
            src = "diff" if g else ""
        gold.append(json.dumps([[a, b] for a, b in g]))
        gfrom.append(src or "none")
        fam.append(C.threat_family(source, str(r.get("condition") or ""),
                                   str(r.get("attack_kind") or "")))
        a = _f(r.get("historical_asr"))
        harm.append(int(a > 0) if a is not None else None)
    out["gold_offsets"] = gold
    out["gold_from"] = gfrom
    out["threat_family"] = fam
    out["harmful_hist"] = harm
    pre = origin or source
    out["pair_uid"] = [f"{pre}:{p}:{c}:{i}" for i, (p, c) in
                       enumerate(zip(out["paper_id"], out["condition"].astype(str)))]
    return out


def _atomic_csv(df: pd.DataFrame, path: str) -> None:
    """Write then rename: a job reading this file never sees it half-written."""
    tmp = f"{path}.tmp{os.getpid()}"
    df.to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, path)


def assemble(train_path: str, test_path: str, out_dir: str, model: str) -> int:
    """
    The normal path: the reusable files from v8_build_data.py -> this model's work copy.
    TEST rows on a paper that also has TRAIN rows (the ARGAR/BTR train half is on
    primary_500 papers) get seen_in_train=1; the report shows results with and without
    them. The SAME attacked document on both sides is real leakage and stops the job.
    """
    if not os.path.exists(test_path):
        raise SystemExit(f"[FATAL] {test_path} not found -- run v8_build_data.py test first")
    te = pd.read_csv(test_path, low_memory=False)
    te["split"] = "test"
    frames, files = [te], {"test": {"path": test_path, "sha": sha(test_path), "rows": len(te)}}
    tr_p: set = set()
    if train_path and os.path.exists(train_path):
        tr = pd.read_csv(train_path, low_memory=False)
        tr["split"] = "train"
        tr_p = set(tr["paper_id"].astype(str))
        both = set(tr["attacked_document"].astype(str)) & set(te["attacked_document"].astype(str))
        if both:
            raise SystemExit(f"[FATAL] {len(both)} attacked documents are in BOTH train and test")
        frames.append(tr)
        files["train"] = {"path": train_path, "sha": sha(train_path), "rows": len(tr)}
    elif train_path:
        print(f"[inputs] no train file at {train_path}: test only (fine for zero-shot)")
    te["paper_id"] = te["paper_id"].astype(str)
    te["seen_in_train"] = te["paper_id"].isin(tr_p).astype(int)
    allp = pd.concat(frames, ignore_index=True)
    allp["seen_in_train"] = allp["seen_in_train"].fillna(0).astype(int)
    if allp["pair_uid"].duplicated().any():
        raise SystemExit("[FATAL] duplicate pair_uid across train and test")
    os.makedirs(out_dir, exist_ok=True)
    _atomic_csv(allp, os.path.join(out_dir, "pairs_all.csv.gz"))
    for src, g in allp.groupby("input_source"):
        _atomic_csv(g, os.path.join(out_dir, f"pairs_{src}.csv.gz"))
    counts = (allp.groupby(["split", "input_source", "data_origin", "threat_family"]).size()
              .reset_index(name="pairs"))
    seen = te.groupby("input_source")["seen_in_train"].agg(["size", "sum"]).reset_index()
    man = {"model": model, "files": files, "n_pairs": len(allp),
           "counts": counts.to_dict(orient="records"),
           "test_rows_on_papers_seen_in_train": seen.to_dict(orient="records")}
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(man, f, indent=2, default=str)
    print(counts.to_string(index=False))
    print("\n[inputs] test rows whose PAPER also has training rows (other attack type):")
    print(seen.rename(columns={"size": "test_pairs", "sum": "seen_in_train"}).to_string(index=False))
    print(f"\n[saved] {out_dir}/pairs_all.csv.gz ({len(allp)} pairs)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="build the concatenated v8 input corpus")
    ap.add_argument("--model", required=True, choices=sorted(C.MODELS))
    ap.add_argument("--insertion-records", dest="ins_test", default="",
                    help="V5 prepare_comparison eval_records.jsonl (insertion TEST pairs)")
    ap.add_argument("--insertion-train", dest="ins_train", default="",
                    help="V4 train_documents.csv (insertion TRAIN pairs)")
    ap.add_argument("--benign-csv", dest="benign_csv", default=C.TEST_CSV)
    ap.add_argument("--argar-csv", dest="argar", default="")
    ap.add_argument("--btr-csv", dest="btr", default="")
    ap.add_argument("--extra-csv", dest="extra", nargs="*", default=[],
                    help="more DVB-format files; source name = filename before '__'")
    ap.add_argument("--split-mode", dest="split_mode", choices=["transfer", "holdout"],
                    default="transfer")
    ap.add_argument("--holdout-frac", dest="holdout_frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", dest="out_dir", default="")
    ap.add_argument("--train-pairs", dest="train_pairs", default="",
                    help="v8_build_data.py train file (the normal path)")
    ap.add_argument("--test-pairs", dest="test_pairs", default="",
                    help="v8_build_data.py test file for this model (the normal path)")
    a = ap.parse_args()
    if a.test_pairs:
        return assemble(a.train_pairs, a.test_pairs,
                        a.out_dir or C.wpath(C.D_INPUTS, a.model.replace("/", "_")), a.model)

    out_dir = a.out_dir or C.wpath(C.D_INPUTS, a.model.replace("/", "_"))
    os.makedirs(out_dir, exist_ok=True)
    frames: List[pd.DataFrame] = []
    files: Dict[str, Dict] = {}
    warnings: List[str] = []

    test_papers: set = set()
    if a.ins_test:
        if not os.path.exists(a.ins_test):
            raise SystemExit(f"[FATAL] --insertion-records not found: {a.ins_test}")
        t = finalize(load_insertion_test(a.ins_test, a.benign_csv), "insertion")
        t["split"] = "test"
        test_papers |= set(t["paper_id"])
        frames.append(t)
        files["insertion_test"] = {"path": a.ins_test, "sha": sha(a.ins_test), "rows": len(t)}
        print(f"[inputs] insertion TEST : {len(t)} pairs, {t['paper_id'].nunique()} papers")
    else:
        warnings.append("no insertion TEST records: the table will have no insertion rows")
    if os.path.exists(C.TEST_CSV):
        b = pd.read_csv(C.TEST_CSV, low_memory=False, usecols=lambda c: c in ("id", "paper_id"))
        test_papers |= set(b.iloc[:, 0].astype(str).str.strip())

    if a.ins_train:
        if os.path.exists(a.ins_train):
            t = finalize(load_insertion_train(a.ins_train), "insertion")
            leak = set(t["paper_id"]) & test_papers
            if leak:
                warnings.append(f"{len(leak)} insertion TRAIN papers are TEST papers; dropped")
                t = t[~t["paper_id"].isin(test_papers)]
            t["split"] = "train"
            frames.append(t)
            files["insertion_train"] = {"path": a.ins_train, "sha": sha(a.ins_train),
                                        "rows": len(t)}
            print(f"[inputs] insertion TRAIN: {len(t)} pairs, {t['paper_id'].nunique()} papers")
            if t["paper_id"].nunique() < 50:
                warnings.append(f"insertion TRAIN has only {t['paper_id'].nunique()} papers: "
                                "a smoke-test corpus, not a training corpus. Rebuild it with "
                                "V4 ft_attack.py over the secondary+tertiary TRAIN papers.")
        else:
            warnings.append(f"--insertion-train not found: {a.ins_train}")

    dvb = [("argar", find_dvb("argar", a.model, a.argar)),
           ("btr", find_dvb("btr", a.model, a.btr))]
    for p in a.extra:
        dvb.append((os.path.basename(p).split("__")[0], p))
    for src, path in dvb:
        if not path:
            warnings.append(f"no {src} file found under {C.DVB_DIR}")
            continue
        d = finalize(load_dvb(path), src)
        d["split"] = ["test" if p in test_papers else "train" for p in d["paper_id"]]
        frames.append(d)
        files[src] = {"path": path, "sha": sha(path), "rows": len(d)}
        n_te = int((d["split"] == "test").sum())
        print(f"[inputs] {src:<10}: {len(d)} pairs ({n_te} on TEST papers, "
              f"{len(d) - n_te} on other papers), gold from "
              f"{dict(d['gold_from'].value_counts())}")
        if n_te == len(d) and a.split_mode == "transfer":
            warnings.append(f"{src}: every pair is on a TEST paper, so {src} is TEST-ONLY: "
                            f"trained rows are evaluated on a threat they never saw "
                            f"(transfer). Use --split-mode holdout to train on some of it.")

    if not frames:
        raise SystemExit("[FATAL] no input source produced any pair")
    allp = pd.concat(frames, ignore_index=True)

    removed = 0
    if a.split_mode == "holdout":
        pool = sorted(set(allp.loc[allp["split"] == "test", "paper_id"]))
        moved = {p for p in pool if _hash_frac(p, a.seed) < a.holdout_frac}
        m = allp["paper_id"].isin(moved)
        removed = int((m & (allp["split"] == "test")).sum())
        allp.loc[m, "split"] = "train"
        warnings.append(f"holdout: {len(moved)} TEST papers moved to TRAIN for every source; "
                        f"{removed} test pairs removed (insertion N below the published 990)")

    # the contamination guard: no paper on both sides
    tr = set(allp.loc[allp["split"] == "train", "paper_id"])
    te = set(allp.loc[allp["split"] == "test", "paper_id"])
    if tr & te:
        raise SystemExit(f"[FATAL] {len(tr & te)} papers on BOTH sides: {sorted(tr & te)[:8]}")

    allp.to_csv(os.path.join(out_dir, "pairs_all.csv.gz"), index=False, compression="gzip")
    for src, g in allp.groupby("input_source"):
        g.to_csv(os.path.join(out_dir, f"pairs_{src}.csv.gz"), index=False,
                 compression="gzip")
    counts = (allp.groupby(["input_source", "split", "threat_family"]).size()
              .reset_index(name="pairs"))
    man = {"model": a.model, "split_mode": a.split_mode, "holdout_frac": a.holdout_frac,
           "seed": a.seed, "n_pairs": len(allp), "n_test_removed_by_holdout": removed,
           "files": files, "warnings": warnings,
           "counts": counts.to_dict(orient="records"),
           "papers": {"train": len(tr), "test": len(te)}}
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(man, f, indent=2, default=str)
    print("\n" + counts.to_string(index=False))
    for w in warnings:
        print(f"[inputs] NOTE: {w}")
    print(f"\n[saved] {out_dir}/pairs_all.csv.gz ({len(allp)} pairs), per-source files, "
          f"manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())