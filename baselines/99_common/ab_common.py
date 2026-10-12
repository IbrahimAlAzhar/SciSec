#!/usr/bin/env python3
"""
ab_common.py — shared core for running the defense baselines on ARGAR and BTR.

    $AD_ROOT/defense_baselines/baselines_for_argar_btr/

WHAT IS EVALUATED
=======================================================================================
The ARGAR (repackaging) and BTR (perturbation) TEST pairs that the v8 agents were scored
on, for the gemma3-27b reviewer:

    $AD_ROOT/data/final_data/v8_defense_data/test_pairs__gemma3_27b.csv.gz
        input_source == "argar"   ->  55 pairs  (110 documents)
        input_source == "btr"     -> 240 pairs  (480 documents)

One pair = the benign manuscript + the attacked version of the same paper. Both are
scored, so every family has its own benign twins and FPR is measured on the SAME papers
that were attacked. This is exactly how v8_report.py builds its Panel A/B rows, so a
baseline row here sits next to the v8 and v7 rows without any re-weighting.

If that file does not exist yet, the same test half is rebuilt from the raw files
($AD_ROOT/data/input_baselines/{argar,btr}__gemma3_27b.csv.gz) with v8's own paper-level
hash split (seed 0, 50/50), and the same pair_uid values, so the replay join in D7 and
the comparison with v8 still line up.  --subset full uses every ARGAR/BTR row instead.

THRESHOLDS (the part a reviewer will ask about)
=======================================================================================
Threshold-free numbers (AUROC, R@FPR5, R@FPR10) are computed on all documents.
Thresholded numbers (Recall, Prec, F1, FPR, HarmRec) need an operating point:

    --thresholds insertion   DEFAULT. The detector's operating point is FROZEN from its
                             run on the insertion attack set: read from
                             defense_baselines/output/<baseline>/predictions_all.csv and
                             re-derived exactly as that script did (best recall at
                             FPR <= 5% on its validation papers). The question answered is
                             "a deployed detector meets a new attack", which is the honest
                             one: nobody gets to re-tune a detector on an attack they have
                             not seen.
    --thresholds own         Re-tune on 20% of the ARGAR/BTR papers (hashed by paper,
                             same hash as common.stable_split) and report the other 80%.
                             Optimistic; use only as a sensitivity check.

    If the insertion predictions are not found the script falls back to `own` and says so
    in output.txt and in meta.json, so the table can never silently mix the two.

LLM
=======================================================================================
Only gemma3-27b-it, served by vLLM (see serve_gemma.sh). Used by D7 (LlamaFirewall
AlignmentCheck) and by the PaperGuard verifier. Every call is cached on disk by prompt
hash, so a crashed or re-submitted job does not pay twice.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ======================================================================================
# paths
# ======================================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
AD_ROOT = os.environ.get(
    "AD_ROOT", ".")


def _def_root() -> str:
    env = os.environ.get("DEF_ROOT", "")
    if env:
        return env
    parent = os.path.dirname(HERE)
    if os.path.exists(os.path.join(parent, "common.py")):
        return parent
    return os.path.join(AD_ROOT, "defense_baselines")


DEF_ROOT = _def_root()                                   # the original baselines
INSERTION_OUT = os.environ.get("INSERTION_OUT", os.path.join(DEF_ROOT, "output"))
OUT_ROOT = os.environ.get("AB_OUT", os.path.join(HERE, "output"))
V8_DATA = os.environ.get("V8_DATA_DIR", os.path.join(AD_ROOT, "data/final_data/v8_defense_data"))
DVB_DIR = os.environ.get("DVB_DIR", os.path.join(AD_ROOT, "data/input_baselines"))
V8_DIR = os.environ.get("V8_DIR", os.path.join(AD_ROOT, "defense/llim_agents_v8_dvb"))
V7AB_DIR = os.environ.get("V7AB_DIR", os.path.join(AD_ROOT, "defense/lim_agents_v7_RL_with_argar_btr"))
MODELS_DIR = os.environ.get("MODELS_DIR", "path/to/anonymous/models")
HF_HOME = os.environ.get("HF_HOME", "path/to/anonymous/hf_cache")

MODEL_KEY = "gemma3-27b-it"           # the reviewer the ARGAR/BTR attacks were run against
MODEL_TAG = "gemma3_27b"              # and the filename tag of its files
GEMMA_PATH = os.environ.get("GEMMA_PATH", os.path.join(MODELS_DIR, "gemma3_27b_it"))

SOURCES_DEFAULT = ("argar", "btr")
FAMILY_OF_SOURCE = {"argar": "repackaging", "btr": "perturbation"}
FAMILY_TITLE = {
    "repackaging": "Multi-round repackaging (ARGAR)",
    "perturbation": "Character/word perturbations (BTR)",
    "instruction": "Instruction-bearing attacks: A5--A9, A11, A12",
    "rhetorical": "Rhetorical suppression attacks: A1--A4, A10",
}


def parent_import(name: str):
    """Import a module from the ORIGINAL defense_baselines folder (common, d3_ppl_regex,
    d6_deberta, ...). Reusing their code is what makes a row here the same detector as the
    row in the insertion table, not a re-implementation that happens to share its name."""
    if DEF_ROOT not in sys.path:
        sys.path.insert(0, DEF_ROOT)
    import importlib
    return importlib.import_module(name)


def first_dir(*cands: str) -> str:
    for c in cands:
        if c and os.path.isdir(c):
            return c
    return cands[0] if cands else ""


# ======================================================================================
# data
# ======================================================================================
DVB_COLS = ["pair_id", "dataset", "paper_id", "row_index", "condition", "source_model",
            "attack_kind", "variant_label", "benign_document", "attacked_document",
            "insert", "gold_spans", "gold_span_kind", "reference_weaknesses",
            "historical_base_rating", "historical_attacked_rating", "historical_asr"]


def _f(x) -> Optional[float]:
    try:
        v = float(x)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


def test_pairs_path() -> str:
    return os.path.join(V8_DATA, f"test_pairs__{MODEL_TAG}.csv.gz")


def _raw_dvb(kind: str) -> pd.DataFrame:
    p = os.path.join(DVB_DIR, f"{kind}__{MODEL_TAG}.csv.gz")
    if not os.path.exists(p):
        alt = [q for q in (os.path.join(DVB_DIR, f"{kind}__{MODEL_TAG}.csv"),) if os.path.exists(q)]
        if not alt:
            raise SystemExit(f"[FATAL] {kind} file not found: {p}")
        p = alt[0]
    d = pd.read_csv(p, low_memory=False, compression="infer")
    for c in DVB_COLS:
        if c not in d.columns:
            d[c] = None
    d["paper_id"] = d["paper_id"].astype(str).str.strip()
    d["source_file"] = os.path.basename(p)
    return d


def _v8_assignment(frames: Dict[str, pd.DataFrame], train_frac: float = 0.5,
                   seed: int = 0) -> Dict[str, str]:
    """v8_build_data.dvb_assignment, verbatim: one hash order over the union of papers."""
    papers = set()
    for d in frames.values():
        papers |= set(d["paper_id"])
    order = sorted(papers, key=lambda x: hashlib.sha1(f"{seed}|{x}".encode()).hexdigest())
    n_tr = int(round(len(order) * train_frac))
    return {p: ("train" if i < n_tr else "test") for i, p in enumerate(order)}


def _pairs_from_raw(subset: str) -> pd.DataFrame:
    frames = {k: _raw_dvb(k) for k in ("argar", "btr")}
    assign = _v8_assignment(frames)
    out = []
    for kind, d in frames.items():
        if subset == "test":
            d = d[d["paper_id"].map(assign) == "test"].reset_index(drop=True)
            origin = f"{kind}_test"
        else:
            d = d.reset_index(drop=True)
            origin = f"{kind}_full"
        d = d.copy()
        d["input_source"] = kind
        d["threat_family"] = FAMILY_OF_SOURCE[kind]
        d["pair_uid"] = [f"{origin}:{p}:{c}:{i}" for i, (p, c) in
                         enumerate(zip(d["paper_id"], d["condition"].astype(str)))]
        d["harmful_hist"] = [(int(a > 0) if a is not None else None)
                             for a in (_f(x) for x in d["historical_asr"])]
        out.append(d)
    return pd.concat(out, ignore_index=True)


def load_pairs(sources: Sequence[str] = SOURCES_DEFAULT, subset: str = "test",
               max_rows: int = 0, pairs_csv: str = "", verbose: bool = True) -> pd.DataFrame:
    """One row per (benign, attacked) pair, in v8_eval.select() order, so --max-rows N
    scores the same N pairs per source as a v8 pilot with MAX_ROWS=N."""
    src_note = ""
    if pairs_csv:
        df = pd.read_csv(pairs_csv, low_memory=False)
        src_note = pairs_csv
    elif subset == "test" and os.path.exists(test_pairs_path()):
        df = pd.read_csv(test_pairs_path(), low_memory=False)
        src_note = test_pairs_path()
    else:
        if "insertion" in sources:
            raise SystemExit("[FATAL] insertion pairs only exist in the v8 test file "
                             f"({test_pairs_path()}); build it with v8_build_data.py test")
        df = _pairs_from_raw(subset)
        src_note = f"raw files in {DVB_DIR} (subset={subset}, v8 hash split)"
        if verbose and subset == "test":
            print(f"[data] {test_pairs_path()} not found -> rebuilt the same test half from "
                  f"the raw ARGAR/BTR files")
    if "split" in df.columns and subset == "test":
        df = df[df["split"].fillna("test") == "test"]
    df = df[df["input_source"].isin(list(sources))].copy()
    if df.empty:
        raise SystemExit(f"[FATAL] no pairs for sources {list(sources)} in {src_note}")
    df["paper_id"] = df["paper_id"].astype(str)
    if "threat_family" not in df.columns or df["threat_family"].isna().any():
        df["threat_family"] = df["input_source"].map(FAMILY_OF_SOURCE).fillna(
            df.get("threat_family"))
    df["_k"] = [hashlib.sha1(str(u).encode()).hexdigest() for u in df["pair_uid"]]
    df = df.sort_values(["input_source", "_k"])
    if max_rows > 0:
        df = df.groupby("input_source", group_keys=False).head(max_rows)
    df = df.drop(columns="_k").reset_index(drop=True)
    df.attrs["source_note"] = src_note
    if verbose:
        print(f"[data] pairs from {src_note}")
        for s, g in df.groupby("input_source"):
            print(f"[data]   {s:<9} {len(g):>4} pairs  {g['paper_id'].nunique():>3} papers  "
                  f"-> {2*len(g)} documents")
    return df


def method_of(r) -> str:
    for c in ("variant_label", "attack_kind", "condition"):
        v = r.get(c)
        if v is not None and str(v).strip() and str(v).lower() != "nan":
            return str(v).strip()
    return str(r.get("input_source"))


def stable_split(paper_id: str, val_frac: float = 0.20) -> str:
    """common.stable_split, same salt: a paper is val in every baseline or in none."""
    h = int(hashlib.sha1(f"defbase::{paper_id}".encode()).hexdigest()[:8], 16)
    return "val" if (h % 10_000) / 10_000.0 < val_frac else "test"


def docs_frame(pairs: pd.DataFrame, replay: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Pairs -> one row per DOCUMENT (benign twin label 0, attacked label 1).

    harmful follows v8_report.docs_frame exactly: the pair's historical ASR when the file
    carries it, otherwise rating(attacked) - rating(benign) >= 1.0 from the replayed
    gemma reviews (when available), otherwise 0."""
    rmap = {}
    if replay is not None and len(replay):
        for _, r in replay.iterrows():
            rmap[str(r["pair_uid"])] = r
    rows = []
    for _, r in pairs.iterrows():
        uid = str(r["pair_uid"])
        hh = _f(r.get("harmful_hist"))
        if hh is None:
            a = _f(r.get("historical_asr"))
            hh = None if a is None else float(a > 0)
        if hh is None and uid in rmap:
            dr = (_f(rmap[uid].get("attacked_rating")) or float("nan")) - \
                 (_f(rmap[uid].get("benign_rating")) or float("nan"))
            hh = float(dr >= 1.0) if dr == dr else None
        harmful = int(hh) if hh is not None else 0
        gold = r.get("gold_offsets")
        common = {"pair_uid": uid, "paper_id": str(r["paper_id"]),
                  "source": str(r["input_source"]), "family": str(r["threat_family"]),
                  "method": method_of(r), "condition": str(r.get("condition")),
                  "split": stable_split(str(r["paper_id"]))}
        rows.append(dict(common, doc_id=f"{uid}|B", label=0, harmful=0,
                         text=str(r.get("benign_document") or ""), gold_offsets="[]"))
        rows.append(dict(common, doc_id=f"{uid}|A", label=1, harmful=harmful,
                         text=str(r.get("attacked_document") or ""),
                         gold_offsets=str(gold) if isinstance(gold, str) else "[]"))
    d = pd.DataFrame(rows)
    d["n_chars"] = d["text"].str.len()
    same = int(((d.label == 1).values & (d["text"].values ==
                                         d["text"].shift(1).values)).sum())
    if same:
        print(f"[data] WARNING: {same} attacked document(s) are identical to their benign "
              f"twin. No detector can separate those.")
    return d


def unique_texts(docs: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
    """Benign twins repeat across a paper's variants: score each distinct text once."""
    keys = [hashlib.sha1(t.encode("utf-8", "ignore")).hexdigest() for t in docs["text"]]
    uniq: Dict[str, int] = {}
    texts: List[str] = []
    inv = np.empty(len(keys), int)
    for i, (k, t) in enumerate(zip(keys, docs["text"])):
        if k not in uniq:
            uniq[k] = len(texts)
            texts.append(t)
        inv[i] = uniq[k]
    return texts, inv


def load_replay(path: str = "") -> Optional[pd.DataFrame]:
    """The gemma reviewer's own reviews of every test pair, written by the v8 zero-shot
    run (benign_review / attacked_review in pairs_detail.csv.gz). D7 replays these."""
    cands = [path] if path else [
        os.path.join(V8_DIR, "runs", MODEL_KEY, ab, "pairs_detail.csv.gz")
        for ab in ("zeroshot", "sft", "sft_dpo", "dpo_only", "sft_dpo_merged", "grpo_prov")]
    for p in cands:
        if p and os.path.exists(p):
            d = pd.read_csv(p, low_memory=False)
            need = {"pair_uid", "benign_review", "attacked_review"}
            if need <= set(d.columns):
                d.attrs["path"] = p
                return d
            print(f"[replay] {p} lacks {sorted(need - set(d.columns))}; skipped")
    return None


# ======================================================================================
# metrics — identical to v8_report.py so the rows are directly comparable
# ======================================================================================
def auroc(y, s) -> float:
    y, s = np.asarray(y, int), np.asarray(s, float)
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    p, n = s[y == 1], s[y == 0]
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    allv = np.concatenate([p, n])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv))
    sv = allv[order]
    r = np.empty(len(allv))
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        r[i:j + 1] = (i + j) / 2 + 1
        i = j + 1
    ranks[order] = r
    return float((ranks[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def recall_at_fpr(y, s, fpr: float) -> float:
    y, s = np.asarray(y, int), np.asarray(s, float)
    s = np.where(np.isnan(s), -np.inf, s)
    neg = np.sort(s[y == 0])[::-1]
    if len(neg) == 0 or (y == 1).sum() == 0:
        return float("nan")
    k = int(math.floor(fpr * len(neg)))
    thr = neg[k] if k < len(neg) else -np.inf
    return float((s[y == 1] > thr).mean())


def det_metrics(d: pd.DataFrame) -> Dict[str, float]:
    if d.empty:
        return {}
    y, p = d["label"].astype(int).values, d["pred"].astype(int).values
    tp = int(((y == 1) & (p == 1)).sum()); fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum()); tn = int(((y == 0) & (p == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec == prec and rec == rec and prec + rec) \
        else float("nan")
    h = d[(d["label"] == 1) & (d["harmful"] == 1)]
    s = d["score"].astype(float).values
    calls = float(d["calls"].mean()) if "calls" in d.columns else 0.0
    return {"N": len(d), "Npos": int((y == 1).sum()), "Nharm": len(h),
            "Acc": (tp + tn) / len(d), "Prec": prec, "Rec": rec, "F1": f1,
            "FPR": fp / (fp + tn) if fp + tn else float("nan"),
            "HarmRec": float(h["pred"].mean()) if len(h) else float("nan"),
            "Calls": calls, "AUROC": auroc(y, s),
            "R@FPR5": recall_at_fpr(y, s, 0.05), "R@FPR10": recall_at_fpr(y, s, 0.10)}


def slices(d: pd.DataFrame) -> Dict[str, Dict]:
    """ALL, per source, per family, per method. Every slice keeps its own benign twins."""
    out = {"ALL": det_metrics(d)}
    for s, g in d.groupby("source"):
        out[f"source:{s}"] = det_metrics(g)
    for f, g in d.groupby("family"):
        out[f"family:{f}"] = det_metrics(g)
    for (s, m), g in d.groupby(["source", "method"]):
        out[f"method:{s}:{m}"] = det_metrics(g)
    return out


# ======================================================================================
# thresholds
# ======================================================================================
def find_insertion_predictions(*names: str) -> str:
    """The insertion run's predictions_all.csv for a baseline, by folder name or glob."""
    import glob
    for n in names:
        p = os.path.join(INSERTION_OUT, n, "predictions_all.csv")
        if os.path.exists(p):
            return p
        hits = sorted(glob.glob(os.path.join(INSERTION_OUT, n, "predictions_all.csv")))
        if hits:
            return hits[0]
    return ""


def threshold_from_insertion(pred_csv: str, score_col: str, fpr: float = 0.05
                             ) -> Tuple[float, Dict]:
    """Re-derive the frozen operating point exactly as the insertion script did:
    common.recall_at_fpr on the rows with split == 'val'."""
    d = pd.read_csv(pred_csv, low_memory=False,
                    usecols=lambda c: c in (score_col, "label", "split"))
    if score_col not in d.columns:
        return float("nan"), {"error": f"{score_col} not in {pred_csv}"}
    v = d[d["split"] == "val"] if "split" in d.columns else d
    C = parent_import("common")
    r, t = C.recall_at_fpr(v["label"].values, v[score_col].values, fpr)
    return float(t), {"from": pred_csv, "column": score_col, "val_rows": int(len(v)),
                      "val_recall_at_fpr": float(r), "fpr_budget": fpr}


def threshold_own(docs: pd.DataFrame, score: np.ndarray, fpr: float = 0.05) -> float:
    v = docs["split"].values == "val"
    C = parent_import("common")
    _, t = C.recall_at_fpr(docs["label"].values[v], score[v], fpr)
    return float(t)


def resolve_threshold(mode: str, docs: pd.DataFrame, score: np.ndarray, pred_csv: str,
                      score_col: str, default: float) -> Tuple[float, Dict, np.ndarray]:
    """(threshold, info, eval_mask). eval_mask is all-True for frozen thresholds and the
    non-validation papers for `own`."""
    info: Dict = {"mode": mode}
    if mode == "insertion":
        if pred_csv:
            t, i2 = threshold_from_insertion(pred_csv, score_col)
            info.update(i2)
            if t == t:          # +inf is legitimate: no threshold met the budget -> flags nothing
                info["threshold"] = t
                return t, info, np.ones(len(docs), bool)
            info["fallback_reason"] = "no threshold could be derived from the insertion predictions"
        else:
            info["fallback_reason"] = "insertion predictions_all.csv not found"
        print(f"[thresh] {info['fallback_reason']} -> falling back to `own`")
        mode = "own"
        info["mode"] = "own (fallback)"
    if mode.startswith("own"):
        t = threshold_own(docs, score)
        if t != t:
            t = default
            info["own_note"] = f"no validation documents; used {default}"
        info["threshold"] = t
        return t, info, docs["split"].values != "val"
    if mode == "fixed":
        info["threshold"] = default
        return default, info, np.ones(len(docs), bool)
    raise SystemExit(f"[FATAL] unknown threshold mode {mode}")


# ======================================================================================
# output
# ======================================================================================
class Out:
    """output/<run>/<baseline>/{output.txt, scores.csv, metrics.json, meta.json}.
    scores.csv is what ab_aggregate.py reads; metrics.json is for a quick look."""

    def __init__(self, baseline: str, run_tag: str = "", out_root: str = ""):
        self.baseline = baseline
        run = run_tag or "default"
        self.dir = os.path.join(out_root or OUT_ROOT, run, baseline)
        os.makedirs(self.dir, exist_ok=True)
        self.f = open(os.path.join(self.dir, "output.txt"), "w", encoding="utf-8")
        self.t0 = time.time()
        self("=" * 92)
        self(f"BASELINE ON ARGAR/BTR: {baseline}   run={run}")
        self(f"started {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self("=" * 92)

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()

    def write_scores(self, docs: pd.DataFrame, variant: str, label: str, meta: Dict,
                     mask: Optional[np.ndarray] = None, extra_cols: Sequence[str] = ()):
        """One table row = one variant: output/<run>/<baseline>/<variant>/scores.csv."""
        d = docs if mask is None else docs[mask]
        vdir = os.path.join(self.dir, variant)
        os.makedirs(vdir, exist_ok=True)
        cols = ["doc_id", "pair_uid", "paper_id", "source", "family", "method", "condition",
                "split", "label", "harmful", "score", "pred", "calls", "n_chars"] + \
            [c for c in extra_cols if c in d.columns]
        d2 = d.copy()
        if "calls" not in d2.columns:
            d2["calls"] = 0.0
        d2[cols].to_csv(os.path.join(vdir, "scores.csv"), index=False)
        m = slices(d2)
        meta = dict(meta, variant=variant, label=label, baseline=self.baseline,
                    n_docs=int(len(d2)))
        with open(os.path.join(vdir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)
        with open(os.path.join(vdir, "metrics.json"), "w") as f:
            json.dump(m, f, indent=2, default=float)
        self.table(m, f"{label}")
        return m

    def table(self, m: Dict[str, Dict], title: str):
        self("")
        self(f"--- {title} " + "-" * max(0, 80 - len(title)))
        cols = ["N", "Npos", "Rec", "Prec", "F1", "FPR", "HarmRec", "R@FPR5", "R@FPR10",
                "AUROC"]
        self(f"{'slice':34s}" + "".join(f"{c:>9s}" for c in cols))
        for k, v in m.items():
            if k.startswith("method:") and len(m) > 40:
                continue
            line = f"{k[:34]:34s}"
            for c in cols:
                x = v.get(c, float("nan"))
                line += (f"{x:9.0f}" if c in ("N", "Npos") else
                         (f"{x:9.3f}" if isinstance(x, float) and x == x else f"{'-':>9s}"))
            self(line)

    def close(self):
        self(f"\nfinished in {time.time() - self.t0:.1f}s -> {self.dir}")
        self.f.close()


# ======================================================================================
# gemma3-27b through vLLM (OpenAI-compatible), with a disk cache
# ======================================================================================
_JSON = re.compile(r"\{.*\}", re.S)


def parse_json(raw: str) -> Optional[Dict]:
    if not raw:
        return None
    t = raw.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I | re.M)
    for cand in (t, *(m.group(0) for m in [_JSON.search(t)] if m)):
        try:
            v = json.loads(cand)
            if isinstance(v, dict):
                return v
        except Exception:
            continue
    return None


def fit(doc: str, max_chars: int) -> str:
    """Over-long: keep the head AND the tail (v8 Reviewer._fit)."""
    if len(doc) <= max_chars:
        return doc
    h = max_chars // 3
    return doc[:h] + "\n\n[...]\n\n" + doc[-(max_chars - h):]


class LLM:
    """Minimal OpenAI-compatible chat client for a local vLLM server.

    Gemma's chat template has no system role, so a system prompt is folded into the user
    turn. Proxies are bypassed (the server is on 127.0.0.1)."""

    def __init__(self, url: str = "", model: str = "", cache_path: str = "",
                 max_tokens: int = 512, temperature: float = 0.0, timeout: int = 900,
                 system_role: bool = False, fake: Optional[Callable[[str], str]] = None):
        self.fake = fake
        self.url = (url or os.environ.get("VLLM_URL", "")).rstrip("/")
        if self.url and not self.url.endswith("/v1"):
            self.url += "/v1"
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.model = model or os.environ.get("VLLM_MODEL", "") or \
            (self._served() if (self.url and not fake) else "fake")
        self.max_tokens, self.temperature, self.timeout = max_tokens, temperature, timeout
        self.system_role = system_role
        self.lock = threading.Lock()
        self.cache: Dict[str, str] = {}
        self.cache_path = cache_path
        self.hits = self.calls = self.errors = 0
        if cache_path and os.path.exists(cache_path):
            for line in open(cache_path, encoding="utf-8"):
                try:
                    d = json.loads(line)
                    self.cache[d["k"]] = d["v"]
                except Exception:
                    continue
        if not fake and not self.url:
            raise SystemExit("[FATAL] no VLLM_URL. Start gemma with serve_gemma.sh first "
                             "(the PBS jobs do this for you).")

    def _served(self) -> str:
        with self.opener.open(f"{self.url}/models", timeout=30) as r:
            data = json.loads(r.read().decode())
        ids = [m["id"] for m in data.get("data", [])]
        if not ids:
            raise SystemExit(f"[FATAL] {self.url}/models lists no model")
        return ids[0]

    def chat(self, system: str, user: str, max_tokens: int = 0) -> str:
        mt = max_tokens or self.max_tokens
        key = hashlib.sha1(json.dumps([self.model, system, user, mt, self.temperature])
                           .encode("utf-8", "ignore")).hexdigest()
        with self.lock:
            if key in self.cache:
                self.hits += 1
                return self.cache[key]
        if self.fake:
            out = self.fake((system + "\n\n" + user) if system else user)
        else:
            msgs = ([{"role": "system", "content": system}, {"role": "user", "content": user}]
                    if (system and self.system_role) else
                    [{"role": "user", "content": (system + "\n\n" + user) if system else user}])
            body = json.dumps({"model": self.model, "messages": msgs, "max_tokens": mt,
                               "temperature": self.temperature}).encode()
            out = ""
            for attempt in range(3):
                try:
                    req = urllib.request.Request(f"{self.url}/chat/completions", data=body,
                                                 headers={"Content-Type": "application/json"})
                    with self.opener.open(req, timeout=self.timeout) as r:
                        out = json.loads(r.read().decode())["choices"][0]["message"]["content"] or ""
                    break
                except Exception as e:
                    if attempt == 2:
                        with self.lock:
                            self.errors += 1
                        print(f"[llm] call failed: {type(e).__name__}: {str(e)[:200]}")
                        return ""
                    time.sleep(5 * (attempt + 1))
        with self.lock:
            self.calls += 1
            self.cache[key] = out
            if self.cache_path and out:
                with open(self.cache_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"k": key, "v": out}, ensure_ascii=False) + "\n")
        return out


def pmap(fn: Callable, items: Sequence, workers: int = 8, label: str = "",
         every: int = 50) -> List:
    """Ordered parallel map with progress lines."""
    out: List = [None] * len(items)
    if workers <= 1:
        for i, x in enumerate(items):
            out[i] = fn(x)
            if label and (i + 1) % every == 0:
                print(f"[{label}] {i + 1}/{len(items)}", flush=True)
        return out
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, x): i for i, x in enumerate(items)}
        done = 0
        for fu in as_completed(futs):
            out[futs[fu]] = fu.result()
            done += 1
            if label and done % every == 0:
                print(f"[{label}] {done}/{len(items)}", flush=True)
    return out


def add_common_args(ap):
    ap.add_argument("--sources", nargs="+", default=list(SOURCES_DEFAULT),
                    choices=["argar", "btr", "insertion"])
    ap.add_argument("--subset", default="test", choices=["test", "full"],
                    help="test = the v8 test half (default, matches the v8 table); "
                         "full = every ARGAR/BTR row")
    ap.add_argument("--pairs-csv", default="", help="explicit pairs file (overrides)")
    ap.add_argument("--max-rows", type=int, default=0, help="pairs per source (0 = all)")
    ap.add_argument("--run-tag", default=os.environ.get("RUN_TAG", ""),
                    help="write to output/<run-tag>/ (default: output/default/)")
    ap.add_argument("--thresholds", default="insertion", choices=["insertion", "own"])
    return ap
