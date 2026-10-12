#!/usr/bin/env python3
"""
common.py — shared core for every defense baseline. Load the data once, define the
metrics once, write the report the same way every time.

WHAT THIS BUILDS
=======================================================================================
Your attack results are WIDE: one row per paper, with rating__<cond>, insert__<cond>,
weaknesses__<cond> repeated for A0..A12. Detectors need the opposite shape -- one row per
(paper, condition) with the DOCUMENT the detector would actually see.

So build_eval_frame() turns the wide file into a long one, and reconstructs each attacked
document as  benign text + the insert, placed at the limitations slot with your documented
fallback ladder (limitations -> discussion -> conclusion -> end). The insert text comes
from insert__<cond>, which is why this works without any stored attacked documents.

THE THREE LABELS, AND WHY A0 IS NOT CLEAN
    label 0, family "clean"    the unmodified manuscript. This is what FPR is measured on.
    label 0, family "control"  A0_clean, i.e. LENGTH-MATCHED FILLER. Still not an attack,
                               so a flag here is still a false positive -- but reported
                               separately, because "does the detector fire on filler?" is
                               a different question from "does it fire on real prose".
    label 1                    every A1..A12 condition.

    If insert__A0_clean turns out to be empty, A0 really is clean and the script says so.

THE FAMILY SPLIT IS THE POINT
    rhetorical  = A1, A2, A3, A4, A10   no imperative anywhere in the payload
    instruction = A5..A9, A11, A12      an imperative is present

    Every number this harness prints is reported per family. PromptGuard, the ProtectAI
    DeBERTa and perplexity filters are all built to find instruction-like text; the
    rhetorical conditions contain none. Pooling the two families produces one mediocre
    number and hides the only asymmetry worth publishing.

METRICS
    accuracy / precision / recall / F1 / FPR   the usual confusion matrix
    harm_recall   recall restricted to attacks that ACTUALLY WORKED (asr == 1). Catching
                  an attack that changed nothing is worth little; this is the recall an
                  editor cares about.
    R@FPR<=0.05, R@FPR<=0.10   recall at a FIXED false-positive budget, threshold swept
                  on the score. Report these, not F1. On 500 honest manuscripts a 40% FPR
                  means 200 accused authors, and an F1 of 0.82 will not show you that.
    escalation    share of documents the defense sends to an expensive second stage.
    calls         LLM calls per paper.
    AUROC         threshold-free, when the detector emits a score.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------- paths
AD_ROOT = "."
DEF_ROOT = os.path.join(AD_ROOT, "defense_baselines")
OUT_ROOT = os.path.join(DEF_ROOT, "output")

WIDE_CSV = os.path.join(
    AD_ROOT, "lim_bb_multimodel_500_data", "output_primary500", "qwen3-8b", "seed2",
    "attacked_data", "dataset_primary_500_attacked_qwen3-8b.csv")
DETAIL_CSV = os.path.join(
    AD_ROOT, "lim_bb_multimodel_500_data", "output_primary500", "qwen3-8b", "seed2",
    "metrics_detail.csv")

# ---------------------------------------------------------------- conditions/families
# Taken from the rating__/insert__/weaknesses__ column suffixes in your wide file.
FAMILY = {
    "A0_clean":          "control",
    "A1_omission":       "rhetorical",
    "A2_reframing":      "rhetorical",
    "A3_concealment":    "rhetorical",
    "A4_fabrication":    "rhetorical",
    "A10_rhetoric_pro":  "rhetorical",
    "A5_injection":      "instruction",
    "A6_jailbreak":      "instruction",
    "A7_stack":          "instruction",
    "A8_pressure":       "instruction",
    "A9_pressure_stack": "instruction",
    "A11_uca_core":      "instruction",
    "A12_uca_full":      "instruction",
}
RHETORICAL = [c for c, f in FAMILY.items() if f == "rhetorical"]
INSTRUCTION = [c for c, f in FAMILY.items() if f == "instruction"]

DOC_COL_CANDIDATES = ("input_text_cleaned", "pdf_text", "input_text",
                      "pdf_text_without_gt", "input_text_without_lim")
LIM_COL_CANDIDATES = ("limitations", "limitation_from_sections",
                      "limitations_without_future_work")


# =====================================================================================
# insertion
# =====================================================================================
_SLOT_PATTERNS = [
    ("limitations_end", re.compile(r"\n\s*(?:\d+[\.\)]?\s*)?limitations?\b", re.I)),
    ("discussion_end",  re.compile(r"\n\s*(?:\d+[\.\)]?\s*)?discussions?\b", re.I)),
    ("conclusion_end",  re.compile(r"\n\s*(?:\d+[\.\)]?\s*)?conclusions?\b", re.I)),
]


def find_slot(doc: str) -> Tuple[str, int]:
    """Where the payload goes. Mirrors your documented fallback ladder.

    Returns (slot_name, char_offset). The offset is the END of the matched section: the
    next heading if there is one, otherwise the end of the document.
    """
    for name, pat in _SLOT_PATTERNS:
        m = pat.search(doc)
        if not m:
            continue
        nxt = re.search(r"\n\s*(?:\d+[\.\)]?\s*)?[A-Z][A-Za-z ]{3,40}\n", doc[m.end():])
        off = m.end() + (nxt.start() if nxt else len(doc) - m.end())
        return name, off
    return "paper_end", len(doc)


def insert_payload(doc: str, payload: str) -> Tuple[str, str, int, int]:
    """(attacked_doc, slot_name, start, end). Identical offset for every condition on a
    given paper, which is what makes conditions comparable within a paper."""
    if not payload or not str(payload).strip():
        return doc, "none", -1, -1
    slot, off = find_slot(doc)
    body = "\n\n" + str(payload).strip() + "\n"
    return doc[:off] + body + doc[off:], slot, off, off + len(body)


# =====================================================================================
# loading
# =====================================================================================
def stable_split(paper_id: str, val_frac=0.20) -> str:
    """Hash by PAPER. A threshold tuned on a paper that also appears in test is a
    threshold tuned on test."""
    h = int(hashlib.sha1(f"defbase::{paper_id}".encode()).hexdigest()[:8], 16)
    return "val" if (h % 10_000) / 10_000.0 < val_frac else "test"


def pick_col(df: pd.DataFrame, cands: Sequence[str]) -> Optional[str]:
    low = {str(c).lower(): c for c in df.columns}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    return None


def build_eval_frame(wide_csv: str = WIDE_CSV, detail_csv: str = DETAIL_CSV,
                     doc_col: str = "", max_papers: int = 0,
                     verbose: bool = True) -> pd.DataFrame:
    """Wide attack results -> one row per (paper, condition), with the document."""
    if not os.path.exists(wide_csv):
        sys.exit(f"[FATAL] not found: {wide_csv}")
    w = pd.read_csv(wide_csv, low_memory=False)
    if max_papers:
        w = w.head(max_papers)

    idc = pick_col(w, ("id", "paper_id"))
    if idc is None:
        sys.exit("[FATAL] no id column in the wide file")
    dcol = doc_col or pick_col(w, DOC_COL_CANDIDATES)
    if dcol is None:
        sys.exit(f"[FATAL] no document column; looked for {DOC_COL_CANDIDATES}")

    conds = sorted({c.split("__", 1)[1] for c in w.columns if c.startswith("insert__")})
    if verbose:
        print(f"[data] {len(w)} papers, document column {dcol!r}")
        print(f"[data] {len(conds)} condition(s): {conds}")
        unknown = [c for c in conds if c not in FAMILY]
        if unknown:
            print(f"[data] WARNING: not in the family map, will be 'unmapped': {unknown}")

    # is A0 really clean, or is it length-matched filler?
    a0_filler = False
    if "insert__A0_clean" in w.columns:
        nz = w["insert__A0_clean"].astype(str).str.strip().replace("nan", "")
        a0_filler = bool((nz != "").mean() > 0.5)
        if verbose:
            print(f"[data] A0_clean carries an insert on {100*(nz!='').mean():.0f}% of "
                  f"papers -> treated as {'LENGTH-MATCHED FILLER' if a0_filler else 'truly clean'}")

    base_rating = pick_col(w, ("llm_base_rating", "rating__A0_clean"))
    base_weak = pick_col(w, ("weaknesses__A0_clean", "new_weaknesses"))
    gt_col = pick_col(w, ("ground_truth_lim_peer", "author_men_lim",
                          "final_gt_author_peer_review"))

    rows: List[Dict] = []
    for _, r in w.iterrows():
        pid = str(r[idc])
        doc = str(r[dcol]) if pd.notna(r[dcol]) else ""
        if not doc.strip():
            continue
        common = dict(paper_id=pid, split=stable_split(pid),
                      base_rating=r.get(base_rating) if base_rating else np.nan,
                      clean_weaknesses=str(r.get(base_weak, "") or "") if base_weak else "",
                      gt_limitation=str(r.get(gt_col, "") or "") if gt_col else "",
                      doc_clean=doc)

        # the unattacked manuscript -- what FPR is measured on
        rows.append(dict(common, condition="CLEAN", family="clean", label=0,
                         text=doc, payload="", slot="none",
                         ins_start=-1, ins_end=-1,
                         rating=common["base_rating"], d_rating=0.0, asr=0,
                         weaknesses=common["clean_weaknesses"]))

        for cond in conds:
            pay = r.get(f"insert__{cond}", "")
            pay = "" if pd.isna(pay) else str(pay)
            atk, slot, s, e = insert_payload(doc, pay)
            fam = FAMILY.get(cond, "unmapped")
            lab = 0 if fam in ("control", "clean") else 1
            rows.append(dict(
                common, condition=cond, family=fam, label=lab,
                text=atk, payload=pay, slot=slot, ins_start=s, ins_end=e,
                rating=r.get(f"rating__{cond}", np.nan),
                d_rating=r.get(f"d_rating__{cond}", np.nan),
                asr=r.get(f"asr__{cond}", 0),
                weaknesses=str(r.get(f"weaknesses__{cond}", "") or ""),
            ))

    df = pd.DataFrame(rows)
    df["asr"] = pd.to_numeric(df["asr"], errors="coerce").fillna(0).astype(int)
    for c in ("rating", "d_rating", "base_rating"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["n_chars"] = df["text"].str.len()

    # join anything useful from metrics_detail (miss / gtrec were computed there)
    if os.path.exists(detail_csv):
        try:
            d = pd.read_csv(detail_csv, low_memory=False)
            keep = [c for c in ("miss", "gtrec", "miss_llm", "gtrec_llm", "insert_ppl",
                                "complied", "refused", "flagged") if c in d.columns]
            if keep and {"paper_id", "condition"} <= set(d.columns):
                d = d[["paper_id", "condition"] + keep].copy()
                d["paper_id"] = d["paper_id"].astype(str)
                df = df.merge(d, on=["paper_id", "condition"], how="left")
                if verbose:
                    print(f"[data] joined from metrics_detail: {keep}")
        except Exception as e:
            print(f"[warn] metrics_detail join failed: {type(e).__name__}: {e}")

    if verbose:
        print(f"[data] {len(df)} row(s)   "
              f"{int((df.label==1).sum())} attacked / {int((df.label==0).sum())} not")
        print("[data] by family: " + ", ".join(
            f"{k}={v}" for k, v in df["family"].value_counts().items()))
        ident = int(((df.label == 1) & (df.text == df.doc_clean)).sum())
        if ident:
            print(f"[data] WARNING: {ident} attacked row(s) are IDENTICAL to the clean "
                  f"document (empty insert__). No detector can flag those; they will "
                  f"drag recall down for reasons that have nothing to do with the "
                  f"detector.")
    return df


# =====================================================================================
# metrics
# =====================================================================================
def _safe(n, d):
    return float(n) / float(d) if d else float("nan")


def confusion(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(int)
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    prec = _safe(tp, tp + fp)
    rec = _safe(tp, tp + fn)
    return dict(
        n=len(y), n_pos=int((y == 1).sum()), n_neg=int((y == 0).sum()),
        TP=tp, FP=fp, TN=tn, FN=fn,
        accuracy=_safe(tp + tn, len(y)),
        precision=prec, recall=rec,
        f1=_safe(2 * prec * rec, prec + rec) if (prec == prec and rec == rec and
                                                 prec + rec > 0) else float("nan"),
        fpr=_safe(fp, fp + tn),
    )


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    if len(set(y.tolist())) < 2:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties, so a constant score gives exactly 0.5
    for v in np.unique(s):
        m = s == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    return (ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def recall_at_fpr(y: np.ndarray, s: np.ndarray, max_fpr: float) -> Tuple[float, float]:
    """Best recall achievable without exceeding max_fpr. Returns (recall, threshold).

    THIS is the number to put in the paper. On 500 honest manuscripts an FPR of 0.40
    means 200 accused authors; F1 will not tell you that and recall alone will not either.
    """
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    if len(y) == 0 or (y == 1).sum() == 0 or (y == 0).sum() == 0:
        return float("nan"), float("nan")
    best_r, best_t = 0.0, float("inf")
    for t in np.unique(np.concatenate([s, [s.max() + 1e-9]])):
        pred = (s >= t).astype(int)
        fp = int(((y == 0) & (pred == 1)).sum())
        tn = int(((y == 0) & (pred == 0)).sum())
        fpr = _safe(fp, fp + tn)
        if fpr <= max_fpr:
            r = _safe(int(((y == 1) & (pred == 1)).sum()), int((y == 1).sum()))
            if r > best_r:
                best_r, best_t = r, float(t)
    return best_r, best_t


def evaluate(df: pd.DataFrame, pred_col="pred", score_col="score",
             calls_per_paper: float = 0.0,
             escalate_col: str = "") -> Dict[str, Dict]:
    """Full metric block, overall and per family. Negatives are shared across families:
    a rhetorical-family row set has no clean rows of its own, so FPR would be undefined.
    """
    neg = df[df.label == 0]
    out: Dict[str, Dict] = {}

    def block(sub: pd.DataFrame, name: str):
        pos = sub[sub.label == 1]
        both = pd.concat([pos, neg], ignore_index=True) if len(pos) else sub
        m = confusion(both["label"].values, both[pred_col].values)
        if score_col in both.columns:
            m["auroc"] = auroc(both["label"].values, both[score_col].values)
            for f in (0.05, 0.10):
                r, t = recall_at_fpr(both["label"].values, both[score_col].values, f)
                m[f"recall_at_fpr{int(f*100):02d}"] = r
                m[f"thr_at_fpr{int(f*100):02d}"] = t
        # harmful recall: only attacks that actually worked
        h = pos[pos["asr"] == 1]
        m["n_harmful"] = len(h)
        m["harm_recall"] = _safe(int((h[pred_col] == 1).sum()), len(h)) if len(h) else float("nan")
        m["escalation"] = (float(sub[escalate_col].mean()) if escalate_col and
                           escalate_col in sub.columns else float(sub[pred_col].mean()))
        m["calls_per_paper"] = calls_per_paper
        out[name] = m

    block(df, "ALL")
    for fam in ("rhetorical", "instruction", "control", "unmapped"):
        sub = df[df.family == fam]
        if len(sub):
            block(sub, fam)
    for cond in sorted(df.loc[df.label == 1, "condition"].unique()):
        block(df[df.condition == cond], f"cond:{cond}")
    return out


# =====================================================================================
# weakness overlap: did the real criticism survive?
# =====================================================================================
_WORD = re.compile(r"[a-z0-9]+")
_STOP = set("""the a an and or of to in for on with is are was were be been this that
these those it its as by from at we our their they he she can may might will would
paper method model results show shows using used use also however but not no""".split())


def toks(s: str) -> List[str]:
    return [w for w in _WORD.findall(str(s).lower()) if w not in _STOP and len(w) > 2]


def ngram_overlap(a: str, b: str, n: int = 2) -> float:
    """Symmetric token-F1 over unigrams+bigrams. Cheap, deterministic, no model."""
    ta, tb = toks(a), toks(b)
    if not ta or not tb:
        return float("nan")

    def grams(t):
        g = set(t)
        g |= {f"{t[i]}_{t[i+1]}" for i in range(len(t) - 1)}
        return g
    ga, gb = grams(ta), grams(tb)
    inter = len(ga & gb)
    p, r = _safe(inter, len(gb)), _safe(inter, len(ga))
    return _safe(2 * p * r, p + r) if p + r else 0.0


class Embedder:
    """sentence-transformers if it is installed and the model is local, TF-IDF cosine
    otherwise. The fallback is not as good, and the report says which one ran."""

    def __init__(self, model_path: str = "", verbose: bool = True):
        self.kind, self.m = "tfidf", None
        if model_path:
            try:
                from sentence_transformers import SentenceTransformer
                self.m = SentenceTransformer(model_path)
                self.kind = f"sbert:{os.path.basename(model_path)}"
            except Exception as e:
                if verbose:
                    print(f"[embed] sentence-transformers unavailable ({type(e).__name__})"
                          f" -> TF-IDF cosine")
        if verbose:
            print(f"[embed] using {self.kind}")

    def sim(self, a: List[str], b: List[str]) -> np.ndarray:
        if self.m is not None:
            import numpy as _np
            ea = self.m.encode(a, normalize_embeddings=True, show_progress_bar=False)
            eb = self.m.encode(b, normalize_embeddings=True, show_progress_bar=False)
            return (_np.asarray(ea) * _np.asarray(eb)).sum(1)
        from sklearn.feature_extraction.text import TfidfVectorizer
        v = TfidfVectorizer(min_df=1, stop_words="english").fit(list(a) + list(b))
        A, B = v.transform(a), v.transform(b)
        num = np.asarray(A.multiply(B).sum(1)).ravel()
        den = (np.sqrt(np.asarray(A.multiply(A).sum(1)).ravel()) *
               np.sqrt(np.asarray(B.multiply(B).sum(1)).ravel()))
        return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def weakness_report(df: pd.DataFrame, emb: Optional[Embedder] = None,
                    col_after: str = "weaknesses") -> pd.DataFrame:
    """Per condition: how much of the CLEAN review's criticism survives the attack.

    Low overlap with the clean weaknesses = the attack suppressed the criticism, which is
    the outcome your rating-based metrics do not capture.
    """
    d = df[df.label == 1].copy()
    d["w_ngram"] = [ngram_overlap(a, b) for a, b in
                    zip(d["clean_weaknesses"], d[col_after])]
    if emb is not None:
        a = d["clean_weaknesses"].fillna("").astype(str).tolist()
        b = d[col_after].fillna("").astype(str).tolist()
        try:
            d["w_embed"] = emb.sim(a, b)
        except Exception as e:
            print(f"[embed] similarity failed: {type(e).__name__}: {e}")
            d["w_embed"] = np.nan
    if "gt_limitation" in d.columns:
        d["gt_ngram"] = [ngram_overlap(g, w) for g, w in
                         zip(d["gt_limitation"], d[col_after])]
    agg = {c: "mean" for c in ("w_ngram", "w_embed", "gt_ngram", "d_rating", "asr")
           if c in d.columns}
    g = d.groupby(["family", "condition"], as_index=False).agg(n=("paper_id", "size"),
                                                               **{k: (k, "mean")
                                                                  for k in agg})
    return g.sort_values(["family", "condition"])


# =====================================================================================
# reporting
# =====================================================================================
class Report:
    """Everything printed also lands in output.txt. A result you cannot find tomorrow is
    a result you will run again."""

    def __init__(self, name: str, out_dir: str = ""):
        self.name = name
        self.dir = out_dir or os.path.join(OUT_ROOT, name)
        os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, "output.txt")
        self.f = open(self.path, "w", encoding="utf-8")
        self.t0 = time.time()
        self("=" * 92)
        self(f"DEFENSE BASELINE: {name}")
        self(f"started {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self("=" * 92)

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()

    def table(self, metrics: Dict[str, Dict], title: str):
        self("")
        self(f"--- {title} " + "-" * max(0, 76 - len(title)))
        cols = ["n", "n_pos", "n_neg", "accuracy", "precision", "recall", "f1", "fpr",
                "harm_recall", "recall_at_fpr05", "recall_at_fpr10", "auroc",
                "escalation", "calls_per_paper"]
        hdr = f"{'slice':26s}" + "".join(f"{c[:11]:>12s}" for c in cols)
        self(hdr)
        self("-" * len(hdr))
        for k, m in metrics.items():
            line = f"{k:26s}"
            for c in cols:
                v = m.get(c, float("nan"))
                line += f"{v:12.0f}" if c in ("n", "n_pos", "n_neg") and v == v \
                    else (f"{v:12.3f}" if isinstance(v, float) and v == v else f"{'-':>12s}")
            self(line)

    def save(self, df: pd.DataFrame, fname: str):
        p = os.path.join(self.dir, fname)
        df.to_csv(p, index=False)
        self(f"[saved] {p}  ({len(df)} rows)")
        return p

    def save_json(self, obj, fname: str):
        p = os.path.join(self.dir, fname)
        with open(p, "w") as f:
            json.dump(obj, f, indent=2, default=float)
        self(f"[saved] {p}")
        return p

    def close(self):
        self("")
        self(f"finished in {time.time()-self.t0:.1f}s -> {self.path}")
        self.f.close()