#!/usr/bin/env python3
"""
ab_classifiers.py — prompt-injection classifiers on ARGAR/BTR: D6 ProtectAI DeBERTa-v2
and PIGuard.

    python ab_classifiers.py --detector protectai --model <dir of protectai-deberta-pi-v2>
    python ab_classifiers.py --detector piguard   --model <dir of leolee99/PIGuard>

ONE PROTOCOL FOR BOTH
    512-token windows, 50% overlap, document score = MAX over windows (one malicious
    paragraph anywhere should fire), positive class read off the model's id2label (never
    assumed to be index 1). This is d6_deberta.py's protocol; for ProtectAI the Detector
    class is imported from it unchanged. PIGuard (Li et al., ACL 2025, leolee99/PIGuard)
    is a DeBERTa-v3 classifier trained to reduce over-defense; it ships custom model code,
    so it is loaded with trust_remote_code=True from a LOCAL snapshot (fetch_models_ab.py).

OPERATING POINT
    --thresholds insertion (default): the threshold the insertion run chose on its
    validation papers (best recall at FPR <= 5%), read from
    output/D6_protectai_deberta/predictions_all.csv, or the PIGuard run's folder (found by
    name; pass --insertion-dir to point at it). Falls back to `own` if not found, and says so.
    The model card's default 0.5 is written as a second variant for reference.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

import ab_common as A


class WindowClassifier:
    """d6_deberta.Detector with trust_remote_code, for models that ship their own code."""

    def __init__(self, model_path: str, max_len: int = 512, stride_frac: float = 0.5,
                 batch: int = 16, trust_remote_code: bool = False):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        self.m = AutoModelForSequenceClassification.from_pretrained(
            model_path, trust_remote_code=trust_remote_code).to(self.dev).eval()
        self.max_len, self.batch = max_len, batch
        self.stride = max(64, int(max_len * stride_frac))
        id2 = getattr(self.m.config, "id2label", {}) or {}
        self.pos = 1
        for i, lab in id2.items():
            if str(lab).upper() in ("INJECTION", "LABEL_1", "UNSAFE", "JAILBREAK",
                                    "MALICIOUS", "ATTACK"):
                self.pos = int(i)
        print(f"[clf] {model_path} on {self.dev}  id2label={id2}  positive={self.pos}")

    def score_doc(self, text: str):
        torch = self.torch
        ids = self.tok(str(text), add_special_tokens=False, truncation=False).input_ids
        if not ids:
            return float("nan"), -1, 0, -1, -1
        w = self.max_len - 2
        starts = list(range(0, max(1, len(ids) - w + 1), self.stride))
        if starts and starts[-1] + w < len(ids):
            starts.append(max(0, len(ids) - w))
        chunks = [ids[s:s + w] for s in starts]
        best, best_i = -1e9, -1
        for i in range(0, len(chunks), self.batch):
            grp = chunks[i:i + self.batch]
            enc = self.tok.pad({"input_ids": [self.tok.build_inputs_with_special_tokens(c)
                                              for c in grp]}, return_tensors="pt").to(self.dev)
            with torch.no_grad():
                prob = torch.softmax(self.m(**enc).logits.float(), -1)[:, self.pos].cpu().numpy()
            j = int(prob.argmax())
            if prob[j] > best:
                best, best_i = float(prob[j]), i + j
        return best, best_i, len(chunks), -1, -1


SPEC = {
    "protectai": dict(baseline="D6_protectai_deberta", label="D6 ProtectAI DeBERTa-v2",
                      dirs=("protectai-deberta-pi-v2",),
                      insertion=("D6_protectai_deberta", "D6*"), remote=False),
    "piguard": dict(baseline="PIGuard", label="PIGuard",
                    dirs=("piguard", "PIGuard", "leolee99-PIGuard"),
                    insertion=("*[Pp][Ii][Gg]uard*", "*PIGUARD*"), remote=True),
}


def main() -> int:
    ap = A.add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--detector", required=True, choices=sorted(SPEC))
    ap.add_argument("--model", default="")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--insertion-dir", default="",
                    help="folder (under defense_baselines/output) of this detector's "
                         "insertion run, if its name is not the usual one")
    a = ap.parse_args()
    sp = SPEC[a.detector]
    model = a.model or A.first_dir(*[f"{A.MODELS_DIR}/{d}" for d in sp["dirs"]],
                                   *[f"{A.HF_HOME}/{d}" for d in sp["dirs"]])
    out = A.Out(sp["baseline"], a.run_tag)
    if a.detector == "protectai":
        out("NOTE: the ProtectAI project is ARCHIVED; cite the model card and say so.")
    pairs = A.load_pairs(a.sources, a.subset, a.max_rows, a.pairs_csv)
    docs = A.docs_frame(pairs)
    texts, inv = A.unique_texts(docs)
    out(f"[data] {len(docs)} documents, {len(texts)} distinct texts")

    try:
        if a.detector == "protectai":
            D6 = A.parent_import("d6_deberta")
            det = D6.Detector(model, a.max_len, batch=a.batch)
            out(f"[reuse] Detector from {D6.__file__}")
        else:
            det = WindowClassifier(model, a.max_len, batch=a.batch, trust_remote_code=sp["remote"])
    except Exception as e:
        out(f"[FATAL] could not load {model}: {type(e).__name__}: {e}")
        out("        The weights must be on disk (HF_HUB_OFFLINE=1). On a login node:")
        out("            python fetch_models_ab.py")
        out.close()
        return 1

    sc = []
    for i, t in enumerate(texts):
        if i % 100 == 0:
            out(f"[{a.detector}] {i}/{len(texts)}")
        try:
            sc.append(det.score_doc(t)[0])
        except Exception as e:
            print(f"[{a.detector}] doc {i}: {type(e).__name__}: {e}")
            sc.append(float("nan"))
    docs["clf_score"] = np.asarray(sc, float)[inv]

    names = (a.insertion_dir,) if a.insertion_dir else sp["insertion"]
    pred_csv = A.find_insertion_predictions(*names)
    out(f"[thresh] mode={a.thresholds}   insertion predictions: {pred_csv or 'NOT FOUND'}")
    s = docs["clf_score"].values
    t, info, mask = A.resolve_threshold(a.thresholds, docs, s, pred_csv, "score", 0.5)
    for variant, label, thr, inf, msk in (
            ("frozen", sp["label"], t, info, mask),
            ("card_default_0.5", f"{sp['label']} (threshold 0.5)", 0.5, {"mode": "fixed 0.5"},
             np.ones(len(docs), bool))):
        d = docs.copy()
        d["score"] = s
        d["pred"] = (np.nan_to_num(s, nan=-np.inf) >= thr).astype(int)
        out(f"[thresh] {variant}: {thr:.6g}   {inf}")
        out.write_scores(d, variant, label, {"threshold": thr, "threshold_info": inf,
                                             "model": model}, msk)

    cl = docs[docs.label == 0]
    if len(cl) > 10 and np.nanstd(cl["clf_score"].values) > 0 and cl["n_chars"].std() > 0:
        r = np.corrcoef(cl["n_chars"].values, np.nan_to_num(cl["clf_score"].values))[0, 1]
        out(f"\n[sanity] score vs length on benign documents: r = {r:+.3f}"
            + ("   (WARNING: the max-over-windows score tracks length)" if abs(r) > 0.4 else ""))
    out.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
