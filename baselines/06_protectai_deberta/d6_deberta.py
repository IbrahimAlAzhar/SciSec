#!/usr/bin/env python3
"""
d6_deberta.py — Baseline D6: ProtectAI deberta-v3-base-prompt-injection-v2.

    # once, on a login node that has network:
    huggingface-cli download protectai/deberta-v3-base-prompt-injection-v2 \
        --local-dir $HF_HOME/protectai-deberta-pi-v2

    python d6_deberta.py --model $HF_HOME/protectai-deberta-pi-v2

WHAT IT IS
    A 184M DeBERTa-v3 binary classifier, the de facto community baseline for prompt
    injection. Its own card reports accuracy 95.25 / precision 91.59 / recall 99.74 /
    F1 95.49 -- on ITS benchmark, which is short injected prompts, not 30,000-character
    manuscripts. Reproducing those numbers here is not the goal; finding out what they
    become on your data is.

    NOTE FOR THE PAPER: the ProtectAI project is ARCHIVED and no longer maintained. Say
    so when you cite it, or a reviewer will flag it as stale. Cite the model card; there
    is no paper.

WHY CHUNKING MATTERS AND HOW IT IS DONE HERE
    The model takes 512 tokens. A manuscript is far longer, so the document is split into
    overlapping windows and the document score is the MAXIMUM over windows -- one injected
    paragraph anywhere should fire, which is the right aggregation for detection and the
    wrong one for precision. That trade is the point of reporting FPR alongside recall.

    Max-over-windows also means the score rises slightly with document length even on
    clean text, because more windows means more chances to exceed any threshold. The
    report prints the score-vs-length correlation on CLEAN documents so you can see
    whether the detector is partly just measuring length.

LOCALIZATION, FREE
    The window with the highest score is recorded. If that window overlaps the true
    insert span (known here, since we placed it), the detection also LOCALIZED the
    attack. loc_iou / loc_hit turn a detector into a localizer at no extra cost, and
    localization is one of the claims your own system makes.
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

import common as C

DEFAULT_MODEL = "protectai/deberta-v3-base-prompt-injection-v2"


class Detector:
    def __init__(self, model_path: str, max_len: int = 512, stride_frac: float = 0.5,
                 batch: int = 16, device: str = "auto"):
        import torch
        from transformers import (AutoModelForSequenceClassification, AutoTokenizer)
        self.torch = torch
        dev = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.dev = dev
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.m = AutoModelForSequenceClassification.from_pretrained(model_path).to(dev).eval()
        self.max_len, self.batch = max_len, batch
        self.stride = max(64, int(max_len * stride_frac))
        # which logit column means "injection"? read it off the config, never assume 1.
        # Which logit column means "injection"? Read it off the config, never assume 1.
        # A flipped label index turns a working detector into an inverted one, and the
        # only symptom is a suspiciously low AUROC that looks like a weak model.
        id2 = getattr(self.m.config, "id2label", {}) or {}
        self.pos = 1
        for i, lab in id2.items():
            if str(lab).upper() in ("INJECTION", "LABEL_1", "UNSAFE", "JAILBREAK",
                                    "MALICIOUS", "ATTACK"):
                self.pos = int(i)
        print(f"[d6] {model_path} on {dev}   id2label={id2}   positive index={self.pos}")

    def windows(self, text: str):
        ids = self.tok(str(text), add_special_tokens=False,
                       truncation=False).input_ids
        if not ids:
            return [], []
        w = self.max_len - 2
        starts = list(range(0, max(1, len(ids) - w + 1), self.stride))
        if starts and starts[-1] + w < len(ids):
            starts.append(max(0, len(ids) - w))
        return [ids[s:s + w] for s in starts], starts

    def score_doc(self, text: str):
        """(max_score, best_window_index, n_windows, char_start, char_end)."""
        torch = self.torch
        chunks, starts = self.windows(text)
        if not chunks:
            return float("nan"), -1, 0, -1, -1
        best, best_i = -1e9, -1
        for i in range(0, len(chunks), self.batch):
            grp = chunks[i:i + self.batch]
            enc = self.tok.pad(
                {"input_ids": [self.tok.build_inputs_with_special_tokens(c) for c in grp]},
                return_tensors="pt").to(self.dev)
            with torch.no_grad():
                logits = self.m(**enc).logits.float()
            prob = torch.softmax(logits, -1)[:, self.pos].cpu().numpy()
            j = int(prob.argmax())
            if prob[j] > best:
                best, best_i = float(prob[j]), i + j
        # map the winning window back to character offsets, approximately
        s_tok = starts[best_i]
        pre = self.tok.decode(self.tok(str(text), add_special_tokens=False,
                                       truncation=False).input_ids[:s_tok])
        cs = len(pre)
        win = self.tok.decode(chunks[best_i])
        return best, best_i, len(chunks), cs, cs + len(win)



def iou(a0, a1, b0, b1) -> float:
    if min(a1, b1) <= max(a0, b0) or a0 < 0 or b0 < 0:
        return 0.0
    inter = min(a1, b1) - max(a0, b0)
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--wide-csv", default=C.WIDE_CSV)
    p.add_argument("--detail-csv", default=C.DETAIL_CSV)
    p.add_argument("--doc-col", default="")
    p.add_argument("--max-papers", type=int, default=0)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--out-dir", default="")
    a = p.parse_args()

    R = C.Report("D6_protectai_deberta", a.out_dir)
    R("NOTE: the ProtectAI project is ARCHIVED and unmaintained. Cite the model card")
    R("      (there is no paper) and state the archival status.")
    df = C.build_eval_frame(a.wide_csv, a.detail_csv, a.doc_col, a.max_papers)

    try:
        det = Detector(a.model, a.max_len, batch=a.batch)
    except Exception as e:
        R(f"[FATAL] could not load {a.model}: {type(e).__name__}: {e}")
        R("        HF_HUB_OFFLINE=1 is set on your cluster, so the weights must already")
        R("        be on disk. On a login node with network:")
        R("            huggingface-cli download protectai/deberta-v3-base-prompt-injection-v2 \\")
        R("                --local-dir $HF_HOME/protectai-deberta-pi-v2")
        R("        then pass --model $HF_HOME/protectai-deberta-pi-v2")
        R.close()
        return 1

    sc, bi, nw, cs, ce = [], [], [], [], []
    for i, t in enumerate(df["text"]):
        if i % 200 == 0:
            R(f"[d6] {i}/{len(df)}")
        s, b, n, c0, c1 = det.score_doc(t)
        sc.append(s); bi.append(b); nw.append(n); cs.append(c0); ce.append(c1)
    df["score"] = sc
    df["best_window"] = bi
    df["n_windows"] = nw
    df["win_start"] = cs
    df["win_end"] = ce
    df["loc_iou"] = [iou(a0, a1, b0, b1) for a0, a1, b0, b1 in
                     zip(df["win_start"], df["win_end"], df["ins_start"], df["ins_end"])]
    df["loc_hit"] = (df["loc_iou"] > 0).astype(int)

    # ---- threshold on val, frozen -------------------------------------------------
    val, test = df[df.split == "val"], df[df.split == "test"].copy()
    R("")
    for f in (0.05, 0.10):
        r, t = C.recall_at_fpr(val["label"].values, val["score"].values, f)
        R(f"[thresh] val recall@FPR<={f:.0%}: {r:.3f} at threshold {t:.4g}")
    _, T = C.recall_at_fpr(val["label"].values, val["score"].values, 0.05)
    if T != T:
        T = 0.5
        R("[thresh] no threshold met the FPR budget on val; falling back to 0.5")
    test["pred"] = (test["score"] >= T).astype(int)

    m = C.evaluate(test, calls_per_paper=0.0)
    R.table({k: v for k, v in m.items() if not k.startswith("cond:")},
            f"D6 / ProtectAI DeBERTa   (threshold {T:.4g}, frozen from val)")
    R.table({k: v for k, v in m.items() if k.startswith("cond:")},
            "D6 / per condition")
    R.save_json(m, "metrics.json")

    # ---- also at the card's own default, for reference ------------------------------
    t2 = test.copy()
    t2["pred"] = (t2["score"] >= 0.5).astype(int)
    m2 = C.evaluate(t2, calls_per_paper=0.0)
    R.table({k: v for k, v in m2.items() if not k.startswith("cond:")},
            "D6 / same model at the default 0.5 threshold")

    # ---- localization ----------------------------------------------------------------
    R("")
    R("--- localization: did the top-scoring window land on the real insert? " + "-" * 20)
    det_rows = test[(test.label == 1) & (test.pred == 1)]
    if len(det_rows):
        for fam, g in det_rows.groupby("family"):
            R(f"  {fam:13s} n={len(g):4d}  loc_hit {g['loc_hit'].mean():.3f}   "
              f"mean IoU {g['loc_iou'].mean():.3f}")
        R("  (computed only on DETECTED attacks; localizing what you did not detect is")
        R("   not a thing)")
    else:
        R("  nothing was detected, so there is nothing to localize.")

    # ---- is the score just measuring length? -----------------------------------------
    cl = test[test.label == 0]
    if len(cl) > 10 and cl["score"].notna().any():
        r = np.corrcoef(cl["n_chars"].values, cl["score"].values)[0, 1]
        R("")
        R(f"--- sanity: score vs length on CLEAN documents, r = {r:+.3f}")
        if abs(r) > 0.4:
            R("  WARNING: on clean papers the score tracks LENGTH. Max-over-windows gives")
            R("  a long document more chances to exceed any threshold, so part of this")
            R("  detector's behaviour is 'this paper is long'. Report it, and consider")
            R("  mean-over-windows or a length-matched comparison.")

    R("")
    R("WHAT TO COMPARE")
    R("  recall on `instruction` vs recall on `rhetorical`. This classifier was trained")
    R("  on injected INSTRUCTIONS. The rhetorical conditions contain none, so near-zero")
    R("  recall there is the expected result and it is worth a line in the paper: an")
    R("  off-the-shelf guardrail does not see suppression attacks at all.")

    R.save(test.drop(columns=["text", "doc_clean"]), "predictions_test.csv")
    R.save(df.drop(columns=["text", "doc_clean"]), "predictions_all.csv")
    R.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())