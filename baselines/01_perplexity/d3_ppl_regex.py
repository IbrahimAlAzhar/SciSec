#!/usr/bin/env python3
"""
d3_ppl_regex.py — Baseline D3: perplexity filter + regex, after Jain et al. (2309.00614).

    python d3_ppl_regex.py --ppl-model /path/to/gpt2        # full
    python d3_ppl_regex.py --no-ppl                          # regex only, pure CPU

WHAT IT DOES
    Two independent screens, reported separately and combined:

    REGEX   hand-written patterns for imperative injection -- "ignore previous
            instructions", "give a positive review", "you must accept", and so on.
            This is the strongest cheap baseline against the instruction family and,
            by construction, blind to the rhetorical one: there is no imperative in a
            vague limitations paragraph to match.

    PPL     sliding-window perplexity under a small causal LM. Jain et al. showed that
            gradient-optimized suffixes are wildly improbable text, so a perplexity
            filter catches them almost for free. The same argument predicts it will do
            nothing against FLUENT attacks -- which is most of yours. The score is
            max window perplexity, so one strange passage in an otherwise normal paper
            is enough to fire.

    Both emit a continuous score, so recall at a fixed FPR is well defined rather than
    depending on where a threshold happened to land.

WHY THIS IS THE MOST INFORMATIVE CHEAP BASELINE YOU CAN RUN
    If regex gets high recall on `instruction` and near-zero on `rhetorical`, that single
    contrast is the evidence that your rhetorical family is a genuinely different threat
    and not a weaker version of the same one. It costs minutes of CPU.
"""

from __future__ import annotations

import argparse
import math
import re
import numpy as np
import pandas as pd

import common as C

# ------------------------------------------------------------------ regex screen
PATTERNS = [
    (r"\bignore\s+(all\s+|any\s+)?(previous|prior|above|earlier)\s+(instruction|prompt|direction)", 3.0),
    (r"\bdisregard\s+(all\s+|any\s+)?(previous|prior|above|the)\b",                                 3.0),
    (r"\bgive\s+(a\s+)?positive\s+review\b",                                                        3.0),
    (r"\b(accept|recommend\s+acceptance)\s+(this|the)\s+(paper|submission|manuscript)\b",           2.5),
    (r"\byou\s+(must|should|need\s+to|have\s+to)\s+(accept|approve|rate|score|give)\b",             2.5),
    (r"\b(as\s+an?\s+)?(ai|language\s+model|llm|reviewer)\s*[,:]?\s*(you|please)\b",                1.5),
    (r"\b(do\s+not|don'?t|never)\s+(mention|report|list|include|raise)\b",                          2.0),
    (r"\b(rate|score|grade)\s+(this|it)\s+(as\s+)?(a\s+)?(10|nine|ten|highest|maximum)\b",          2.5),
    (r"\bsystem\s*(prompt|message|instruction)\b",                                                  1.5),
    (r"\b(new|updated|revised)\s+(instruction|task|directive)s?\s*[:：]",                           2.0),
    (r"\b(important|urgent|critical|mandatory)\s*[:：]\s*",                                         1.0),
    (r"\bthis\s+paper\s+(must|should)\s+be\s+accepted\b",                                           3.0),
    (r"\b(reviewer|referee)s?\s+(who|that)\s+(reject|criticize)\b",                                 2.0),
    (r"\b(no|without)\s+(weakness|limitation|flaw|criticism)e?s?\b",                                 1.5),
    (r"[​-‏‪-‮﻿]",                                                          3.0),  # invisible
]
COMPILED = [(re.compile(p, re.I), w) for p, w in PATTERNS]


def regex_score(text: str) -> float:
    """Sum of weights of matching patterns. A score, not a boolean, so a threshold can
    be swept."""
    t = str(text)
    return float(sum(w for rx, w in COMPILED if rx.search(t)))


def regex_hits(text: str):
    return [rx.pattern[:44] for rx, _ in COMPILED if rx.search(str(text))]


# ------------------------------------------------------------------ perplexity screen
class PPL:
    def __init__(self, model_path: str, window: int = 128, stride: int = 64,
                 device: str = "auto", max_windows: int = 120):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        dev = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.dev = dev
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.m = AutoModelForCausalLM.from_pretrained(model_path).to(dev).eval()
        self.window, self.stride, self.max_windows = window, stride, max_windows
        print(f"[ppl] {model_path} on {dev}   window={window} stride={stride}")

    def max_window_ppl(self, text: str) -> float:
        """Highest perplexity of any window. One improbable passage is enough."""
        torch = self.torch
        ids = self.tok(str(text), return_tensors="pt",
                       truncation=False).input_ids[0]
        if len(ids) < 8:
            return float("nan")
        starts = list(range(0, max(1, len(ids) - self.window + 1), self.stride))
        if len(starts) > self.max_windows:                    # cap cost on long papers
            idx = np.linspace(0, len(starts) - 1, self.max_windows).astype(int)
            starts = [starts[i] for i in idx]
        worst = 0.0
        with torch.no_grad():
            for i in range(0, len(starts), 8):                # batch of windows
                chunk = [ids[s:s + self.window] for s in starts[i:i + 8]]
                chunk = [c for c in chunk if len(c) >= 8]
                if not chunk:
                    continue
                L = max(len(c) for c in chunk)
                pad = self.tok.pad_token_id or self.tok.eos_token_id or 0
                batch = torch.full((len(chunk), L), pad, dtype=torch.long)
                mask = torch.zeros((len(chunk), L), dtype=torch.long)
                for j, c in enumerate(chunk):
                    batch[j, :len(c)] = c
                    mask[j, :len(c)] = 1
                batch, mask = batch.to(self.dev), mask.to(self.dev)
                logits = self.m(input_ids=batch, attention_mask=mask).logits
                lp = torch.log_softmax(logits[:, :-1].float(), -1)
                tgt = batch[:, 1:]
                nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                mm = mask[:, 1:].float()
                mean_nll = (nll * mm).sum(1) / mm.sum(1).clamp(min=1)
                worst = max(worst, float(mean_nll.max().exp()))
        return worst


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--wide-csv", default=C.WIDE_CSV)
    p.add_argument("--detail-csv", default=C.DETAIL_CSV)
    p.add_argument("--doc-col", default="")
    p.add_argument("--max-papers", type=int, default=0)
    p.add_argument("--ppl-model", default="gpt2",
                   help="local path to a small causal LM. HF_HUB_OFFLINE=1 is set on "
                        "your cluster, so this must already be on disk.")
    p.add_argument("--no-ppl", action="store_true", help="regex only")
    p.add_argument("--window", type=int, default=128)
    p.add_argument("--stride", type=int, default=64)
    p.add_argument("--out-dir", default="")
    a = p.parse_args()

    R = C.Report("D3_ppl_regex", a.out_dir)
    df = C.build_eval_frame(a.wide_csv, a.detail_csv, a.doc_col, a.max_papers)

    # ---- regex ---------------------------------------------------------------------
    R("")
    R("[regex] scoring")
    df["regex_score"] = [regex_score(t) for t in df["text"]]
    df["regex_pred"] = (df["regex_score"] > 0).astype(int)

    # ---- perplexity ----------------------------------------------------------------
    df["ppl_score"] = np.nan
    ppl_ok = False
    if not a.no_ppl:
        try:
            eng = PPL(a.ppl_model, a.window, a.stride)
            vals = []
            for i, t in enumerate(df["text"]):
                if i % 200 == 0:
                    R(f"[ppl] {i}/{len(df)}")
                try:
                    vals.append(eng.max_window_ppl(t))
                except Exception:
                    vals.append(float("nan"))
            df["ppl_score"] = vals
            ppl_ok = True
        except Exception as e:
            R(f"[ppl] UNAVAILABLE: {type(e).__name__}: {e}")
            R(f"[ppl] the model could not be loaded. With HF_HUB_OFFLINE=1 it must be on")
            R(f"[ppl] disk already. On a login node with network:")
            R(f"[ppl]     huggingface-cli download gpt2 --local-dir $HF_HOME/gpt2")
            R(f"[ppl] then re-run with --ppl-model $HF_HOME/gpt2")
            R(f"[ppl] Continuing with the REGEX screen only.")

    # ---- thresholds: pick on val, freeze, apply to test ----------------------------
    val = df[df.split == "val"]
    R("")
    R("[thresh] chosen on the validation half, then frozen. A threshold tuned on the")
    R("[thresh] test set is not a result.")
    thr = {}
    for col in (["regex_score"] + (["ppl_score"] if ppl_ok else [])):
        r05, t05 = C.recall_at_fpr(val["label"].values, val[col].values, 0.05)
        r10, t10 = C.recall_at_fpr(val["label"].values, val[col].values, 0.10)
        thr[col] = dict(t05=t05, t10=t10, val_r05=r05, val_r10=r10)
        R(f"  {col:12s} val recall@FPR5% {r05:.3f} (thr {t05:.4g})   "
          f"@FPR10% {r10:.3f} (thr {t10:.4g})")

    test = df[df.split == "test"].copy()

    variants = {"regex_only": ("regex_score", thr["regex_score"]["t05"])}
    if ppl_ok:
        variants["ppl_only"] = ("ppl_score", thr["ppl_score"]["t05"])

    for name, (col, t) in list(variants.items()):
        d = test.copy()
        d["score"] = d[col]
        d["pred"] = (d[col] >= t).astype(int) if t == t else 0
        m = C.evaluate(d, calls_per_paper=0.0)
        R.table({k: v for k, v in m.items() if not k.startswith("cond:")},
                f"D3 / {name}   (threshold {t:.4g}, frozen from val)")
        R.table({k: v for k, v in m.items() if k.startswith("cond:")},
                f"D3 / {name} — per condition")
        R.save_json(m, f"metrics_{name}.json")

    if ppl_ok:
        d = test.copy()
        d["score"] = (d["regex_score"] / max(1e-9, test["regex_score"].max()) +
                      d["ppl_score"].rank(pct=True).fillna(0))
        r05, t = C.recall_at_fpr(val["label"].values,
                                 (val["regex_score"] / max(1e-9, val["regex_score"].max())
                                  + val["ppl_score"].rank(pct=True).fillna(0)).values, 0.05)
        d["pred"] = (d["score"] >= t).astype(int) if t == t else 0
        m = C.evaluate(d, calls_per_paper=0.0)
        R.table({k: v for k, v in m.items() if not k.startswith("cond:")},
                "D3 / regex+ppl combined")
        R.save_json(m, "metrics_combined.json")

    # ---- what fired, and on what ---------------------------------------------------
    R("")
    R("--- which regex patterns fired, by family " + "-" * 48)
    from collections import Counter
    for fam in ("instruction", "rhetorical", "control", "clean"):
        sub = df[df.family == fam]
        if not len(sub):
            continue
        cnt = Counter()
        for t in sub["text"]:
            for h in regex_hits(t):
                cnt[h] += 1
        R(f"  {fam} (n={len(sub)}):")
        for pat, n in cnt.most_common(6):
            R(f"      {n:5d}  {pat}")
        if not cnt:
            R("      (nothing matched)")

    R("")
    R("READ THIS BEFORE THE NEXT BASELINE")
    R("  Compare `recall` on the instruction rows against `recall` on the rhetorical")
    R("  rows. A large gap is not a weakness of this baseline -- it is the finding. It")
    R("  says the cheap detector everyone uses covers one of your two threat families")
    R("  and not the other, and it costs minutes of CPU to establish.")

    R.save(test.drop(columns=["text", "doc_clean"]), "predictions_test.csv")
    R.save(df.drop(columns=["text", "doc_clean"]), "predictions_all.csv")
    R.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())