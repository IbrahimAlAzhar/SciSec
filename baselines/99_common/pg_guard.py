"""
pg_guard.py — PIGuard as a DOCUMENT-level defence: scoring, windowing, attack localisation,
and removal.

WHAT IS PIGUARD, UNCHANGED
=======================================================================================
The classifier is exactly the released one (Li et al., ACL 2025):
    DeBERTa-v3-base -> [CLS] hidden state -> Linear(768, 2);  label 0 = benign, 1 = injection
Two ways to load it, both producing the same logits:
    kind="hf"   the released checkpoint leolee99/PIGuard (AutoModelForSequenceClassification,
                trust_remote_code — the repo's modeling_piguard.py), used in eval_hf.py
    kind="pth"  a state_dict written by PIGuard-main/train.py (class PIGuard in PIGuard.py),
                i.e. a checkpoint retrained with pg_train.py
The decision rule is PIGuard's own: argmax of the two logits, i.e. P(injection) >= 0.5.

WHAT HAD TO BE ADDED FOR MANUSCRIPTS, AND WHY
=======================================================================================
PIGuard classifies a PROMPT of at most 512 tokens (max_length=512 in params.py; eval_hf.py
truncates). A manuscript in this corpus is up to 30,000 characters (~7,500 tokens) and the
attack sits near the END (slot limitations_end). Used literally, PIGuard would read the first
512 tokens (the abstract) and could never see the attack. So the document is covered with
overlapping windows of PIGuard's own training length, and the document is flagged iff ANY
window is classified as injection:

    score(doc) = max_w P(injection | window w)      pred(doc) = score(doc) >= 0.5

That is the only adaptation of the detector. `window_mode="truncate"` keeps the literal
as-released behaviour (first `trunc_len` tokens only) for anyone who wants that row too.

LOCALISATION (the "where is the attack?" question)
=======================================================================================
PIGuard gives one number per window, not per token. The attack is located by asking PIGuard
itself which sentences make a flagged window fire:

  1. greedy occlusion: while the window still fires, remove the sentence whose deletion lowers
     P(injection | window) the most (at most `max_remove` sentences per window). This yields
     a MINIMAL set: the fewest sentences whose removal makes PIGuard stop firing.
  2. repeat on the sanitised document (up to `max_rounds`), because deleting text shifts the
     windows and can expose a remainder that straddled two of them.
  (optional, --standalone: first drop every sentence that PIGuard flags on its own. Off by
   default — on manuscripts it removes honest limitation sentences whenever the classifier
   over-defends, which is exactly the failure PIGuard's paper is about.)

The removed sentences ARE the localisation. Nothing else is consulted — no LLM, no gold — so the
localisation is PIGuard's, and it is scored against the gold diff (attacked vs its own clean
document) only afterwards.
"""

from __future__ import annotations

import bisect
import difflib
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Interval = Tuple[int, int]


# ======================================================================================
# the model — PIGuard-main/PIGuard.py, ported without changing the computation
# ======================================================================================
def _torch():
    import torch
    return torch


def build_piguard_module(model_name: str, num_labels: int, device):
    """
    PIGuard-main/PIGuard.py::PIGuard, verbatim in structure:
        self.deberta = AutoModel(...);  self.classifier = nn.Linear(hidden, num_labels)
        logits = classifier(last_hidden_state[:, 0, :]);  loss = CrossEntropyLoss
    The one change: output_attentions is left False. The original sets it True in the config,
    which returns every attention map and changes nothing about the logits or the gradients.
    """
    import torch.nn as nn
    from transformers import AutoConfig, AutoModel

    class PIGuard(nn.Module):
        def __init__(self):
            super().__init__()
            self.device = device
            self.config = AutoConfig.from_pretrained(model_name)
            self.deberta = AutoModel.from_pretrained(model_name, config=self.config).to(device)
            self.classifier = nn.Linear(self.deberta.config.hidden_size, num_labels).to(device)
            self.loss_fct = nn.CrossEntropyLoss().to(device)

        def forward(self, input_ids, attention_mask, labels=None, mode="train"):
            outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs["last_hidden_state"][:, 0, :]
            logits = self.classifier(pooled_output)
            loss = 0
            if labels is not None:
                loss = self.loss_fct(logits.view(-1, self.classifier.out_features),
                                     labels.view(-1))
            return logits, loss

    return PIGuard()


class GuardScorer:
    """P(injection) for a batch of texts, from either checkpoint kind."""

    def __init__(self, path: str, *, kind: str = "auto", backbone: str = "",
                 tokenizer_path: str = "", device: str = "", max_length: int = 512,
                 batch_size: int = 32, fp16: bool = False):
        torch = _torch()
        from transformers import AutoTokenizer
        self.path = path
        if kind == "auto":
            kind = "pth" if os.path.isfile(path) else "hf"
        self.kind = kind
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.n_forward = 0

        tok_candidates = [tokenizer_path] if tokenizer_path else []
        tok_candidates += [path] if kind == "hf" else [backbone]
        tok_candidates += [os.environ.get("PIGUARD_PATH", "")]
        self.tokenizer = None
        err = None
        for tp in [t for t in tok_candidates if t]:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tp)
                self.tokenizer_path = tp
                break
            except Exception as e:           # a slow->fast conversion can fail offline
                err = e
        if self.tokenizer is None:
            raise RuntimeError(f"could not load a tokenizer from {tok_candidates}: {err}")
        # the document is tokenised in one piece for windowing; silence the length warning
        self.tokenizer.model_max_length = 10 ** 9

        if kind == "hf":
            from transformers import AutoModelForSequenceClassification
            self.model = AutoModelForSequenceClassification.from_pretrained(
                path, trust_remote_code=True).to(self.device)
            l2i = {str(k).lower(): int(v) for k, v in
                   (getattr(self.model.config, "label2id", None) or {}).items()}
            self.inj_index = l2i.get("injection", 1)
            self._hf = True
        elif kind == "pth":
            if not backbone:
                raise ValueError("kind='pth' needs --backbone (microsoft/deberta-v3-base)")
            self.model = build_piguard_module(backbone, 2, self.device)
            sd = torch.load(path, map_location=self.device)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            res = self.model.load_state_dict(sd, strict=False)
            missing = [k for k in res.missing_keys if not k.startswith("loss_fct")]
            if any(k.startswith("classifier") for k in missing) or len(missing) > 5:
                raise RuntimeError(f"{path} does not look like a PIGuard state_dict; "
                                   f"missing keys: {missing[:10]}")
            print(f"[guard] loaded {path}: {len(missing)} missing, "
                  f"{len(res.unexpected_keys)} unexpected key(s)")
            self.inj_index = 1
            self._hf = False
        else:
            raise ValueError(f"unknown kind {kind!r}")
        self.model.eval()
        self.fp16 = bool(fp16 and self.device.type == "cuda")
        self.is_fast = bool(getattr(self.tokenizer, "is_fast", False))
        self.cls_id = self.tokenizer.cls_token_id
        self.sep_id = self.tokenizer.sep_token_id
        print(f"[guard] kind={self.kind} path={path} device={self.device} "
              f"tokenizer={self.tokenizer_path} fast={self.is_fast} "
              f"max_length={self.max_length} inj_index={self.inj_index}")

    # ------------------------------------------------------------------ scoring
    def _logits(self, input_ids, attention_mask):
        if self._hf:
            return self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        return self.model(input_ids, attention_mask, labels=None)[0]

    def score_texts(self, texts: Sequence[str]) -> np.ndarray:
        """P(injection) per text. Tokenised as PIGuard does: truncation at max_length."""
        torch = _torch()
        texts = list(texts)
        out = np.zeros(len(texts), dtype=np.float64)
        if not texts:
            return out
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        with torch.no_grad():
            for b in range(0, len(order), self.batch_size):
                idx = order[b: b + self.batch_size]
                enc = self.tokenizer([texts[i] for i in idx], truncation=True,
                                     max_length=self.max_length, padding=True,
                                     return_tensors="pt")
                ii = enc["input_ids"].to(self.device)
                am = enc["attention_mask"].to(self.device)
                if self.fp16:
                    with torch.autocast("cuda", dtype=torch.float16):
                        lg = self._logits(ii, am)
                else:
                    lg = self._logits(ii, am)
                p = torch.softmax(lg.float(), dim=-1)[:, self.inj_index].cpu().numpy()
                out[idx] = p
                self.n_forward += len(idx)
        return out

    def score_ids(self, id_lists: Sequence[Sequence[int]]) -> np.ndarray:
        """P(injection) for pre-tokenised inputs, [CLS] + ids + [SEP] (MOF token recheck)."""
        torch = _torch()
        out = np.zeros(len(id_lists), dtype=np.float64)
        pad = self.tokenizer.pad_token_id or 0
        with torch.no_grad():
            for b in range(0, len(id_lists), self.batch_size):
                chunk = [[self.cls_id] + list(x)[: self.max_length - 2] + [self.sep_id]
                         for x in id_lists[b: b + self.batch_size]]
                L = max(len(x) for x in chunk)
                ii = torch.tensor([x + [pad] * (L - len(x)) for x in chunk],
                                  device=self.device)
                am = torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x in chunk],
                                  device=self.device)
                lg = self._logits(ii, am)
                out[b: b + len(chunk)] = torch.softmax(lg.float(), -1)[:, self.inj_index].cpu().numpy()
                self.n_forward += len(chunk)
        return out

    def offsets(self, text: str) -> List[Interval]:
        """Character offsets of every token of `text` (no special tokens)."""
        if not text:
            return []
        if self.is_fast:
            enc = self.tokenizer(text, add_special_tokens=False,
                                 return_offsets_mapping=True, truncation=False)
            return [tuple(o) for o in enc["offset_mapping"]]
        # slow tokenizer: approximate one token per 4 characters
        return [(i, min(i + 4, len(text))) for i in range(0, len(text), 4)]


# ======================================================================================
# windows
# ======================================================================================
def token_windows(offs: Sequence[Interval], text_len: int, max_length: int = 512,
                  stride: int = 256) -> List[Interval]:
    """
    Character spans of overlapping token windows. Each window holds max_length-2 tokens
    (room for [CLS] and [SEP]), consecutive windows start `stride` tokens apart, and the last
    window is aligned to the end of the text so the tail — where the attack usually is — is
    always seen in a full window.
    """
    n = len(offs)
    if n == 0:
        return []
    w = max(1, int(max_length) - 2)
    s = max(1, int(stride))
    if n <= w:
        return [(0, text_len)]
    starts = list(range(0, n - w + 1, s))
    if starts[-1] + w < n:
        starts.append(n - w)
    spans = []
    for st in starts:
        en = min(n, st + w)
        c0 = offs[st][0]
        c1 = offs[en - 1][1] if en < n else text_len
        spans.append((c0, max(c1, c0)))
    return spans


def truncate_window(offs: Sequence[Interval], text_len: int, trunc_len: int) -> List[Interval]:
    """eval_hf.py behaviour: only the first `trunc_len` tokens are ever read."""
    if not offs:
        return []
    k = min(len(offs), max(1, trunc_len - 2))
    return [(0, offs[k - 1][1] if k < len(offs) else text_len)]


# ======================================================================================
# sentences — a partition of the text
# ======================================================================================
_SENT_END = re.compile(r"(?<=[.!?])[\"'\)\]]*\s+(?=[\"'\(\[]?[A-Z0-9])|\n\s*\n")


def sentence_bounds(text: str, max_chars: int = 500) -> List[int]:
    """
    Boundaries b_0=0 < b_1 < ... < b_n=len(text); sentence i = text[b_i:b_{i+1}] INCLUDING its
    trailing whitespace, so the sentences partition the text exactly and deleting any subset
    leaves well-formed prose. Over-long "sentences" (PDF tables, equations) are cut at
    whitespace every ~max_chars so one bad split cannot remove a page.
    """
    n = len(text)
    if n == 0:
        return [0]
    cuts = {0, n}
    for m in _SENT_END.finditer(text):
        cuts.add(m.end())
    b = sorted(c for c in cuts if 0 <= c <= n)
    out = [0]
    for i in range(1, len(b)):
        a, e = out[-1], b[i]
        while e - a > max_chars:
            cut = text.rfind(" ", a + max_chars // 2, a + max_chars)
            if cut <= a:
                cut = a + max_chars
            out.append(cut + 1 if text[cut:cut + 1] == " " else cut)
            a = out[-1]
        if e > out[-1]:
            out.append(e)
    return out


def render(text: str, bounds: Sequence[int], keep: Sequence[bool]
           ) -> Tuple[str, List[Tuple[int, int, int]]]:
    """Kept sentences joined; plus (start, end, sentence_index) in RENDERED coordinates."""
    parts, smap, pos = [], [], 0
    for i in range(len(bounds) - 1):
        if not keep[i]:
            continue
        seg = text[bounds[i]:bounds[i + 1]]
        if not seg:
            continue
        parts.append(seg)
        smap.append((pos, pos + len(seg), i))
        pos += len(seg)
    return "".join(parts), smap


def _cut(text: str, c0: int, c1: int, intervals: Sequence[Interval]) -> str:
    segs, pos = [], c0
    for a, b in sorted((max(a, c0), min(b, c1)) for a, b in intervals):
        if b <= a:
            continue
        if a > pos:
            segs.append(text[pos:a])
        pos = max(pos, b)
    if pos < c1:
        segs.append(text[pos:c1])
    return "".join(segs)


def merge_intervals(iv: Sequence[Interval]) -> List[Interval]:
    out: List[List[int]] = []
    for a, b in sorted((int(a), int(b)) for a, b in iv if b > a):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


# ======================================================================================
# the document analysis
# ======================================================================================
@dataclass
class GuardConfig:
    threshold: float = 0.5          # PIGuard's argmax rule
    max_length: int = 512           # PIGuard training length (params.py)
    stride: int = 256
    window_mode: str = "slide"      # slide | truncate
    trunc_len: int = 2048           # eval_hf.py sets model_max_length=2048
    scope: str = "full"             # full | region
    region_chars: int = 6000        # the region the agents read (ft_roles.region)
    localize: bool = True
    standalone: bool = False        # pre-remove sentences flagged on their own (see header)
    max_remove: int = 10            # sentences removed per flagged window, at most
    max_rounds: int = 3
    max_cands: int = 48
    max_sent_chars: int = 500


def _windows(guard: GuardScorer, text: str, cfg: GuardConfig) -> List[Interval]:
    offs = guard.offsets(text)
    if cfg.window_mode == "truncate":
        return truncate_window(offs, len(text), cfg.trunc_len)
    return token_windows(offs, len(text), cfg.max_length, cfg.stride)


def analyze_document(guard: GuardScorer, document: str, cfg: GuardConfig) -> Dict:
    """
    Detection + (optional) localisation + removal for one manuscript.

    Returns
      score, pred           max window P(injection) on the ORIGINAL document, and >= threshold
      n_windows, n_flagged  windows scanned / windows at or above threshold
      top_windows           up to 3 (start, end, p) in document coordinates
      removed_spans         merged (start, end) of the removed sentences, document coordinates
      sanitized             the document with those sentences deleted (None if nothing removed)
      residual_score        max window P(injection) on the sanitised document
      resolved              residual_score < threshold
      passes                guard forward passes spent on this document
    """
    doc = document or ""
    f0 = guard.n_forward
    if cfg.scope == "region" and len(doc) > cfg.region_chars:
        base = len(doc) - cfg.region_chars
    else:
        base = 0
    text = doc[base:]

    wins = _windows(guard, text, cfg)
    ps = guard.score_texts([text[a:b] for a, b in wins]) if wins else np.zeros(0)
    score = float(ps.max()) if len(ps) else 0.0
    flagged0 = [(a, b, float(p)) for (a, b), p in zip(wins, ps) if p >= cfg.threshold]
    top = sorted(((a + base, b + base, float(p)) for (a, b), p in zip(wins, ps)),
                 key=lambda t: -t[2])[:3]
    res = {"score": score, "pred": int(score >= cfg.threshold), "n_windows": len(wins),
           "n_flagged": len(flagged0), "top_windows": top,
           "flagged_spans": merge_intervals([(a + base, b + base) for a, b, _ in flagged0]),
           "removed_spans": [],
           "removed_chars": 0, "n_removed_sents": 0, "sanitized": None,
           "residual_score": score, "resolved": int(score < cfg.threshold),
           "rounds": 0, "removed_text": ""}
    if not flagged0 or not cfg.localize:
        res["passes"] = guard.n_forward - f0
        return res

    bounds = sentence_bounds(text, cfg.max_sent_chars)
    nsent = len(bounds) - 1
    keep = [True] * nsent
    thr = cfg.threshold
    flagged = flagged0
    rendered, smap = text, None
    rounds = 0
    for rnd in range(cfg.max_rounds):
        rendered, smap = render(text, bounds, keep)
        if rnd > 0:
            wins = _windows(guard, rendered, cfg)
            ps = guard.score_texts([rendered[a:b] for a, b in wins]) if wins else np.zeros(0)
            flagged = [(a, b, float(p)) for (a, b), p in zip(wins, ps) if p >= thr]
        if not flagged:
            break
        rounds += 1
        starts = [s0 for s0, _, _ in smap]
        sint = {si: (s0, s1) for s0, s1, si in smap}
        round_removed: List[int] = []
        for c0, c1, pw in sorted(flagged, key=lambda t: -t[2]):
            lo = max(0, bisect.bisect_right(starts, c0) - 1)
            in_win = []
            for s0, s1, si in smap[lo:]:
                if s0 >= c1:
                    break
                if s1 > c0:
                    in_win.append(si)
            in_win = [si for si in in_win
                      if si not in round_removed and text[bounds[si]:bounds[si + 1]].strip()]
            already = [sint[si] for si in round_removed
                       if sint[si][0] < c1 and sint[si][1] > c0]
            cands = in_win[: cfg.max_cands]
            if not cands:
                continue
            chosen: List[int] = []
            cur = float(guard.score_texts([_cut(rendered, c0, c1, already)])[0]) \
                if already else pw
            if cfg.standalone:
                sp = guard.score_texts([rendered[sint[si][0]:sint[si][1]] for si in cands])
                chosen = [si for si, p in sorted(zip(cands, sp), key=lambda t: -t[1])
                          if p >= thr][: cfg.max_remove]
                if chosen:
                    cur = float(guard.score_texts(
                        [_cut(rendered, c0, c1, already + [sint[s] for s in chosen])])[0])
            # greedy occlusion until the window stops firing
            while cur >= thr and len(chosen) < cfg.max_remove:
                rem = [si for si in cands if si not in chosen]
                if not rem:
                    break
                texts = [_cut(rendered, c0, c1,
                              already + [sint[s] for s in chosen] + [sint[si]])
                         for si in rem]
                pr = guard.score_texts(texts)
                j = int(np.argmin(pr))
                chosen.append(rem[j])
                cur = float(pr[j])
            round_removed.extend(chosen)
        if not round_removed:
            break
        for si in round_removed:
            keep[si] = False

    rendered, _ = render(text, bounds, keep)
    removed = merge_intervals([(bounds[i] + base, bounds[i + 1] + base)
                               for i in range(nsent) if not keep[i]])
    if removed:
        wins = _windows(guard, rendered, cfg)
        pr = guard.score_texts([rendered[a:b] for a, b in wins]) if wins else np.zeros(0)
        resid = float(pr.max()) if len(pr) else 0.0
        res.update(removed_spans=removed,
                   removed_chars=sum(b - a for a, b in removed),
                   n_removed_sents=sum(1 for k in keep if not k),
                   sanitized=doc[:base] + rendered, residual_score=resid,
                   resolved=int(resid < thr),
                   removed_text=" ... ".join(" ".join(doc[a:b].split())
                                             for a, b in removed)[:600])
    res["rounds"] = rounds
    res["passes"] = guard.n_forward - f0
    return res


# ======================================================================================
# gold: where the attack actually is, from the attacked/clean pair
# ======================================================================================
def gold_region(attacked: str, clean: str, refine_factor: float = 3.0) -> List[Interval]:
    """
    Character intervals of `attacked` that are not in its clean partner.

    Common prefix/suffix trimming finds the edited region in O(n); for a pure insertion that
    is exactly the insert. If the region is much longer than the length difference (e.g. a
    rewrite plus a truncation shift), it is refined with a word-level diff so the gold span is
    not inflated by text the attack never touched.
    """
    a, c = attacked or "", clean or ""
    if a == c:
        return []
    p = len(os.path.commonprefix([a, c]))
    s = len(os.path.commonprefix([a[p:][::-1], c[p:][::-1]]))
    a1, c1 = len(a) - s, len(c) - s
    if a1 <= p:
        return []
    extra = abs(len(a) - len(c))
    if (a1 - p) <= max(refine_factor * extra, extra + 2000):
        return _trim_ws(a, [(p, a1)])
    wa = [(m.start() + p, m.end() + p, m.group()) for m in re.finditer(r"\S+", a[p:a1])]
    wc = [m.group() for m in re.finditer(r"\S+", c[p:c1])]
    sm = difflib.SequenceMatcher(None, [w for _, _, w in wa], wc, autojunk=False)
    out = []
    for tag, i1, i2, _j1, _j2 in sm.get_opcodes():
        if tag in ("replace", "delete") and i2 > i1:
            out.append((wa[i1][0], wa[i2 - 1][1]))
    return _trim_ws(a, merge_intervals(out))


def _trim_ws(text: str, iv: Sequence[Interval]) -> List[Interval]:
    out = []
    for a, b in iv:
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        if b > a:
            out.append((a, b))
    return out


def overlap_stats(pred: Sequence[Interval], gold: Sequence[Interval]) -> Dict:
    """Character-level precision / recall / IoU of the removed text against the gold region."""
    P, G = merge_intervals(pred), merge_intervals(gold)
    pc = sum(b - a for a, b in P)
    gc = sum(b - a for a, b in G)
    ov, i, j = 0, 0, 0
    while i < len(P) and j < len(G):
        a = max(P[i][0], G[j][0])
        b = min(P[i][1], G[j][1])
        if b > a:
            ov += b - a
        if P[i][1] < G[j][1]:
            i += 1
        else:
            j += 1
    return {"pred_chars": pc, "gold_chars": gc, "overlap_chars": ov,
            "precision": (ov / pc) if pc else None,
            "recall": (ov / gc) if gc else None,
            "iou": (ov / (pc + gc - ov)) if (pc + gc - ov) else None,
            "hit": int(ov > 0)}
