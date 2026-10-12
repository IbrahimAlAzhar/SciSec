"""
v8_rewards.py — the reward model. Rule-based, decided by gold, never by an LLM.

ONE DEFINITION, THREE CONSUMERS
    rollout scoring   every sampled completion gets a reward and a `usable` flag
    DPO pairs         chosen = highest reward, rejected = lowest, SAME prompt, and only
                      when the gap is at least `min_margin` (a coin-flip pair is noise)
    GRPO advantage    (r - mean(group)) / std(group) over the k samples of one prompt
Weights live in ft_config.RW, so changing one changes all three consistently.

WHY RULE-BASED AND NOT A LEARNED REWARD MODEL
    Everything the reward needs -- the label, the exact gold characters, whether a quote
    is really in the text -- is known by construction. A learned reward model can only
    approximate those numbers, and its errors become the policy's target. A learned RM
    earns its place where gold is ABSENT (real submissions), which is inference-time
    reranking, not training. This file is the training reward.

THE SHAPE OF EACH REWARD, AND THE FAILURE IT IS AIMED AT
    LOCATOR    +coverage of gold inside the window, -false alarms outside it, -fabricated
               quotes. The v4 workers were rewarded for FINDING things; this one is also
               rewarded for finding NOTHING on a clean window.
    EVIDENCE   reviewer_redirection correctness per item + quote grounding. factual_support
               is NOT scored: gold cannot supervise it (see v8_agents docstring).
    VERIFIER   per item: CONFIRMED on gold / DISMISSED off gold; UNCERTAIN earns partial
               credit so honest abstention beats a confident wrong call.
    LEADER     verdict (FP costs fp_cost > fn_cost), verbatim span, on-target span,
    (detect)   calibration.
    LEADER     character F1 against gold, minus COLLATERAL (authors' text deleted) and
    (localize) RESIDUAL (manipulation left in). On a clean paper: +restraint for deleting
               nothing.
    PROVENANCE (GRPO ablation) LEADER-detect reward + span IoU + a TWIN term: a quote that
               also appears verbatim in the benign version of the same paper is the
               authors' own text, whatever the verdict says, and is penalised. This term is
               PER SAMPLE, so it survives group normalisation -- a pair-level constant
               would cancel out of every GRPO advantage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import ft_config as C
import v8_agents as A
import v8_tools as T

RW = C.RW
CLIP = 4.0


@dataclass
class Score:
    reward: float = 0.0
    correct: Optional[bool] = None       # the task's primary correctness, when defined
    parsed: bool = False
    components: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    # set when the answer is right for the wrong reason (e.g. correct verdict, quoting the
    # authors' own prose): fine to reward partially, never to IMITATE
    blocked: bool = False

    @property
    def usable(self) -> bool:
        """Good enough to be an SFT target or a DPO `chosen`."""
        return (self.parsed and self.correct is not False and self.reward > 0.0
                and not self.blocked)

    def as_dict(self) -> Dict:
        return {"reward": round(self.reward, 4), "correct": self.correct,
                "parsed": self.parsed, "usable": self.usable,
                "components": {k: round(v, 4) for k, v in self.components.items()},
                "notes": list(self.notes)}


def _finish(sc: Score) -> Score:
    sc.reward = max(-CLIP, min(CLIP, sum(sc.components.values())))
    return sc


def _conf(obj: Dict, key: str = "confidence") -> float:
    try:
        v = float(obj.get(key, 0.5))
        return min(1.0, max(0.0, v))
    except (TypeError, ValueError):
        return 0.5


# ======================================================================================
# LOCATOR
# ======================================================================================
def score_locator(raw: str, *, doc: str, window: T.Span, gold: Sequence[T.Span],
                  label: int) -> Score:
    sc = Score()
    obj = A.extract_json(raw)
    if not isinstance(obj, dict):
        sc.components["schema"] = -1.0
        sc.notes.append("no JSON")
        sc.correct = False
        return _finish(sc)
    sc.parsed = True
    cands, claims, conf = A.parse_locator(obj)
    sc.components["schema"] = RW.schema if "candidates" in obj else -RW.schema
    wtext = doc[window[0]:window[1]]
    loc = T.Located(doc)
    spans, n_fab = [], 0
    for c in cands:
        sp = str(c.get("span") or "")
        if T.verbatim(sp, wtext) is not True:
            n_fab += 1
            continue
        hit = loc.find(sp, 0)
        if hit:
            spans.append(hit)
    gw = T.merge([(max(a, window[0]), min(b, window[1])) for a, b in gold
                  if min(b, window[1]) > max(a, window[0])]) if label == 1 else []
    if gw:
        prf = T.span_prf(spans, gw)
        hit = any(T.overlap_frac(s, gw) >= 0.3 for s in spans)
        sc.correct = hit
        sc.components["detect"] = 1.0 if hit else -RW.fn_cost
        sc.components["coverage"] = 0.8 * prf["recall"]
        n_false = sum(1 for s in spans if T.overlap_frac(s, gw) < 0.3)
        sc.components["false_alarm"] = -0.3 * n_false
    else:
        sc.correct = not spans
        sc.components["detect"] = 1.0 if not spans else -RW.fp_cost
        sc.components["false_alarm"] = -0.2 * max(0, len(spans) - 1)
    if n_fab:
        sc.components["fabricated"] = -RW.fabricated * n_fab
        sc.notes.append(f"{n_fab} candidate(s) not verbatim in the window")
    elif spans:
        sc.components["grounded"] = RW.grounded * 0.5
    if len(claims) > 4:
        sc.components["claim_spam"] = -0.1 * (len(claims) - 4)
    sc.components["calibration"] = -RW.calibration * (conf - float(bool(sc.correct))) ** 2
    return _finish(sc)


# ======================================================================================
# EVIDENCE and VERIFIER (one call covers every item of a document)
# ======================================================================================
def score_evidence(raw: str, *, items: Sequence[A.Item], gold: Sequence[T.Span],
                   family: str, retrieved: Dict[int, List[Dict]]) -> Score:
    sc = Score()
    obj = A.extract_json(raw)
    if not isinstance(obj, dict) or not items:
        sc.components["schema"] = -1.0 if items else 0.0
        sc.correct = False if items else None
        sc.parsed = isinstance(obj, dict)
        return _finish(sc)
    sc.parsed = True
    out = A.parse_evidence(obj)
    redir_fam = family in ("instruction", "rhetorical")
    per, n_fab, n_missing = [], 0, 0
    for it in items:
        o = out.get(it.id)
        if o is None:
            n_missing += 1
            per.append(-0.5)
            continue
        want = bool(redir_fam and A.item_is_gold(it, gold))
        got = bool(o.get("reviewer_redirection"))
        if got == want:
            per.append(1.0)
        else:
            per.append(-RW.fp_cost if (got and not want) else -RW.fn_cost)
        q = str(o.get("evidence_quote") or "")
        if q.strip():
            pool = " ".join(e["text"] for e in retrieved.get(it.id) or [])
            if T.verbatim(q, pool, min_chars=8) is not True:
                n_fab += 1
    sc.components["redirection"] = sum(per) / len(per)
    sc.correct = all(p > 0 for p in per)
    if n_fab:
        sc.components["fabricated"] = -RW.fabricated * 0.5 * n_fab
    if n_missing:
        sc.components["schema"] = -RW.schema * n_missing
        sc.notes.append(f"{n_missing} item(s) missing")
    else:
        sc.components["schema"] = RW.schema
    return _finish(sc)


def score_verifier(raw: str, *, items: Sequence[A.Item], gold: Sequence[T.Span]) -> Score:
    sc = Score()
    obj = A.extract_json(raw)
    if not isinstance(obj, dict) or not items:
        sc.components["schema"] = -1.0 if items else 0.0
        sc.correct = False if items else None
        sc.parsed = isinstance(obj, dict)
        return _finish(sc)
    sc.parsed = True
    out = A.parse_verifier(obj)
    per, n_missing = [], 0
    for it in items:
        o = out.get(it.id)
        if o is None:
            n_missing += 1
            per.append(-0.5)
            continue
        on = A.item_is_gold(it, gold)
        d = o["decision"]
        if d == "UNCERTAIN":
            per.append(RW.uncertain_credit)
        elif (d == "CONFIRMED") == on:
            per.append(1.0)
        else:
            per.append(-RW.fp_cost if d == "CONFIRMED" else -RW.fn_cost)
    sc.components["decisions"] = sum(per) / len(per)
    sc.correct = all(p > 0 for p in per)
    sc.components["schema"] = RW.schema if not n_missing else -RW.schema * n_missing
    return _finish(sc)


# ======================================================================================
# LEADER — detect, localize, provenance
# ======================================================================================
def score_leader(raw: str, *, doc: str, gold: Sequence[T.Span], label: int) -> Score:
    sc = Score()
    obj = A.extract_json(raw)
    p = A.parse_leader(obj)
    if not p["parsed"]:
        sc.components["schema"] = -1.0
        sc.correct = False
        sc.notes.append("verdict missing or not ATTACKED/CLEAN")
        return _finish(sc)
    sc.parsed = True
    sc.components["schema"] = RW.schema
    said = p["verdict"] == "ATTACKED"
    sc.correct = said == (label == 1)
    if sc.correct:
        sc.components["verdict"] = 1.0
    else:
        sc.components["verdict"] = -RW.fp_cost if said else -RW.fn_cost
        sc.notes.append("FALSE POSITIVE" if said else "MISS")
    if said:
        g = T.verbatim(p["span"], doc)
        if g is not True:
            # At inference an ATTACKED verdict without a real quote is not acted on, so on
            # an attacked paper it is a MISS -- plus a hallucination. Scoring it as a
            # correct verdict would rank inventing evidence above honestly missing.
            if label == 1:
                sc.components["verdict"] = -RW.fn_cost
                sc.correct = False
            sc.components["grounded"] = -RW.fabricated
            sc.notes.append("ATTACKED without a verbatim span")
        else:
            sc.components["grounded"] = RW.grounded
            if label == 1 and gold:
                hit = T.Located(doc).find(p["span"])
                on = hit is not None and T.overlap_frac(hit, gold) >= 0.3
                sc.components["on_target"] = RW.on_target if on else -RW.on_target
                if not on:
                    sc.blocked = True
                    sc.notes.append("right verdict, quoted the authors' own text")
    sc.components["calibration"] = -RW.calibration * (p["confidence"] -
                                                      float(bool(sc.correct))) ** 2
    return _finish(sc)


def score_localize(raw: str, *, doc: str, gold: Sequence[T.Span], label: int) -> Score:
    sc = Score()
    obj = A.extract_json(raw)
    p = A.parse_localize(obj)
    if not p["parsed"]:
        sc.components["schema"] = -1.0
        sc.correct = False
        return _finish(sc)
    sc.parsed = True
    sc.components["schema"] = RW.schema
    spans, n_fab = T.spans_from_quotes(doc, p["delete"])
    if n_fab:
        sc.components["fabricated"] = -RW.fabricated * 0.5 * n_fab
        sc.notes.append(f"{n_fab} deletion(s) not verbatim")
    if label == 1 and gold:
        prf = T.span_prf(spans, gold)
        sc.components["span_f1"] = RW.span_f1 * prf["f1"]
        sc.components["collateral"] = -RW.collateral * prf["collateral"]
        sc.components["residual"] = -RW.residual * prf["residual"]
        sc.correct = prf["f1"] >= 0.5 and prf["collateral"] <= 0.3
    else:
        n = T.total(spans)
        sc.correct = n == 0
        sc.components["restraint"] = (RW.clean_restraint if n == 0 else
                                      -RW.collateral * min(2.0, 0.5 + n / 1000.0))
    return _finish(sc)


def score_provenance(raw: str, *, doc: str, gold: Sequence[T.Span], label: int,
                     twin_doc: str = "") -> Score:
    """LEADER-detect reward + isolation (IoU) + a per-sample twin-provenance term."""
    sc = score_leader(raw, doc=doc, gold=gold, label=label)
    if not sc.parsed:
        return sc
    p = A.parse_leader(A.extract_json(raw))
    if p["verdict"] == "ATTACKED" and T.verbatim(p["span"], doc) is True:
        hit = T.Located(doc).find(p["span"])
        if hit is not None and label == 1 and gold:
            sc.components["isolation"] = 0.5 * T.span_prf([hit], gold)["iou"]
        if twin_doc:
            in_twin = T.verbatim(p["span"], twin_doc) is True
            # A quote that also exists in the other version of the SAME paper is text the
            # authors wrote in both -- it cannot be the provenance of the manipulation.
            sc.components["twin"] = -RW.twin if in_twin else (RW.twin if label == 1 else 0)
            if in_twin:
                sc.notes.append("quoted text exists in the benign twin")
    return _finish(sc)


# ======================================================================================
# GRPO group advantage
# ======================================================================================
def group_advantages(rewards: Sequence[float], eps: float = 1e-6) -> Optional[List[float]]:
    """None when the group has no reward variance: no signal, dropped (not zero-padded)."""
    n = len(rewards)
    if n < 2:
        return None
    mu = sum(rewards) / n
    var = sum((r - mu) ** 2 for r in rewards) / (n - 1)
    sd = var ** 0.5
    if sd < 1e-4:
        return None
    return [(r - mu) / (sd + eps) for r in rewards]
