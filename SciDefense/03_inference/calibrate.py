"""
calibrate.py — V3: set the escalation band from DATA, at a chosen false-positive budget.

WHY V2's BAND WAS WRONG, IN ITS OWN NUMBERS
=======================================================================================
V2 shipped escalate_low = -0.15, chosen from hand-written fixtures where a clean document
scored -0.92. On the real corpus clean documents evidently score much higher, because the
insertion arm escalated 81.1% of everything and the agents then said "attacked" to all of
it: FPR 1.000, accuracy 0.500, worse than tier 0 alone.

A constant taken from a fixture is a guess about a distribution. This module measures the
distribution instead.

THE RULE
=======================================================================================
    escalate_low  = the q-th percentile of the BENIGN score distribution
    escalate_high = the percentile of the benign distribution at (1 - fpr_budget)

Read that second line carefully, because it is the part that fixes the table: the HIGH
gate is set so that at most `fpr_budget` of benign documents can be auto-flagged by tier 0.
FPR is then bounded BY CONSTRUCTION rather than hoped for. With fpr_budget = 0.02 the
tier-0 false-positive rate cannot exceed 2% no matter what the attacks look like.

`escalate_low` controls COST, not correctness: everything below it is settled clean for
free, so a lower percentile means fewer agent calls and a slightly higher chance of missing
a quiet attack. The default (60th percentile of benign) settles the clear majority of clean
papers at zero cost while leaving the ambiguous tail to the verifier.

LEAKAGE
=======================================================================================
Calibration uses ONLY the benign scores of INDEX papers — never the eval papers, never any
attacked document. Fitting the threshold on the attacked side, or on the papers being
scored, would be choosing the operating point that flatters the test set, and the reported
accuracy would not survive a reviewer asking how the threshold was picked.

WHAT TO REPORT
=======================================================================================
Print the fitted thresholds and the benign quantiles in the paper. "We set the high gate at
the 98th percentile of benign scores on held-out index papers" is a defensible sentence;
"we used -0.15" is not.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence


def _percentile(xs: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile; no numpy dependency for four numbers."""
    v = sorted(float(x) for x in xs if x is not None)
    if not v:
        return 0.0
    if len(v) == 1:
        return v[0]
    k = (len(v) - 1) * max(0.0, min(1.0, q))
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


@dataclass
class Calibration:
    threat: str = "insertion"
    n_benign: int = 0
    low: float = 0.0
    high: float = 0.0
    fpr_budget: float = 0.02
    low_q: float = 0.60
    benign_min: float = 0.0
    benign_p50: float = 0.0
    benign_p90: float = 0.0
    benign_p98: float = 0.0
    benign_max: float = 0.0
    note: str = ""

    def as_dict(self) -> Dict:
        return asdict(self)

    def report(self) -> str:
        return (
            f"[calibrate:{self.threat}] n_benign={self.n_benign}  "
            f"benign quantiles: min={self.benign_min:+.3f} p50={self.benign_p50:+.3f} "
            f"p90={self.benign_p90:+.3f} p98={self.benign_p98:+.3f} "
            f"max={self.benign_max:+.3f}\n"
            f"[calibrate:{self.threat}] escalate_low={self.low:+.3f} "
            f"(q={self.low_q:.2f} of benign — everything below is settled clean for free)\n"
            f"[calibrate:{self.threat}] escalate_high={self.high:+.3f} "
            f"(the {100*(1-self.fpr_budget):.0f}th benign percentile, so tier-0 FPR is "
            f"bounded at {100*self.fpr_budget:.0f}% BY CONSTRUCTION)\n"
            f"[calibrate:{self.threat}] {self.note}"
        )


def fit(benign_scores: Sequence[float], *, threat: str = "insertion",
        fpr_budget: float = 0.02, low_q: float = 0.60,
        min_gap: float = 0.10) -> Calibration:
    """
    Fit (low, high) from benign scores alone.

    `min_gap` guards a degenerate corpus: if the benign distribution is so wide that the
    computed high gate lands at or below the low gate, the band would be empty or inverted
    and every document would be auto-classified. Widening the high gate in that case fails
    SAFE — more documents reach the verifier, none are auto-flagged.
    """
    xs = [float(s) for s in benign_scores if s is not None]
    c = Calibration(threat=threat, n_benign=len(xs), fpr_budget=fpr_budget, low_q=low_q)
    if not xs:
        c.note = ("NO BENIGN SCORES — thresholds left at their defaults. This is not a "
                  "calibrated run; say so if you report it.")
        return c

    c.benign_min = min(xs)
    c.benign_p50 = _percentile(xs, 0.50)
    c.benign_p90 = _percentile(xs, 0.90)
    c.benign_p98 = _percentile(xs, 0.98)
    c.benign_max = max(xs)

    c.low = _percentile(xs, low_q)
    c.high = _percentile(xs, 1.0 - fpr_budget)

    if c.high <= c.low + min_gap:
        c.high = c.low + min_gap
        c.note = (f"benign spread is wide, so the high gate was pushed to low+{min_gap} "
                  f"to keep the band non-empty. More documents reach the verifier; none "
                  f"are auto-flagged. This fails safe.")
    else:
        n_above = sum(1 for x in xs if x >= c.high)
        c.note = (f"{n_above}/{len(xs)} benign index documents sit at or above the high "
                  f"gate ({100.0*n_above/len(xs):.1f}%), which is the tier-0 FPR this "
                  f"operating point admits.")
    return c


def save(cals: Dict[str, Calibration], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({k: v.as_dict() for k, v in cals.items()}, f, indent=2)


def load(path: str) -> Dict[str, Calibration]:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        blob = json.load(f)
    return {k: Calibration(**v) for k, v in blob.items()}


def apply_to(cfg, cals: Dict[str, Calibration], threat: str) -> bool:
    """
    Push a fitted calibration onto a DefenceConfig. Returns True if anything changed.

    Threat-specific, because the three screens are on three unrelated scales: the insertion
    margin is signed and centred near zero, the repackaging score is a non-negative weighted
    sum, and the perturbation score is a rate times 40. A single pair of numbers cannot
    serve all three, and V2's bug was exactly that kind of borrowed constant.
    """
    c = cals.get(threat)
    if c is None or not c.n_benign:
        return False
    if threat == "repackaging":
        cfg.repack_low, cfg.repack_high = c.low, c.high
    elif threat == "perturbation":
        cfg.perturb_low, cfg.perturb_high = c.low, c.high
    else:
        cfg.escalate_low, cfg.escalate_high = c.low, c.high
    return True
