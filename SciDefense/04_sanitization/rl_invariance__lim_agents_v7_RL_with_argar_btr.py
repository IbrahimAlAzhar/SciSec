#!/usr/bin/env python3
"""
rl_invariance.py — decision invariance as an objective, and the detection-rate bound.

WHAT THIS FILE CLAIMS, AND WHAT IT DOES NOT
=======================================================================================
It claims one small, honestly-scoped inequality: given the largest rating shift an
attacker can buy, and the residual shift that survives after a detected span is excised,
there is a MINIMUM verifier recall below which the expected decision shift cannot be held
under a target. Below that recall no amount of tuning helps, because the shift is coming
from the attacks you did not catch.

It does not claim a certificate, a guarantee against adaptive attackers, or a bound on
anything the reviewer model does internally. A modest inequality with its assumptions
written down is worth more in review than a grand formula that does not survive being
read carefully, and this one is falsifiable from your own tables.

THE SETUP
    A manuscript x. An attacker picks an injection d from a constrained set D and submits
    x (+) d. A reviewer R maps a manuscript to a rating in [1, 10]. Write

        S(d) = | R(x (+) d) - R(x) |            the shift this injection buys
        S_max = max over d in D of E[S(d)]      the worst the attacker can do in D

    A defence is a detector with recall q on D and false-positive rate p on clean
    manuscripts, followed by excision and a re-review. Write

        e_res = E[ |R(sanitised) - R(x)| | detected ]     residual shift after excision
        h     = E[ |R(sanitised) - R(x)| | clean, flagged ]   damage done to honest authors

ASSUMPTIONS, ALL FOUR OF THEM
    A1  The shift is bounded: S(d) <= S_max for every d in D. Estimated, not assumed --
        `estimate_s_max` reads it off the fusion CSV.
    A2  Detection and shift magnitude are independent. This is the weak point and it is
        weak in the DEFENCE's favour or against it depending on the detector: your regex
        screen catches the loud attacks, which are also the large-shift ones, so the real
        bound is slightly kinder than this one. Stated, and testable with
        `check_independence`.
    A3  Excision does not create a shift of its own on a correctly localised span. The
        collateral term in rl_rewards prices the violations.
    A4  One injection per manuscript. Your corpus is built that way.

THE BOUND
    Expected shift, decomposing over detected and missed:

        E[S] <= q * e_res + (1 - q) * S_max + p * h                       (1)

    Requiring E[S] <= tau and solving for q:

        q >= ( S_max + p*h - tau ) / ( S_max - e_res )                    (2)

    valid when e_res < S_max and the numerator is positive; when it is not, tau is already
    met with no detector at all and the bound is vacuous, which the code reports rather
    than returning a negative recall.

WHY THIS IS USEFUL TO YOU SPECIFICALLY
    Your Qwen3-8B fusion table has d_rating 4.133 for F1_flash_only and 4.100 for
    B3_flash_payload -- text alone reaching 99.2% of the white-box ceiling. Put S_max =
    4.1, a residual of 0.3 after excision, an FPR of 0.107 costing 0.4 of a point to an
    honest author, and a tolerance of one rating point, and (2) gives

        q >= (4.1 + 0.107*0.4 - 1.0) / (4.1 - 0.3) = 0.827

    Your best detector is at recall 0.776. So the bound does not say "do better"; it says
    you are 5 points of recall short of the weakest tolerance worth defending, and it says
    it in a form a reviewer can check. That is the motivation paragraph for the RL work,
    and `main()` computes it from your CSV rather than from these illustrative numbers.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import rl_config as C


# ======================================================================================
# the bound
# ======================================================================================
@dataclass
class Bound:
    s_max: float
    e_res: float
    tau: float
    p_fpr: float
    h_collateral: float
    required_recall: Optional[float]
    vacuous: bool = False
    infeasible: bool = False
    note: str = ""

    def as_dict(self) -> Dict:
        return asdict(self)


def required_recall(s_max: float, e_res: float, tau: float, *,
                    p_fpr: float = 0.0, h_collateral: float = 0.0) -> Bound:
    """
    The minimum detector recall consistent with holding the expected shift at or under
    `tau`. Equation (2) in the module docstring.

    Three outcomes, and each is a different sentence in a paper:
      * a number in (0, 1]  -- the recall target. Compare it against what you have.
      * vacuous             -- tau is already met with no detector. Then the threat model
                               is not worth defending at this tolerance, and the honest
                               move is to tighten tau, not to report a defence.
      * infeasible          -- even perfect recall leaves the shift above tau, because the
                               residual after excision or the collateral on honest papers
                               is too large on its own. Then the detector is not the
                               bottleneck and improving it is wasted effort.
    """
    b = Bound(s_max=s_max, e_res=e_res, tau=tau, p_fpr=p_fpr,
              h_collateral=h_collateral, required_recall=None)
    if s_max <= e_res:
        b.infeasible = True
        b.note = ("the residual shift after excision is as large as the attack itself, "
                  "so excision is not repairing anything and no recall helps")
        return b
    floor = p_fpr * h_collateral + e_res
    if tau >= s_max + p_fpr * h_collateral:
        b.vacuous = True
        b.note = (f"tau={tau:.3f} is already above the worst the attacker can do "
                  f"({s_max:.3f}) plus the collateral on honest papers. No detector is "
                  f"needed at this tolerance -- tighten tau before claiming a defence.")
        b.required_recall = 0.0
        return b
    if tau < floor:
        b.infeasible = True
        b.note = (f"even at recall 1.0 the expected shift is {floor:.3f} "
                  f"(residual {e_res:.3f} + collateral {p_fpr*h_collateral:.3f}), "
                  f"above tau={tau:.3f}. The DETECTOR is not the bottleneck: reduce the "
                  f"residual (better localisation) or the false-positive rate.")
        return b
    q = (s_max + p_fpr * h_collateral - tau) / (s_max - e_res)
    b.required_recall = max(0.0, min(1.0, q))
    return b


def expected_shift(q: float, s_max: float, e_res: float, *,
                   p_fpr: float = 0.0, h_collateral: float = 0.0) -> float:
    """Equation (1): the expected rating shift at a given recall. The bound inverted."""
    return q * e_res + (1.0 - q) * s_max + p_fpr * h_collateral


# ======================================================================================
# estimating the inputs from your own data
# ======================================================================================
def estimate_s_max(episodes: Sequence, *, access: str = "text_only",
                   filler_adjust: bool = True) -> Tuple[float, Dict[str, float]]:
    """
    S_max: the largest mean |d_rating| any condition at this access level achieves.

    FILLER-ADJUSTED BY DEFAULT, and this changes the number materially. Your Qwen table
    has C1_filler at -0.417: neutral length-matched padding alone costs four-tenths of a
    rating point. So the shift ATTRIBUTABLE TO THE CONTENT of an insert is
    d_rating(attack) - d_rating(filler), which for B3 is 4.100 - (-0.417) = 4.517, not
    4.100. Reporting the unadjusted number UNDERSTATES your attack and, here, understates
    the recall the bound demands.

    Per access level, because the white-box conditions are a ceiling and not an author
    threat, and a bound computed over a pooled table would demand a recall for a threat
    model that does not apply to a venue using a hosted reviewer.
    """
    by_cond: Dict[str, List[float]] = {}
    filler: List[float] = []
    for e in episodes:
        if e.d_rating is None:
            continue
        if e.condition == C.FILLER_CONDITION:
            filler.append(float(e.d_rating))
            continue
        if e.access_level != access or e.label != 1:
            continue
        by_cond.setdefault(e.condition, []).append(float(e.d_rating))
    if not by_cond:
        return 0.0, {}
    f_mu = (sum(filler) / len(filler)) if (filler and filler_adjust) else 0.0
    means = {k: (sum(v) / len(v)) - f_mu for k, v in by_cond.items()}
    s_max = max(abs(v) for v in means.values())
    return s_max, means


def estimate_residual(excise_rows: Sequence[Dict]) -> Tuple[float, int]:
    """
    e_res: the mean |sanitised rating - benign rating| over DETECTED attacks.

    Measured on the detected arm only. Averaging over everything would fold the misses in
    and make the residual look like the whole problem, which is the term the bound already
    separates.
    """
    gaps = [abs(float(r["sanitized_rating"]) - float(r["base_rating"]))
            for r in excise_rows
            if r.get("label") == 1 and r.get("pred") == 1
            and r.get("sanitized_rating") is not None
            and r.get("base_rating") is not None]
    return (sum(gaps) / len(gaps) if gaps else float("nan")), len(gaps)


def estimate_collateral(excise_rows: Sequence[Dict]) -> Tuple[float, int]:
    """h: the mean rating damage done to a CLEAN manuscript the defence flagged."""
    gaps = [abs(float(r["sanitized_rating"]) - float(r["base_rating"]))
            for r in excise_rows
            if r.get("label") == 0 and r.get("pred") == 1
            and r.get("sanitized_rating") is not None
            and r.get("base_rating") is not None]
    return (sum(gaps) / len(gaps) if gaps else 0.0), len(gaps)


def check_independence(episodes: Sequence, preds: Dict[str, int]) -> Optional[float]:
    """
    Assumption A2, tested: is detection correlated with the size of the shift?

    Returns the point-biserial correlation between |d_rating| and "was detected", over
    attacked episodes. A strongly POSITIVE value means the detector catches the loud
    attacks, so the true expected shift is lower than (1) says and the bound is
    conservative -- report it that way. A NEGATIVE value is the bad case: the attacks that
    move ratings most are the ones being missed, and (1) understates the problem.

    None when there is nothing to correlate.
    """
    xs: List[float] = []
    ys: List[float] = []
    for e in episodes:
        if e.label != 1 or e.d_rating is None:
            continue
        if e.doc_id not in preds:
            continue
        xs.append(abs(float(e.d_rating)))
        ys.append(float(preds[e.doc_id]))
    n = len(xs)
    if n < 10 or len(set(ys)) < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    dy = math.sqrt(sum((b - my) ** 2 for b in ys))
    return num / (dx * dy) if dx and dy else None


# ======================================================================================
# reporting
# ======================================================================================
def bound_table(s_max: float, e_res: float, *, p_fpr: float = 0.0,
                h_collateral: float = 0.0,
                taus: Sequence[float] = (0.25, 0.5, 1.0, 1.5, 2.0),
                have_recall: Optional[float] = None) -> str:
    L: List[str] = []
    L.append("")
    L.append(f"  S_max (filler-adjusted)  {s_max:.3f} rating points")
    L.append(f"  residual after excision  {e_res:.3f}")
    L.append(f"  FPR on clean papers      {p_fpr:.3f}")
    L.append(f"  damage per false alarm   {h_collateral:.3f}")
    L.append("")
    L.append(f"  {'tolerance tau':>14}  {'required recall':>16}  {'you have':>9}  "
             f"{'verdict':<46}")
    L.append("  " + "-" * 92)
    for tau in taus:
        b = required_recall(s_max, e_res, tau, p_fpr=p_fpr,
                            h_collateral=h_collateral)
        if b.infeasible:
            L.append(f"  {tau:>14.2f}  {'INFEASIBLE':>16}  "
                     f"{('-' if have_recall is None else f'{have_recall:.3f}'):>9}  "
                     f"{b.note[:46]:<46}")
            continue
        if b.vacuous:
            L.append(f"  {tau:>14.2f}  {'(vacuous)':>16}  "
                     f"{('-' if have_recall is None else f'{have_recall:.3f}'):>9}  "
                     f"{'tau already met with no detector':<46}")
            continue
        q = b.required_recall or 0.0
        if have_recall is None:
            verdict = ""
        elif have_recall >= q:
            verdict = f"MET (margin +{have_recall - q:.3f})"
        else:
            verdict = f"SHORT by {q - have_recall:.3f} recall"
        L.append(f"  {tau:>14.2f}  {q:>16.3f}  "
                 f"{('-' if have_recall is None else f'{have_recall:.3f}'):>9}  "
                 f"{verdict:<46}")
    L.append("")
    L.append("  HOW TO READ IT. Pick the tolerance a venue would accept -- one rating")
    L.append("  point is already generous on a 1-10 scale -- and the row tells you the")
    L.append("  recall the detector must reach for the EXPECTED shift to stay inside it.")
    L.append("  The bound assumes detection is independent of shift size (A2); run")
    L.append("  check_independence to see which way that assumption is bending.")
    return "\n".join(L)


def latex_table(s_max: float, e_res: float, p_fpr: float, h_collateral: float,
                taus: Sequence[float], have_recall: Optional[float]) -> str:
    L = [r"\begin{table}[t]\centering\small",
         r"\caption{Minimum verifier recall consistent with holding the expected rating "
         r"shift below $\tau$, from Eq.~(2). $S_{\max}$ is filler-adjusted; the residual "
         r"and collateral terms are measured on the excision arm.}",
         r"\label{tab:invariance-bound}",
         r"\begin{tabular}{rrr}", r"\toprule",
         r"$\tau$ (rating pts) & required recall $q^\star$ & attained \\", r"\midrule"]
    for tau in taus:
        b = required_recall(s_max, e_res, tau, p_fpr=p_fpr, h_collateral=h_collateral)
        if b.infeasible:
            q = "infeasible"
        elif b.vacuous:
            q = "--"
        else:
            q = f"{b.required_recall:.3f}"
        got = "--" if have_recall is None else f"{have_recall:.3f}"
        L.append(f"{tau:.2f} & {q} & {got} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


# ======================================================================================
def main() -> int:
    p = argparse.ArgumentParser(
        description="the detection-rate bound, computed from your own CSVs")
    p.add_argument("--flash-csv", dest="flash_csv", default=C.FLASH_CSV)
    p.add_argument("--excise-json", dest="excise_json", default="",
                   help="excise/<model>/<ablation>/rows.json from rl_excise.py. Without "
                        "it the residual and collateral are taken from --residual and "
                        "--collateral and the output is a SENSITIVITY analysis, not a "
                        "measurement.")
    p.add_argument("--access", default="text_only",
                   choices=["text_only", "query", "white_box", "control"])
    p.add_argument("--residual", type=float, default=0.30)
    p.add_argument("--collateral", type=float, default=0.40)
    p.add_argument("--fpr", type=float, default=0.107,
                   help="your V5 zero-shot FPR, as the default")
    p.add_argument("--recall", type=float, default=0.776,
                   help="your V5 zero-shot recall, as the default")
    p.add_argument("--no-filler-adjust", dest="filler", action="store_false")
    p.set_defaults(filler=True)
    p.add_argument("--out-dir", dest="out_dir", default=C.wpath(C.D_FINAL))
    a = p.parse_args()

    import rl_data as D
    eps = D.load_flash(a.flash_csv, require_document=False)

    print()
    print("=" * 96)
    print(f"DECISION-INVARIANCE BOUND — access level: {a.access}")
    print("=" * 96)

    s_max, means = estimate_s_max(eps, access=a.access, filler_adjust=a.filler)
    if not means:
        print(f"[FATAL] no attacked conditions at access level {a.access!r} in "
              f"{os.path.basename(a.flash_csv)}")
        return 1
    print(f"\n  per-condition mean d_rating"
          f"{' (filler-adjusted)' if a.filler else ''}:")
    for k, v in sorted(means.items(), key=lambda kv: -abs(kv[1])):
        print(f"    {k:<28} {v:+.3f}")
    print(f"\n  S_max = {s_max:.3f}")

    e_res, n_res = a.residual, 0
    h_col, n_col = a.collateral, 0
    measured = False
    if a.excise_json and os.path.exists(a.excise_json):
        rows = json.load(open(a.excise_json))
        rows = rows.get("rows", rows) if isinstance(rows, dict) else rows
        r_est, n_res = estimate_residual(rows)
        h_est, n_col = estimate_collateral(rows)
        if r_est == r_est:
            e_res, h_col, measured = r_est, h_est, True
            print(f"\n  residual measured on {n_res} detected attacks: {e_res:.3f}")
            print(f"  collateral measured on {n_col} flagged clean papers: {h_col:.3f}")
        else:
            print(f"\n  {a.excise_json} has no re-reviewed attacked rows — falling back "
                  f"to the --residual / --collateral defaults")
    if not measured:
        print(f"\n  residual and collateral are ASSUMED ({e_res:.3f}, {h_col:.3f}). Run "
              f"rl_excise.py and pass --excise-json to measure them; until then this is a "
              f"sensitivity analysis, and it should be labelled as one in the paper.")

    print(bound_table(s_max, e_res, p_fpr=a.fpr, h_collateral=h_col,
                      have_recall=a.recall))

    # the one-line version for the motivation paragraph
    b1 = required_recall(s_max, e_res, 1.0, p_fpr=a.fpr, h_collateral=h_col)
    if b1.required_recall is not None and not b1.vacuous and not b1.infeasible:
        gap = b1.required_recall - a.recall
        print()
        print("  FOR THE MOTIVATION PARAGRAPH")
        print(f"    To hold the expected rating shift under one point on a 1-10 scale, "
              f"the verifier")
        print(f"    needs recall {b1.required_recall:.3f}. The zero-shot cascade attains "
              f"{a.recall:.3f}"
              + (f", a shortfall of {gap:.3f}." if gap > 0 else ", which already meets it."))
        if gap > 0:
            print(f"    That shortfall, not an F1 delta, is what the RL stage has to "
                  f"close.")

    os.makedirs(a.out_dir, exist_ok=True)
    out = {"access": a.access, "s_max": s_max, "means": means,
           "residual": e_res, "collateral": h_col, "fpr": a.fpr,
           "attained_recall": a.recall, "measured": measured,
           "filler_adjusted": a.filler,
           "bounds": {str(t): required_recall(s_max, e_res, t, p_fpr=a.fpr,
                                              h_collateral=h_col).as_dict()
                      for t in (0.25, 0.5, 1.0, 1.5, 2.0)}}
    jp = os.path.join(a.out_dir, f"invariance_bound_{a.access}.json")
    with open(jp, "w") as f:
        json.dump(out, f, indent=2)
    tp = os.path.join(a.out_dir, f"invariance_bound_{a.access}.tex")
    with open(tp, "w") as f:
        f.write(latex_table(s_max, e_res, a.fpr, h_col,
                            (0.25, 0.5, 1.0, 1.5, 2.0), a.recall) + "\n")
    print(f"\n[saved] {jp}")
    print(f"[saved] {tp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
