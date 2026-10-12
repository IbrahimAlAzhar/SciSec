#!/usr/bin/env python3
"""
rl_sweep.py — train one adapter per hyperparameter setting, then rank them.

    python rl_sweep.py --plan                 # print the grid and the cost, run nothing
    python rl_sweep.py --emit > sweep.sh      # the qsub lines; bash sweep.sh to submit
    python rl_sweep.py --report               # rank every finished run
    python rl_sweep.py --report --pick        # print the qsub line for the winner

WHY A SWEEP, AND WHY IT IS NOT JUST lr
=======================================================================================
The evidence said lr alone was not the story. One run reported

    mean_kl 321   mean_ratio 0.992   clipfrac 3.9%   max_ratio 14.3

and those numbers do not describe a policy that drifted. lp_old == lp_ref in this trainer,
so d = -log(rho) and

    k3 = 1/rho + log(rho) - 1

which means the KL is driven ENTIRELY by tokens whose probability collapsed:

    rho 14.3  -> k3   1.7      <- max_ratio, the only tail that was instrumented
    rho 0.01  -> k3  94
    rho 0.001 -> k3 992

So mean_kl 321 alongside mean_ratio 0.992 means almost every token sat still while a few
fell through the floor. Raising or lowering lr changes how often that happens; it is not
the mechanism. Two changes address the mechanism directly, and both are in place before
this sweep runs, because a sweep that ranks configs by an outlier-dominated statistic
just picks the luckiest seed:

    min_ratio, kl_p50, kl_p95, kl_max, collapsed_frac   now in grpo_summary.json
    kl_clamp                                            caps one token's contribution to
                                                        the PENALTY, whose gradient is
                                                        exponential in d and was the
                                                        largest destabilising force once
                                                        any token collapsed

READ kl_p50 AGAINST mean_kl. If p50 is tiny and the mean is huge, the run is a few bad
tokens and the config may be fine. If p50 itself is large, the whole policy moved.

WHY SHORT RUNS
    A full pass is ~265 updates and 80-90 minutes, so a 12-point grid is 18 hours. KL
    divergence shows up in the first few dozen updates, so the grid runs at MAX_STEPS=40
    (~12 min each) and only the surviving configs get a full run. 12 short runs is about
    2.5 hours, which fits one allocation.

WHY EACH RUN GETS ITS OWN DIRECTORY
    train_grpo derives its output path from the ablation, so every run in a sweep would
    otherwise overwrite the same adapter and you would keep only the last. Each point here
    writes to adapters/sweep/<ablation>__<tag>/ with an hparams.json beside it, and the
    canonical adapters/<model>-<stage>/ is never touched by a sweep. Promote the winner
    explicitly, with --pick, once you have looked at the table.
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import sys
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------------------
# THE GRID
#
# lr          1e-6 measured inert (mean_kl 0.0009), 2e-5 measured diverged (321). The
#             usable value is between them and much nearer the bottom, so the points are
#             log-spaced across that interval rather than centred in it.
# beta_kl     the KL penalty weight. 0.04 was in use; 0.15 holds the policy closer and is
#             the cheapest defence against the collapsed-token failure.
# grad_accum  updates, not step size. 1060 examples / accum = updates. 16 gave 67 updates,
#             4 gives 265. If a small lr is inert, THIS is the dial to turn, not lr.
#
# clip_eps is deliberately NOT swept: at 3.9% clipfrac it barely engages, so varying it
# would change nothing and cost a third of the grid.
# --------------------------------------------------------------------------------------
GRID: Dict[str, List] = {
    "LR": [1e-6, 3e-6, 6e-6, 1.2e-5],
    "BETA": [0.04, 0.15],
    "GRAD_ACCUM": [4, 16],
}
SHORT_STEPS = 40


def tag_of(pt: Dict) -> str:
    """A directory-safe name that states the setting, so a listing is self-describing."""
    # .2g, not .0e: 1.2e-05 formatted with .0e becomes "1e-05", which is the SAME tag as
    # 1e-06 once the exponent is shortened, and two grid points would then share a
    # directory and overwrite each other.
    return (f"lr{pt['LR']:.2g}".replace("e-0", "e-").replace(".", "p") +
            f"_b{pt['BETA']:g}".replace(".", "p") +
            f"_ga{pt['GRAD_ACCUM']}")


def points() -> List[Dict]:
    keys = list(GRID)
    return [dict(zip(keys, combo)) for combo in itertools.product(*GRID.values())]


def sweep_dir(abl: str, tag: str, work: Optional[str] = None) -> str:
    return os.path.join(work or HERE, "adapters", "sweep", f"{abl}__{tag}")


# ======================================================================================
def plan(abl: str, iter_: int, short: int) -> None:
    pts = points()
    per = 12 if short else 85
    print(f"grid: {' x '.join(f'{k}({len(v)})' for k, v in GRID.items())} "
          f"= {len(pts)} runs")
    print(f"each: MAX_STEPS={short or 'full'}  ~{per} min  ->  total ~{len(pts)*per/60:.1f} "
          f"GPU hours\n")
    w = max(len(tag_of(p)) for p in pts)
    print(f"  {'tag':<{w}}  {'LR':>8} {'BETA':>5} {'GRAD_ACCUM':>10} {'updates':>8}")
    print("  " + "-" * (w + 38))
    for p in pts:
        upd = short or (1060 // p["GRAD_ACCUM"])
        print(f"  {tag_of(p):<{w}}  {p['LR']:>8.1e} {p['BETA']:>5g} "
              f"{p['GRAD_ACCUM']:>10} {upd:>8}")
    print(f"\noutputs: adapters/sweep/{abl}__<tag>/   "
          f"(canonical adapters/<model>-<stage>/ is NOT touched)")
    print("then:    python rl_sweep.py --report")


def emit(abl: str, iter_: int, short: int, job: str) -> None:
    print("#!/bin/bash")
    print(f"# {len(points())} runs. Submitted serially: each holds one GPU, and a sweep")
    print("# that queues them all at once will have most of them waiting anyway.")
    print("# Each writes adapters/sweep/<abl>__<tag>/ plus hparams.json.")
    print("set -u")
    prev = ""
    for p in points():
        tag = tag_of(p)
        out = f"$AD_ROOT/defense/lim_agents_v7_RL/adapters/sweep/{abl}__{tag}"
        dep = f" -W depend=afterany:$PREV" if prev else ""
        print(f'\n# ---- {tag} : lr={p["LR"]:g} beta_kl={p["BETA"]:g} '
              f'grad_accum={p["GRAD_ACCUM"]} ----')
        print(f'JID=$(qsub{dep} -v ABL={abl},ITER={iter_},LR={p["LR"]:g},'
              f'BETA={p["BETA"]:g},GRAD_ACCUM={p["GRAD_ACCUM"]},'
              f'MAX_STEPS={short},OUT_DIR={out} {job})')
        print('echo "  submitted $JID  ' + tag + '"')
        print("PREV=$JID")
        prev = tag
    print('\necho; echo "when they finish:  python rl_sweep.py --report"')


# ======================================================================================
def load_runs(abl: Optional[str]) -> List[Dict]:
    out = []
    for s in sorted(glob.glob(os.path.join(HERE, "adapters", "sweep", "*",
                                           "grpo_summary.json"))):
        d = os.path.dirname(s)
        name = os.path.basename(d)
        if abl and not name.startswith(abl + "__"):
            continue
        try:
            rec = json.load(open(s))
        except Exception as e:
            print(f"[warn] unreadable {s}: {e}", file=sys.stderr)
            continue
        # hparams.json is written by the PBS with a shell heredoc, so it can land EMPTY or
        # half-expanded: under `set -u` an undefined variable inside a heredoc makes bash
        # abandon the expansion after having already truncated the file, and the enclosing
        # loop carries on. The authoritative numbers are in grpo_summary.json, which the
        # trainer writes with json.dump, so an unreadable hparams.json must never take the
        # report down with it.
        hp: Dict = {}
        hpp = os.path.join(d, "hparams.json")
        if os.path.exists(hpp):
            try:
                hp = json.load(open(hpp)) or {}
            except Exception as e:
                print(f"[warn] {hpp}: {type(e).__name__}: {e} — using grpo_summary.json "
                      f"only", file=sys.stderr)
        rec["_tag"] = name.split("__", 1)[-1]
        rec["_dir"] = d
        rec["_hparams"] = hp
        out.append(rec)
    return out


def score(r: Dict) -> "tuple":
    """
    Rank key. The usable band FIRST, then stability, then how far it got.

    Sorting by KL alone would put the inert runs on top, since 0.0009 is closest to zero
    and useless. So a run is bucketed by whether its mean_kl lands in [0.01, 0.1], and only
    inside a bucket does the tail behaviour decide.
    """
    kl = r.get("mean_kl")
    kl = float("inf") if kl is None else kl
    band = 0 if 0.01 <= kl <= 0.1 else (1 if kl < 0.01 else 2)
    # A run with no collapsed_frac predates the diagnostics and cannot be judged on the
    # tail. `or 0.0` would score it as PERFECTLY stable and float it to the top of its
    # band -- a missing measurement winning on the strength of being missing. It sorts
    # last within its band instead, and `complete` keeps it out of --pick.
    coll = r.get("collapsed_frac")
    incomplete = 0 if coll is not None else 1
    return (band, incomplete, coll if coll is not None else 0.0, abs(kl - 0.03))


REQUIRED_FOR_PICK = ("lr", "beta_kl", "grad_accum")


def complete(r: Dict) -> bool:
    """Enough fields recorded to reconstruct the run's command line."""
    return all(r.get(k) is not None for k in REQUIRED_FOR_PICK)


def report(abl: Optional[str], pick: bool, job: str) -> int:
    runs = load_runs(abl)
    if not runs:
        print(f"no finished sweep runs under {os.path.join(HERE, 'adapters', 'sweep')}.")
        print("run  python rl_sweep.py --plan  then  --emit")
        return 0
    runs.sort(key=score)
    w = max(len(r["_tag"]) for r in runs)

    def band_label(kl) -> str:
        if kl is None:
            return "no kl"
        if kl < 0.005:
            return "INERT"
        if kl <= 0.1:
            return "USABLE"
        if kl <= 1.0:
            return "HOT"
        if kl <= 10:
            return "TOO FAR"
        return "DIVERGED"

    print("Ranked by: usable KL band, then fewest collapsed tokens.")
    print("mean_kl is the MEAN over tokens; read kl_p50 beside it. p50 tiny + mean huge")
    print("means a few collapsed tokens, not a policy that drifted.\n")
    def cell(rec: Dict, key: str, spec: str, width: int) -> str:
        """One value, or an em dash of the same width when the run predates the field."""
        v = rec.get(key)
        if v is None:
            return f"{'—':>{width}}"
        try:
            return f"{format(v, spec):>{width}}"
        except (TypeError, ValueError):
            return f"{str(v):>{width}}"

    cols = [("mean_kl", ".4f", 9), ("kl_p50", ".4f", 8), ("kl_p95", ".4f", 8),
            ("min_ratio", ".2e", 9), ("collapsed_frac", ".2%", 8)]
    head = f"  {'tag':<{w}} {'lr':>8} {'beta':>5} {'ga':>3} {'upd':>4}"
    for name, _spec, width in cols:
        head += f" {name:>{width}}"
    print(head + f" {'verdict':>9}")
    print("  " + "-" * (len(head) + 10))
    for r in runs:
        line = (f"  {r['_tag']:<{w}} {r.get('lr', 0):>8.1e} "
                f"{r.get('beta_kl', 0):>5g} {r.get('grad_accum', 0):>3} "
                f"{r.get('n_updates', 0):>4}")
        for name, spec, width in cols:
            line += " " + cell(r, name, spec, width)
        print(line + f" {band_label(r.get('mean_kl')):>9}")

    best = runs[0]
    kl = best.get("mean_kl")
    print()
    if kl is None or not (0.01 <= kl <= 0.1):
        print("NO CONFIG LANDED IN THE USABLE BAND.")
        if kl is not None and kl < 0.01:
            print("  Everything is inert. Raise UPDATES before step size: GRAD_ACCUM=2,")
            print("  and check stage A reported reward variance — a group whose members all")
            print("  score the same contributes no gradient at any learning rate.")
        else:
            print("  Everything diverged. Lower the lr floor in GRID and raise BETA.")
        return 0
    coll = best.get("collapsed_frac")
    print(f"BEST: {best['_tag']}   mean_kl {kl:.4f}   "
          f"collapsed {f'{coll:.2%}' if coll is not None else 'not recorded'}")
    print(f"      {best['_dir']}")
    if pick and not complete(best):
        missing = [k for k in REQUIRED_FOR_PICK if best.get(k) is None]
        print(f"\n[cannot --pick] {best['_tag']} is missing {missing} in its "
              f"grpo_summary.json,\n  so the command line cannot be reconstructed. It was "
              f"trained before those fields\n  were recorded. Re-run it through rl_sweep, "
              f"or read its hparams.json by hand:\n  {best['_dir']}/hparams.json")
        return 0
    if pick:
        a = abl or "G_grpo_prov"
        print("\nFull run at that setting, writing the CANONICAL adapter:")
        print(f"  qsub -v ABL={a},ITER=0,LR={best.get('lr'):g},"
              f"BETA={best.get('beta_kl'):g},GRAD_ACCUM={best.get('grad_accum')} {job}")
        print("\nThen evaluate it:")
        print(f"  qsub -v ABL={a} jobs/job_v7_c_gemma_infer.pbs")
        print("\nA short run is 40 updates. Re-check mean_kl on the full run before")
        print("evaluating: the same lr over 265 updates travels further than over 40.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="hyperparameter sweep for GRPO adapters")
    p.add_argument("--ablation", "--abl", dest="abl", default="G_grpo_prov")
    p.add_argument("--iter", dest="iter_", type=int, default=0)
    p.add_argument("--short", type=int, default=SHORT_STEPS,
                   help=f"MAX_STEPS per sweep point (default {SHORT_STEPS}; 0 = full run)")
    p.add_argument("--job", default="jobs/job_v7_b_gemma_train.pbs")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--plan", action="store_true")
    g.add_argument("--emit", action="store_true")
    g.add_argument("--report", action="store_true")
    p.add_argument("--pick", action="store_true",
                   help="with --report: print the qsub line for the winner")
    a = p.parse_args()

    if a.plan:
        plan(a.abl, a.iter_, a.short)
    elif a.emit:
        emit(a.abl, a.iter_, a.short, a.job)
    else:
        return report(a.abl, a.pick, a.job)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
