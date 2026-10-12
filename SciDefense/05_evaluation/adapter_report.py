#!/usr/bin/env python3
"""
adapter_report.py — what learning rate trained each adapter, and how far it moved.

    python adapter_report.py                       # v7's adapters + V4's SFT/DPO adapters
    python adapter_report.py --roots DIR [DIR ...] # only these
    python adapter_report.py --stage dpo           # dpo / sft / grpo / all
    python adapter_report.py --json                # machine-readable

WHY NOT JUST grep THE JSON
=======================================================================================
Three reasons, and the first one has already produced a wrong conclusion in this project.

1. SYMLINKS. `adapters_serve_<tag>/` and `adapters_view_<tag>/` are generated symlink
   VIEWS of the real adapters directory -- stage C builds one with `ln -sfn` on every run.
   A glob over both roots therefore reports the same adapter twice, and renaming an entry
   inside a view renames the LINK while the target stays put. That is how two rows came
   back with byte-identical mean_kl 320.9786 and looked like two different training runs
   at two different learning rates.

   So every path is resolved with os.path.realpath and duplicates are collapsed, with the
   links that pointed at each target listed underneath.

2. THE STAGES DO NOT SHARE A METRIC, and pretending they do is worse than saying so.

       GRPO  grpo_summary.json  has mean_kl: the k3 KL from the reference policy. This is
             a real divergence measurement and the number the trust region watches.
       DPO   ft_meta.json       has beta, which is the COEFFICIENT on the implicit KL
             penalty, not a measured divergence. There is NO mean KL for a DPO adapter
             unless a trainer_state.json survived, and reporting beta in a "KL" column
             would be a category error. Where trainer_state.json exists this prints the
             DPO quantities that do mean something: reward margin and reward accuracy.
       SFT   ft_meta.json       has no divergence metric at all. Cross-entropy to the
             target, no reference policy. The column is n/a and that is correct.

3. WHICH FILE IS MISSING IS ITSELF THE ANSWER. An adapter with adapter_config.json and no
   metadata was either trained by something that did not write any, or copied without it.
   This prints the directory anyway, marked, rather than silently skipping it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional


def default_roots() -> List[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    ad = os.environ.get("AD_ROOT") or os.path.dirname(os.path.dirname(here))
    v4 = os.environ.get("V4_FT_DIR") or os.path.join(ad, "defense",
                                                     "llim_agents_v4_SFT_DPO")
    cands = [os.path.join(here, "adapters"), os.path.join(v4, "adapters")]
    # the generated views, so their links are reported as links rather than as adapters
    for base in (here, v4):
        for entry in sorted(os.listdir(base)) if os.path.isdir(base) else []:
            if entry.startswith(("adapters_view", "adapters_serve")):
                cands.append(os.path.join(base, entry))
    return [c for c in cands if os.path.isdir(c)]


def load_json(path: str) -> Optional[Dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def dpo_metrics(d: str) -> Dict:
    """
    Reward margin / accuracy from a surviving trainer_state.json.

    DPO's meaningful health numbers are rewards/margins (how much more implicit reward the
    chosen response gets) and rewards/accuracies (how often chosen beats rejected). A
    margin near zero means the preference pairs taught it nothing, which is the DPO
    equivalent of the inert-adapter problem. `beta` is a setting, not a measurement.
    """
    out: Dict = {}
    for cand in ("trainer_state.json",
                 os.path.join("checkpoint-last", "trainer_state.json")):
        st = load_json(os.path.join(d, cand))
        if not st:
            continue
        hist = [h for h in (st.get("log_history") or []) if isinstance(h, dict)]
        for key, short in (("rewards/margins", "margin"),
                           ("rewards/accuracies", "acc"),
                           ("train_loss", "loss"), ("loss", "loss")):
            vals = [h[key] for h in hist if key in h]
            if vals and short not in out:
                out[short] = float(vals[-1])
        if hist:
            out["steps"] = int(hist[-1].get("step", 0) or 0)
        break
    return out


def rank_of(d: str) -> Optional[int]:
    cfg = load_json(os.path.join(d, "adapter_config.json"))
    return int(cfg.get("r")) if cfg and cfg.get("r") is not None else None


def verdict(stage: str, rec: Dict) -> str:
    if stage == "grpo":
        kl = rec.get("mean_kl")
        if kl is None:
            return "no mean_kl recorded"
        if kl < 0.005:
            return "INERT — decisions will not move"
        if kl <= 0.1:
            return "USABLE — this is the band"
        if kl <= 1.0:
            return "HOT — drifting from the reference"
        if kl <= 10.0:
            return "TOO FAR — check completions before using"
        return "DIVERGED — do not evaluate"
    if stage == "dpo":
        m = rec.get("margin")
        if m is None:
            return "no trainer_state.json — margin unknown"
        if abs(m) < 0.05:
            return "FLAT — the preference pairs taught it ~nothing"
        return "trained"
    if stage == "sft":
        return "SFT — no reference policy, no KL by construction"
    return "unknown stage"


def collect(roots: List[str]) -> "tuple[List[Dict], Dict[str, List[str]]]":
    by_real: Dict[str, Dict] = {}
    links: Dict[str, List[str]] = {}
    for root in roots:
        for entry in sorted(os.listdir(root)):
            p = os.path.join(root, entry)
            if not os.path.isdir(p):
                continue
            real = os.path.realpath(p)
            if real != os.path.abspath(p):
                links.setdefault(real, []).append(os.path.abspath(p))
            if not os.path.exists(os.path.join(real, "adapter_config.json")):
                continue
            if real in by_real:
                continue

            grpo = load_json(os.path.join(real, "grpo_summary.json"))
            ftm = load_json(os.path.join(real, "ft_meta.json"))
            rec: Dict = {"dir": real, "name": os.path.basename(real),
                         "r": rank_of(real)}
            if grpo:
                rec["stage"] = "grpo"
                rec.update({k: grpo.get(k) for k in
                            ("lr", "mean_kl", "mean_ratio", "max_ratio",
                             "mean_clipfrac", "n_updates", "minutes", "beta_kl")})
            elif ftm:
                rec["stage"] = str(ftm.get("stage") or "?").lower()
                rec.update({k: ftm.get(k) for k in
                            ("lr", "epochs", "n_train", "beta", "role",
                             "from_sft", "ref_is_sft")})
                if rec["stage"].startswith("dpo"):
                    rec.update(dpo_metrics(real))
            else:
                rec["stage"] = "no-metadata"
            by_real[real] = rec
    return list(by_real.values()), links


def fmt(v, spec: str, na: str = "—") -> str:
    if v is None:
        return f"{na:>{len(f'{0:{spec}}')}}"
    try:
        return f"{v:{spec}}"
    except (TypeError, ValueError):
        return f"{str(v):>8}"


def main() -> int:
    p = argparse.ArgumentParser(description="learning rate and divergence per adapter")
    p.add_argument("--roots", nargs="+", default=None)
    p.add_argument("--stage", default="all",
                   help="all | grpo | dpo | sft | no-metadata (substring match)")
    p.add_argument("--json", dest="as_json", action="store_true")
    a = p.parse_args()

    roots = a.roots or default_roots()
    if not roots:
        print("[fatal] no adapter directories found. Pass --roots.", file=sys.stderr)
        return 1
    print("[roots]")
    for r in roots:
        print(f"    {r}")
    print()

    recs, links = collect(roots)
    if a.stage != "all":
        recs = [r for r in recs if a.stage.lower() in r["stage"]]
    if not recs:
        print("no adapters matched.")
        return 0
    recs.sort(key=lambda r: (r["stage"], r["name"]))

    if a.as_json:
        print(json.dumps({"adapters": recs, "symlinks": links}, indent=2, default=str))
        return 0

    # ---- GRPO: mean_kl is a real measurement -----------------------------------------
    g = [r for r in recs if r["stage"] == "grpo"]
    if g:
        w = max(len(r["name"]) for r in g)
        print("GRPO — mean_kl is the measured k3 KL from the reference policy")
        print(f"  {'adapter':<{w}} {'lr':>8} {'upd':>5} {'mean_kl':>10} {'rho':>6} "
              f"{'max_rho':>8} {'clip':>6} {'r':>3}  verdict")
        print("  " + "-" * (w + 68))
        for r in g:
            print(f"  {r['name']:<{w}} {fmt(r.get('lr'), '8.1e')} "
                  f"{fmt(r.get('n_updates'), '5d')} {fmt(r.get('mean_kl'), '10.4f')} "
                  f"{fmt(r.get('mean_ratio'), '6.3f')} {fmt(r.get('max_ratio'), '8.1f')} "
                  f"{fmt(r.get('mean_clipfrac'), '6.2%')} {fmt(r.get('r'), '3d')}  "
                  f"{verdict('grpo', r)}")
        print("  bands: <0.005 inert | 0.01-0.1 usable | >1 hot | >10 diverged\n")

    # ---- DPO: beta is a SETTING; margin/accuracy are the measurements -----------------
    d = [r for r in recs if r["stage"].startswith("dpo")]
    if d:
        w = max(len(r["name"]) for r in d)
        print("DPO — beta is the KL-penalty COEFFICIENT, not a measured divergence.")
        print("      There is no mean KL for a DPO adapter; margin/acc come from")
        print("      trainer_state.json when it survived the save.")
        print(f"  {'adapter':<{w}} {'lr':>8} {'beta':>6} {'epochs':>7} {'n_train':>8} "
              f"{'margin':>8} {'acc':>7} {'r':>3}  verdict")
        print("  " + "-" * (w + 70))
        for r in d:
            print(f"  {r['name']:<{w}} {fmt(r.get('lr'), '8.1e')} "
                  f"{fmt(r.get('beta'), '6.3f')} {fmt(r.get('epochs'), '7.2f')} "
                  f"{fmt(r.get('n_train'), '8d')} {fmt(r.get('margin'), '8.3f')} "
                  f"{fmt(r.get('acc'), '7.2%')} {fmt(r.get('r'), '3d')}  "
                  f"{verdict('dpo', r)}")
        print()

    # ---- SFT --------------------------------------------------------------------------
    s = [r for r in recs if r["stage"] == "sft"]
    if s:
        w = max(len(r["name"]) for r in s)
        print("SFT — cross-entropy to a target, no reference policy, so no KL exists")
        print(f"  {'adapter':<{w}} {'lr':>8} {'epochs':>7} {'n_train':>8} {'role':>16} {'r':>3}")
        print("  " + "-" * (w + 48))
        for r in s:
            print(f"  {r['name']:<{w}} {fmt(r.get('lr'), '8.1e')} "
                  f"{fmt(r.get('epochs'), '7.2f')} {fmt(r.get('n_train'), '8d')} "
                  f"{str(r.get('role') or '-'):>16} {fmt(r.get('r'), '3d')}")
        print()

    n = [r for r in recs if r["stage"] == "no-metadata"]
    if n:
        print("NO METADATA — adapter_config.json present, but no grpo_summary.json or")
        print("              ft_meta.json, so the learning rate is unrecoverable here.")
        for r in n:
            print(f"  {r['name']:<44} {r['dir']}")
        print()

    if links:
        print("SYMLINKS COLLAPSED — these paths point at an adapter already listed above.")
        print("  A view directory is generated; renaming an entry inside one renames the")
        print("  LINK, not the adapter, and both names then report identical numbers.")
        for real, ls in sorted(links.items()):
            print(f"  {os.path.basename(real)}")
            for l in ls:
                print(f"      <- {l}")
        print()

    print(f"{len(recs)} distinct adapter(s); {sum(len(v) for v in links.values())} "
          f"symlink path(s) collapsed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
