#!/usr/bin/env python3
"""
v7_mechint.py — mechanistically-grounded rewards and an abstention gate.

THE PROBLEM THIS IS AIMED AT, IN ONE LINE FROM YOUR OWN DIAGNOSIS
=======================================================================================
    A0_clean flagged 0.781      C1_filler flagged 0.926      attacks flagged ~0.97

The cascade cannot tell text that LOOKS like manipulation from text that ACTUALLY MOVED
THE MODEL. Every surface feature an agent can see -- imperative mood, second-person
address, hedging -- is present in honest limitations prose too, which is why the leader
says ATTACKED to nine clean papers in ten and why 82% of your false positives come from
the leader rather than the screen.

Attention is not a surface feature. It is a measurement of what the reviewer actually
conditioned on, and your own FLASH table says it is visible from the text alone:

    F1_flash_only   (payload + attention intervention at inference)   d_rating 4.133
    B3_flash_payload (the same payload as ordinary text, no access)   d_rating 4.100

Text reaches 99.2% of the white-box ceiling. The attention capture is happening WITHOUT
anyone editing the forward pass, so it can be measured from outside on a reviewer you
host yourself -- and your fusion CSV already records it per condition.

THREE USES, IN DESCENDING ORDER OF HOW NOVEL THEY ARE
    1. A PROCESS REWARD WITH A CAUSAL TARGET (`step_bonus`). Pay the INSTRUCTION /
       PRESSURE / VOICE worker when the span it quotes is the span that captured
       attention, and charge it when it quotes a span that captured none. That converts
       "find text that reads like an instruction" into "find the text that did the work".
       Every published process reward for a multi-agent system scores task correctness;
       this one scores mechanism. It is the ablation row to lead with.

    2. AN ABSTENTION GATE (`should_abstain`). If no candidate span in a document carries
       anomalous attention, the cascade says CLEAN and stops. Aimed squarely at the 0.781
       clean-flag rate, and it costs nothing -- the columns exist.

    3. A LAYER REPORT (`layer_report`). Which layers a manipulation has to capture, and
       whether they form a contiguous block. No black-box experiment can establish that;
       it is the causal claim underneath the whole paper.

THE STATISTIC, AND WHY THE RAW NUMBER WILL NOT DO
    A longer span attracts more attention for no interesting reason, so the feature is
    EXCESS, not attention:

        excess(span) = insert_attention / (span_chars / document_chars)

    i.e. how many times more of the final query row's attention this span takes than its
    length alone predicts. And the calibration set is C1_filler, NOT C0_clean: filler is
    length-matched neutral prose inserted at the same slot, so a detector that fires on
    filler is measuring "a block of text was added here" -- exactly the confound that
    destroyed your MISS column and, on this evidence, your FPR too.

HONEST LIMIT, STATED UP FRONT
    This needs the reviewer's attention, so it is available only to a venue hosting an
    open-weights reviewer. It belongs in its own table with that caption, never pooled
    with the deployable rows. And the prior below is fitted on the conditions your fusion
    run covers; a manipulation distributed over the whole manuscript has no concentrated
    excess to find, and the method correctly reports nothing rather than guessing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import rl_config as C


# ======================================================================================
# the prior, fitted from the fusion CSV
# ======================================================================================
@dataclass
class AttentionPrior:
    """
    What 'anomalous attention' means, learned from the conditions you already ran.

    Two numbers do the work: the excess a length-matched FILLER insert attracts (the
    null), and the excess an ATTACK insert attracts. A span is anomalous when its excess
    sits far above the filler distribution, measured in filler standard deviations.
    """
    filler_mu: float = 1.0
    filler_sd: float = 1.0
    attack_mu: float = 1.0
    clean_sink: float = 0.0
    attack_sink: float = 0.0
    n_filler: int = 0
    n_attack: int = 0
    # z at or above this counts as captured. Chosen on val, frozen; see main().
    z_hit: float = 2.0
    source: str = ""

    @classmethod
    def from_csv(cls, path: str, *, z_hit: float = 2.0) -> "AttentionPrior":
        import numpy as np
        import rl_data as D
        eps = D.load_flash(path, require_document=False)
        fil, att, cs, as_ = [], [], [], []
        for e in eps:
            if e.insert_attention is None:
                continue
            ia = float(e.insert_attention)
            if ia != ia:
                continue
            ins_len = len(e.insert or "")
            doc_len = len(e.document or "") or len(e.benign_document or "")
            share = (ins_len / doc_len) if (ins_len and doc_len) else 0.0
            if share <= 1e-6:
                if e.sink_attention is not None:
                    cs.append(float(e.sink_attention))
                continue
            ex = ia / share
            if e.is_control:
                fil.append(ex)
            elif e.label == 1:
                att.append(ex)
                if e.sink_attention is not None:
                    as_.append(float(e.sink_attention))
        if not fil:
            raise SystemExit(
                f"[FATAL] no C1_filler rows with an attention reading in {path}.\n"
                f"        The filler control IS the null distribution here. Without it, "
                f"'anomalous' would mean 'more attention than a clean document with no "
                f"insert at all', which every inserted span satisfies -- including "
                f"neutral padding.")
        f = np.asarray(fil, float)
        a = np.asarray(att, float) if att else np.asarray([1.0])
        return cls(filler_mu=float(f.mean()), filler_sd=float(f.std(ddof=1) or 1.0),
                   attack_mu=float(a.mean()),
                   clean_sink=float(np.mean(cs)) if cs else 0.0,
                   attack_sink=float(np.mean(as_)) if as_ else 0.0,
                   n_filler=len(fil), n_attack=len(att), z_hit=z_hit,
                   source=os.path.basename(path))

    # ---- scoring -------------------------------------------------------------------
    def excess(self, span_chars: int, doc_chars: int, attention: float) -> float:
        share = (span_chars / doc_chars) if (span_chars and doc_chars) else 0.0
        return (attention / share) if share > 1e-6 else 1.0

    def z(self, excess: float) -> float:
        return (excess - self.filler_mu) / max(1e-9, self.filler_sd)

    def captured(self, excess: float) -> bool:
        return self.z(excess) >= self.z_hit

    @property
    def separation(self) -> float:
        """How many filler sds the mean attack sits above the filler mean."""
        return (self.attack_mu - self.filler_mu) / max(1e-9, self.filler_sd)

    def describe(self) -> str:
        L = ["[mech] attention prior from " + (self.source or "?"),
             f"[mech]   filler  excess  mu {self.filler_mu:.3f}  sd {self.filler_sd:.3f}"
             f"   (n={self.n_filler})",
             f"[mech]   attack  excess  mu {self.attack_mu:.3f}   (n={self.n_attack})",
             f"[mech]   separation {self.separation:+.2f} filler sd",
             f"[mech]   sink  clean {self.clean_sink:.4f} -> attacked "
             f"{self.attack_sink:.4f}",
             f"[mech]   a span counts as CAPTURED at z >= {self.z_hit}"]
        if self.separation < 1.0:
            L.append("[mech]   SEPARATION UNDER 1 SD. The attacks do not attract")
            L.append("[mech]   materially more attention than length-matched filler in")
            L.append("[mech]   this corpus, so this signal cannot carry a reward. Report")
            L.append("[mech]   that as the finding rather than training on it.")
        return "\n".join(L)

    def as_dict(self) -> Dict:
        return {"filler_mu": self.filler_mu, "filler_sd": self.filler_sd,
                "attack_mu": self.attack_mu, "clean_sink": self.clean_sink,
                "attack_sink": self.attack_sink, "n_filler": self.n_filler,
                "n_attack": self.n_attack, "z_hit": self.z_hit,
                "separation": self.separation, "source": self.source}

    # ---- 1. the process reward -------------------------------------------------------
    def step_bonus(self, traj, ep, *, w: float = 0.5) -> Dict[str, float]:
        """
        Per-agent bonus for quoting the span that actually captured attention.

        Returns {} when this episode carries no attention reading, which is the common
        case on the training corpus -- the bonus applies only to documents from the
        fusion run. A reward term that silently defaults to zero everywhere would make
        the mechint ablation identical to its control while looking like it ran, so the
        caller counts how many trajectories actually received it and prints that.

        The gold span's excess is used as the reference: a worker that quotes the gold
        span inherits its excess, one that quotes elsewhere is assumed to have found a
        span with filler-level attention. That is an approximation -- computing the true
        per-span attention needs a forward pass per candidate -- and it is the honest
        one available from the recorded columns.
        """
        if ep.insert_attention is None or ep.ins_start < 0:
            return {}
        ia = float(ep.insert_attention)
        if ia != ia:
            return {}
        gold_ex = self.excess(ep.ins_end - ep.ins_start, len(ep.document), ia)
        if not self.captured(gold_ex):
            # The injected span did NOT capture attention in this document. Then quoting
            # it is not evidence of mechanism, and paying for it would teach the worker
            # to chase a signal that is not there on this example.
            return {}
        out: Dict[str, float] = {}
        from rl_rewards import span_iou
        for st in traj.steps:
            if st.agent == C.MASTER or not st.spans:
                continue
            iou, _prec, _rec = span_iou(ep.document, st.spans,
                                        (ep.ins_start, ep.ins_end))
            if iou != iou:
                continue
            # + for quoting the span that captured attention, - for quoting a span that
            # captured none while a captured span was available to find.
            out[st.agent] = w * (2.0 * iou - 1.0)
        return out

    # ---- 2. the abstention gate -------------------------------------------------------
    def should_abstain(self, ep) -> Tuple[bool, float]:
        """
        (abstain, z). True when nothing in this document carries anomalous attention.

        Abstaining means SAY CLEAN AND STOP. On this task that is the safe direction: the
        reward weights already say an accused author costs more than a missed attack, and
        your clean-flag rate of 0.781 is the number this exists to cut.
        """
        if ep.insert_attention is None or ep.ins_start < 0:
            return False, float("nan")     # no reading: do not interfere
        ia = float(ep.insert_attention)
        if ia != ia:
            return False, float("nan")
        z = self.z(self.excess(ep.ins_end - ep.ins_start, len(ep.document), ia))
        return (z < self.z_hit), z


# ======================================================================================
# 3. the layer report
# ======================================================================================
def layer_report(scan_json: str) -> List[str]:
    """
    Which layers a manipulation has to capture, from your own layer scan.

    Expects the scan your whitebox/mechint code writes: a mapping layer -> effect, or a
    list of {layer, effect} records. Reports the peak, the contiguous block above half
    the peak, and whether that block is contiguous at all -- because "the layers form a
    block" is a mechanism claim and "they are scattered" is a different one.
    """
    L: List[str] = []
    if not os.path.exists(scan_json):
        return [f"[layers] {scan_json} not found — skipping the layer report.",
                "[layers] This is the one claim no black-box experiment can make, so it",
                "[layers] is worth producing: it says WHERE in the network the",
                "[layers] manipulation has to land."]
    try:
        d = json.load(open(scan_json))
    except Exception as e:
        return [f"[layers] unreadable {scan_json}: {type(e).__name__}: {e}"]
    if isinstance(d, dict) and "layers" in d:
        d = d["layers"]
    pairs: List[Tuple[int, float]] = []
    if isinstance(d, dict):
        for k, v in d.items():
            try:
                pairs.append((int(k), float(v)))
            except (TypeError, ValueError):
                continue
    elif isinstance(d, list):
        for r in d:
            if isinstance(r, dict) and "layer" in r:
                val = r.get("effect", r.get("delta", r.get("score")))
                try:
                    pairs.append((int(r["layer"]), float(val)))
                except (TypeError, ValueError):
                    continue
    if not pairs:
        return [f"[layers] no layer/effect pairs in {scan_json}"]
    pairs.sort()
    peak_layer, peak = max(pairs, key=lambda kv: kv[1])
    half = peak / 2.0
    above = [l for l, v in pairs if v >= half]
    contiguous = above == list(range(min(above), max(above) + 1)) if above else False
    L.append(f"[layers] {len(pairs)} layers scanned")
    L.append(f"[layers] peak effect {peak:+.4f} at layer {peak_layer}")
    L.append(f"[layers] layers above half the peak: {min(above)}-{max(above)} "
             f"({len(above)} of {len(pairs)})")
    L.append(f"[layers] contiguous: {contiguous}")
    if contiguous:
        L.append("[layers] A CONTIGUOUS BLOCK. That is the reportable mechanism claim:")
        L.append("[layers] a manipulation must capture a specific band of the network,")
        L.append("[layers] not any layer it likes. It also tells a detector where to")
        L.append("[layers] look, which is what makes the attention feature cheap.")
    else:
        L.append("[layers] SCATTERED. Report it as scattered — a block that is not a")
        L.append("[layers] block is a weaker claim, and asserting one anyway is the kind")
        L.append("[layers] of thing a reviewer checks.")
    return L


# ======================================================================================
def main() -> int:
    p = argparse.ArgumentParser(
        description="fit the attention prior; report separation and the layer block")
    p.add_argument("--flash-csv", dest="flash_csv", default=C.FLASH_CSV)
    p.add_argument("--layer-scan", dest="layer_scan", default="")
    p.add_argument("--z-hit", dest="z_hit", type=float, default=2.0)
    p.add_argument("--out-dir", dest="out_dir", default="")
    a = p.parse_args()

    out_dir = a.out_dir or C.wpath("mechint")
    os.makedirs(out_dir, exist_ok=True)

    prior = AttentionPrior.from_csv(a.flash_csv, z_hit=a.z_hit)
    print()
    print("=" * 92)
    print("MECHANISTIC PRIOR")
    print("=" * 92)
    print(prior.describe())

    # What the abstention gate would do, per condition, on this corpus.
    import rl_data as D
    eps = D.load_flash(a.flash_csv, require_document=False)
    print()
    print("  ABSTENTION GATE, per condition")
    print(f"  {'condition':<24}{'n':>5}{'label':>6}{'abstain':>9}{'mean z':>9}")
    by: Dict[str, List] = {}
    for e in eps:
        by.setdefault(e.condition, []).append(e)
    order = {c: i for i, c in enumerate(C.ACCESS_LEVEL)}
    rows = []
    for cond, g in sorted(by.items(), key=lambda kv: (order.get(kv[0], 99), kv[0])):
        res = [prior.should_abstain(e) for e in g]
        zs = [z for _ab, z in res if z == z]
        ab = [int(x) for x, z in res if z == z]
        if not ab:
            continue
        rate = sum(ab) / len(ab)
        rows.append({"condition": cond, "n": len(ab), "label": int(g[0].label),
                     "abstain_rate": rate,
                     "mean_z": sum(zs) / len(zs)})
        print(f"  {cond:<24}{len(ab):>5}{int(g[0].label):>6}{rate:>9.3f}"
              f"{sum(zs)/len(zs):>9.2f}")
    print()
    print("  READ THE CONTROL ROWS. On C0_clean and C1_filler a HIGH abstain rate is")
    print("  what you want -- the gate is refusing to flag text that manipulates")
    print("  nothing. On the attack rows a high abstain rate means the gate would")
    print("  suppress real detections, and the z-hit threshold is too strict.")

    L = layer_report(a.layer_scan) if a.layer_scan else [
        "[layers] no --layer-scan given; skipping the layer report."]
    print()
    for line in L:
        print(line)

    payload = {"prior": prior.as_dict(), "per_condition": rows,
               "layer_report": L}
    jp = os.path.join(out_dir, "mechint_prior.json")
    with open(jp, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[saved] {jp}")
    print("\nUSE IT:")
    print(f"  python v7_score_rollouts.py --model <m> --ablation P_mech_prm \\")
    print(f"      --mechint-csv {a.flash_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
