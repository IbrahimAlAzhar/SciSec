#!/usr/bin/env python3
"""
rl_diagnose.py — why does a row look like that? Answered from files you already have.

    python rl_diagnose.py --model gemma3-27b-it
    python rl_diagnose.py --model gemma3-27b-it --a A_zeroshot_v5 --b G_grpo_prov

CPU only, seconds, no GPU and no server. It reads stage1_detail.csv, the rollout
stats.json and grpo_summary.json, and answers the three questions a bad ablation table
raises, in the order they are worth asking:

    1. IS THE BASELINE ROW EVEN RIGHT?
       If the zero-shot row does not reproduce the number you published, nothing below it
       is interpretable, and every hour spent on the RL rows is wasted. This is checked
       FIRST and everything else is reported as secondary.

    2. WHERE DO THE FALSE POSITIVES COME FROM?
       A flag can be raised in two completely different places, and they need opposite
       fixes:
          tier-0 auto-flag   the retrieval screen scored above the high gate and the
                             document was never shown to an agent (escalated == 0,
                             pred == 1). Fix: recalibrate the gate. Costs nothing.
          leader flag        the cascade ran and the leader said ATTACKED with a
                             quotable span. Fix: the policy, which is what RL is for.
       Reporting a single FPR hides which one you have.

    3. DID THE ADAPTER CHANGE ANYTHING AT ALL?
       Two rows that differ by 0.001 on every metric are usually not a weak improvement;
       they are the same model answering twice. Predictions are compared DOCUMENT BY
       DOCUMENT, because aggregate agreement can hide two errors cancelling out.

WHAT IT DELIBERATELY DOES NOT DO
    It does not tune anything. It reports what threshold WOULD have held the
    false-positive rate at a budget, on the split you point it at, and then tells you to
    fix it on val and re-run on test. A threshold chosen by looking at the test set is
    not a result.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import rl_config as C


def _p(x, n: int = 3) -> str:
    if x is None:
        return "-"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "-"
    return "-" if v != v else f"{v:.{n}f}"


def load_detail(work: str, model: str, ablation: str):
    import pandas as pd
    mk = model.replace("/", "_")
    p = os.path.join(work, C.D_RUNS, mk, ablation, "stage1_detail.csv")
    if not os.path.exists(p):
        return None, p
    return pd.read_csv(p, low_memory=False), p


def confusion(df) -> Dict:
    tp = int(((df.label == 1) & (df.pred == 1)).sum())
    fp = int(((df.label == 0) & (df.pred == 1)).sum())
    tn = int(((df.label == 0) & (df.pred == 0)).sum())
    fn = int(((df.label == 1) & (df.pred == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": prec, "recall": rec,
            "f1": (2 * prec * rec / (prec + rec)) if (prec and rec) else None,
            "fpr": fp / (fp + tn) if fp + tn else None,
            "flag_rate": (tp + fp) / max(1, len(df))}


# ======================================================================================
# 1. the baseline check
# ======================================================================================
def check_baseline(df, published_f1: float, published_fpr: float) -> List[str]:
    out: List[str] = []
    m = confusion(df)
    out.append("")
    out.append("=" * 92)
    out.append("1. DOES THE ZERO-SHOT ROW REPRODUCE YOUR PUBLISHED BASELINE?")
    out.append("=" * 92)
    out.append(f"  {'':18}{'published':>12}{'this run':>12}{'':>6}")
    for name, pub, got in (("F1", published_f1, m["f1"]),
                           ("FPR", published_fpr, m["fpr"])):
        flag = ""
        if got is not None and pub:
            r = got / pub
            if name == "FPR" and r > 1.5:
                flag = f"  {r:.1f}x WORSE"
            elif name == "F1" and r < 0.9:
                flag = f"  {r:.2f}x"
        out.append(f"  {name:18}{pub:>12.3f}{_p(got):>12}{flag}")
    out.append(f"  {'flag rate':18}{'':>12}{_p(m['flag_rate']):>12}"
               + ("   <-- flags almost everything" if (m["flag_rate"] or 0) > 0.8 else ""))
    if m["fpr"] is not None and m["fpr"] > published_fpr * 1.5:
        out.append("")
        out.append("  THE BASELINE IS NOT REPRODUCING. Stop here.")
        out.append("  Every RL row is measured against this one, so until it matches, a")
        out.append("  difference between rows cannot be attributed to the method. The")
        out.append("  v6 cascade was a REIMPLEMENTATION. v7 does not have one: it routes each")
        out.append("  agent to its own adapter, which runner_def.py cannot do -- so it")
        out.append("  does NOT inherit v5's tier-0 calibration, repair probe or span")
        out.append("  fallback. Section 2 says which of those is costing you.")
    return out


# ======================================================================================
# 2. where the false positives are raised
# ======================================================================================
def fp_decomposition(df) -> List[str]:
    out: List[str] = []
    out.append("")
    out.append("=" * 92)
    out.append("2. WHERE ARE THE FALSE POSITIVES RAISED?")
    out.append("=" * 92)
    ben = df[df.label == 0]
    fps = ben[ben.pred == 1]
    if not len(ben):
        out.append("  no benign documents in this file")
        return out
    if "escalated" not in df.columns:
        out.append("  no `escalated` column — cannot separate tier-0 from leader flags")
        return out
    tier0 = fps[fps.escalated == 0]
    leader = fps[fps.escalated == 1]
    out.append(f"  benign documents            {len(ben)}")
    out.append(f"  false positives             {len(fps)}   "
               f"(FPR {len(fps)/len(ben):.3f})")
    out.append(f"    raised by the TIER-0 SCREEN  {len(tier0):>5}  "
               f"({len(tier0)/max(1,len(fps)):.0%} of all FPs)")
    out.append(f"      -- never shown to an agent; the screen scored above the high gate")
    out.append(f"    raised by the LEADER         {len(leader):>5}  "
               f"({len(leader)/max(1,len(fps)):.0%} of all FPs)")
    out.append(f"      -- the cascade ran and the leader flagged with a quotable span")
    out.append("")
    if len(tier0) > len(leader):
        out.append("  MOST OF YOUR FALSE POSITIVES NEVER REACHED AN AGENT.")
        out.append("  That is a THRESHOLD problem, not a policy problem, and no amount of")
        out.append("  RL can fix it: the documents those flags came from were never shown")
        out.append("  to the policy being trained. Recalibrate the gate (section 3).")
    else:
        out.append("  Most false positives come from the LEADER, which is the part RL can")
        out.append("  actually move. The reward's w_false_positive term is the knob.")

    # what the leader does once it IS escalated
    esc = df[df.escalated == 1]
    if len(esc):
        m = confusion(esc)
        out.append("")
        out.append(f"  On the {len(esc)} ESCALATED documents alone:")
        out.append(f"    recall {_p(m['recall'])}   FPR {_p(m['fpr'])}   "
                   f"precision {_p(m['precision'])}")
        out.append("    This is the cascade's own performance with the screen's decisions")
        out.append("    taken out. Compare it against the pooled number above.")
    if "leader_parsed" in df.columns and len(esc):
        pr = esc["leader_parsed"].mean()
        out.append("")
        out.append(f"  leader JSON parsed on {pr:.1%} of escalated documents")
        if pr < 0.9:
            out.append("    BELOW 90%. A completion that does not parse is scored as a")
            out.append("    non-answer, so the leader looks like it cannot detect when the")
            out.append("    real problem is format. SFT fixes this in the first epoch.")
    return out


# ======================================================================================
# 3. screen calibration
# ======================================================================================
def screen_calibration(df, budget: float = 0.10) -> List[str]:
    out: List[str] = []
    out.append("")
    out.append("=" * 92)
    out.append(f"3. WHAT WOULD THE TIER-0 GATES HAVE TO BE? (budget: FPR <= {budget:.0%})")
    out.append("=" * 92)
    if "screen_score" not in df.columns or df["screen_score"].notna().sum() < 20:
        out.append("  no usable screen_score column — the tier-0 memory did not load, so")
        out.append("  every document escalated and the LEADER prompt said 'not available'")
        out.append("  where it said a number during training. That alone changes the")
        out.append("  distribution the adapters were fitted to.")
        return out
    d = df[df["screen_score"].notna()]
    ben = d[d.label == 0]["screen_score"].values
    att = d[d.label == 1]["screen_score"].values
    if not len(ben) or not len(att):
        out.append("  need both classes")
        return out
    import numpy as np
    ben, att = np.asarray(ben, float), np.asarray(att, float)
    out.append(f"  screen score      {'benign':>12}{'attacked':>12}")
    for q, lab in ((50, "median"), (75, "p75"), (90, "p90"), (95, "p95"), (99, "p99")):
        out.append(f"    {lab:<14}{np.percentile(ben, q):>12.3f}"
                   f"{np.percentile(att, q):>12.3f}")
    out.append(f"    {'max':<14}{ben.max():>12.3f}{att.max():>12.3f}")
    out.append("")
    out.append(f"  CURRENT GATES (hardcoded): low -0.15   high 0.35")
    cur_hi = float((ben >= 0.35).mean())
    out.append(f"    at high=0.35, {cur_hi:.1%} of BENIGN documents are auto-flagged "
               f"with no agent call")
    if cur_hi > budget:
        need_hi = float(np.percentile(ben, 100 * (1 - budget)))
        rec_at = float((att >= need_hi).mean())
        out.append("")
        out.append(f"  To hold the tier-0 false-alarm rate at {budget:.0%}, the high gate")
        out.append(f"  would have to move from 0.35 to about {need_hi:+.3f}.")
        out.append(f"    it would then auto-flag {rec_at:.1%} of attacked documents")
        out.append("")
        out.append("  DO NOT TUNE THIS ON THE TEST SPLIT. Pick it on val, freeze it,")
        out.append("  pick the gate there, freeze it, and re-run on test. A gate chosen by")
        out.append("  looking at test is not a result, and a reviewer will ask.")
        out.append("")
        out.append("  Then pass it through:")
        out.append(f"    --escalate-high {need_hi:.3f}")
    else:
        out.append(f"    that is within the {budget:.0%} budget; the screen is not the "
                   f"problem")
    return out


# ======================================================================================
# 4. did the adapter change anything
# ======================================================================================
def compare_rows(da, db, name_a: str, name_b: str) -> List[str]:
    out: List[str] = []
    out.append("")
    out.append("=" * 92)
    out.append(f"4. DID THE ADAPTER CHANGE ANYTHING?   {name_a}  vs  {name_b}")
    out.append("=" * 92)
    if da is None or db is None:
        out.append("  one of the two runs is missing")
        return out
    m = da[["doc_id", "pred"]].merge(db[["doc_id", "pred"]], on="doc_id",
                                     suffixes=("_a", "_b"))
    if not len(m):
        out.append("  the two runs share no doc_id — different corpora")
        return out
    diff = int((m.pred_a != m.pred_b).sum())
    out.append(f"  documents in both runs      {len(m)}")
    out.append(f"  predictions that DIFFER     {diff}   ({diff/len(m):.2%})")
    if "n_llm_calls" in da.columns and "n_llm_calls" in db.columns:
        mc = da[["doc_id", "n_llm_calls"]].merge(
            db[["doc_id", "n_llm_calls"]], on="doc_id", suffixes=("_a", "_b"))
        cdiff = int((mc.n_llm_calls_a != mc.n_llm_calls_b).sum())
        out.append(f"  call counts that DIFFER     {cdiff}   ({cdiff/len(mc):.2%})")
    if "span" in da.columns and "span" in db.columns:
        ms = da[["doc_id", "span"]].merge(db[["doc_id", "span"]], on="doc_id",
                                          suffixes=("_a", "_b"))
        ms = ms.fillna("")
        sdiff = int((ms.span_a.astype(str) != ms.span_b.astype(str)).sum())
        out.append(f"  quoted spans that DIFFER    {sdiff}   ({sdiff/len(ms):.2%})")
    out.append("")
    if diff / len(m) < 0.02:
        out.append("  UNDER 2% OF DECISIONS CHANGED. This is not a weak improvement, it")
        out.append("  is almost certainly the SAME MODEL answering twice. Three causes,")
        out.append("  cheapest to check first:")
        out.append("    a) the adapter was not applied. rl_infer prints the served name")
        out.append("       per agent -- check it is the LoRA name, not the base path.")
        out.append("    b) the adapter is mathematically the identity. PEFT starts lora_B")
        out.append("       at ZERO, so an adapter that never trained serves and changes")
        out.append("       nothing. Section 5 reads the KL.")
        out.append("    c) it trained on too few distinct papers to generalise.")
    else:
        out.append(f"  {diff/len(m):.1%} of decisions moved, so the adapter IS doing")
        out.append("  something. Whether it is doing something GOOD is the FPR column.")
    return out


# ======================================================================================
# 5. the training run itself
# ======================================================================================
def training_health(work: str, model: str) -> List[str]:
    out: List[str] = []
    out.append("")
    out.append("=" * 92)
    out.append("5. WAS THE TRAINING RUN HEALTHY?")
    out.append("=" * 92)
    mk = model.replace("/", "_")
    found = False
    for p in sorted(glob.glob(os.path.join(work, C.D_TRAJ, mk, "*", "stats.json"))):
        try:
            s = json.load(open(p))
        except Exception:
            continue
        found = True
        tag = os.path.basename(os.path.dirname(p))
        out.append(f"  rollout {tag}")
        out.append(f"    groups {s.get('n_groups')}   trajectories {s.get('n_traj')}   "
                   f"mean reward {_p(s.get('mean_reward'))}")
        dr = s.get("dropped_frac")
        out.append(f"    groups with NO reward variance  {_p(dr)}"
                   + ("   <-- most groups carry no gradient" if (dr or 0) > 0.4 else ""))
        out.append(f"    sampled recall {_p(s.get('recall'))}  FPR {_p(s.get('fpr'))}")
        if (s.get("fpr") or 0) > 0.5:
            out.append("    THE POLICY ALREADY FLAGGED EVERYTHING DURING THE ROLLOUT.")
            out.append("    GRPO can only re-weight behaviours that APPEAR in the")
            out.append("    samples. If almost every rollout says ATTACKED, there are")
            out.append("    almost no CLEAN completions to prefer, the advantage has")
            out.append("    nothing to push toward, and the policy cannot learn restraint")
            out.append("    from this data no matter how many steps you run.")
    for p in sorted(glob.glob(os.path.join(work, C.D_ADAPTERS, "*", "grpo_summary.json"))):
        try:
            s = json.load(open(p))
        except Exception:
            continue
        found = True
        out.append(f"  adapter {os.path.basename(os.path.dirname(p))}")
        out.append(f"    steps {s.get('n_updates')}   mean KL {_p(s.get('mean_kl'), 5)}   "
                   f"mean rho {_p(s.get('mean_ratio'))}   clip {_p(s.get('mean_clipfrac'))}")
        kl = s.get("mean_kl")
        if kl is not None and kl == kl and kl < 1e-4:
            out.append("    KL IS ESSENTIALLY ZERO -- the policy did not move. Either the")
            out.append("    learning rate is too low, or every advantage was ~0 because")
            out.append("    the groups had no reward variance.")
    if not found:
        out.append("  no rollout stats.json or grpo_summary.json found under "
                   f"{os.path.join(work, C.D_TRAJ, mk)}")
    return out


# ======================================================================================
# 6. per condition, with the control
# ======================================================================================
def per_condition(df) -> List[str]:
    out: List[str] = []
    out.append("")
    out.append("=" * 92)
    out.append("6. FLAG RATE PER CONDITION")
    out.append("=" * 92)
    out.append(f"  {'condition':<26}{'access':<12}{'n':>5}{'label':>6}{'flagged':>9}"
               f"{'rate':>8}  ")
    order = {c: i for i, c in enumerate(C.ACCESS_LEVEL)}
    for cond, g in sorted(df.groupby("condition"),
                          key=lambda kv: (order.get(kv[0], 99), kv[0])):
        lab = int(g.label.iloc[0])
        rate = float(g.pred.mean())
        note = ""
        if lab == 0:
            note = "FALSE ALARMS" if rate > 0.2 else ""
        out.append(f"  {str(cond):<26}{str(g.get('access', ['?']).iloc[0]):<12}"
                   f"{len(g):>5}{lab:>6}{int(g.pred.sum()):>9}{rate:>8.3f}  {note}")
    ctl = df[df.get("is_control", 0) == 1] if "is_control" in df.columns else df.iloc[0:0]
    if len(ctl):
        out.append("")
        out.append(f"  FILLER CONTROL: flagged {ctl.pred.mean():.3f} of {len(ctl)} "
                   f"length-matched neutral inserts.")
        out.append("  Every one of those is a false alarm on text that manipulates")
        out.append("  nothing. If this rate is near the attacked recall, the detector is")
        out.append("  finding 'a block of text was added here', not an attack.")
    return out


# ======================================================================================
def main() -> int:
    p = argparse.ArgumentParser(description="diagnose an ablation table")
    p.add_argument("--model", required=True, choices=sorted(C.MODELS))
    p.add_argument("--work", default=C.WORK)
    p.add_argument("--a", default="A_zeroshot_v5", help="the baseline row")
    p.add_argument("--b", default="", help="the row to compare against it")
    p.add_argument("--published-f1", dest="pf1", type=float, default=0.807)
    p.add_argument("--published-fpr", dest="pfpr", type=float, default=0.107)
    p.add_argument("--fpr-budget", dest="budget", type=float, default=0.10)
    p.add_argument("--out", default="")
    a = p.parse_args()

    da, pa = load_detail(a.work, a.model, a.a)
    if da is None:
        raise SystemExit(
            f"[FATAL] {pa} not found.\n"
            f"  A MEASURED row comes from stage C, which runs v5's own "
            f"runner_def.py\n  with --vllm-model <adapter> and --out-dir that "
            f"directory.\n"
            f"  An IMPORTED row is read from C.BASELINE_RUNS and is never re-run.")
    b_key = a.b
    if not b_key:
        mk = a.model.replace("/", "_")
        cands = [os.path.basename(d) for d in
                 sorted(glob.glob(os.path.join(a.work, C.D_RUNS, mk, "*")))
                 if os.path.basename(d) != a.a
                 and os.path.exists(os.path.join(d, "stage1_detail.csv"))]
        b_key = cands[0] if cands else ""
    db, pb = (load_detail(a.work, a.model, b_key) if b_key else (None, ""))

    L: List[str] = []
    L.append("=" * 92)
    L.append(f"DIAGNOSIS — {a.model}")
    L.append("=" * 92)
    L.append(f"  baseline row  {a.a}   ({len(da)} documents)")
    if db is not None:
        L.append(f"  compared row  {b_key}   ({len(db)} documents)")

    L += check_baseline(da, a.pf1, a.pfpr)
    L += fp_decomposition(da)
    L += screen_calibration(da, a.budget)
    if db is not None:
        L += compare_rows(da, db, a.a, b_key)
    L += training_health(a.work, a.model)
    L += per_condition(da)

    L.append("")
    L.append("=" * 92)
    L.append("WHAT TO DO, IN ORDER")
    L.append("=" * 92)
    L.append("  Fix section 1 before anything else. An RL row compared against a broken")
    L.append("  baseline tells you nothing, however many GPU hours it cost.")

    txt = "\n".join(L)
    print(txt)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        open(a.out, "w").write(txt + "\n")
        print(f"\n[saved] {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
