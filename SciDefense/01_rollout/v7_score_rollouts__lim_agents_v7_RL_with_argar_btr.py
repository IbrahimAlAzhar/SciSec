#!/usr/bin/env python3
"""
v7_score_rollouts.py — turn ft_trajectories.py's samples.jsonl into GRPO groups.

WHY THIS FILE EXISTS INSTEAD OF A ROLLOUT SCRIPT
=======================================================================================
v6 shipped its own rollout script. That was a mistake: llim_agents_v4_SFT_DPO already
samples k completions per (document, role) from the served policy, at temperature, with
the SAME prompts the adapters were trained on, and scores every one against gold. That
is a GRPO rollout. Re-implementing it produced a cascade that lacked v5's confidence
gate and reported FPR 0.797 where v5 reports 0.107 -- the numbers were measuring the
re-implementation, not the method.

So v7 generates nothing. It READS what ft_trajectories.py already wrote and adds the two
things GRPO needs on top:

    reward   per sample, from the score ft_roles already computed plus the gold insert
    credit   per agent, from rl_credit -- the part that is actually new

THE GROUP IS ALREADY THERE, AND THIS IS THE KEY POINT
    GRPO's baseline is the mean reward of the OTHER completions OF THE SAME PROMPT.
    ft_trajectories samples k completions per (paper, condition, role) from one prompt
    and writes them with a `k` field. That set IS the group -- same prompt, same policy,
    sampled together at temperature 0.9. Nothing needs to be regenerated.

    For the trajectory-level reward the k-th sample of each role is paired into one
    pseudo-trajectory. That pairing is legitimate because ft_trajectories conditions the
    LEADER on the workers' ACCEPTED findings, so sample k of the leader already saw the
    workers' output for that document.

WHAT COMES OUT
    rollouts.jsonl   one line per (document, group member): reward, step rewards, credit
    turns.jsonl      one line per (document, member, agent): prompt messages + completion
    stats.json       accept rates, reward spread, and the no-signal group fraction

    Byte-identical in shape to what train_grpo.py already reads, so the trainer is
    unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import rl_config as C
import rl_rewards as RW
from rl_credit import Group, assign
from rl_data import Episode, locate_insert
from rl_rewards import AgentStep, Trajectory


# ======================================================================================
# reading what ft_trajectories wrote
# ======================================================================================
def load_samples(path: str) -> List[Dict]:
    if not os.path.exists(path):
        raise SystemExit(
            f"[FATAL] {path} not found.\n"
            f"        v7 does not generate rollouts. Run the V4 rollout stage first:\n"
            f"            cd {C.V4_FT_DIR}\n"
            f"            qsub -v MODEL=<model> job_ft_a_rollout.pbs\n"
            f"        then point --samples at its trajectories/<model>/samples.jsonl")
    out = []
    for line in open(path):
        if line.strip():
            out.append(json.loads(line))
    return out


def load_documents(path: str) -> Dict[Tuple[str, str], Dict]:
    """(paper_id, condition) -> the row ft_attack.py wrote, for the gold insert."""
    import pandas as pd
    if not os.path.exists(path):
        print(f"[warn] {path} not found — span rewards will be unavailable and the "
              f"reward collapses to the decision term.")
        return {}
    df = pd.read_csv(path, low_memory=False)
    out: Dict[Tuple[str, str], Dict] = {}
    for _, r in df.iterrows():
        out[(str(r["paper_id"]), str(r["condition"]))] = {
            "document": str(r.get("document") or ""),
            "insert": str(r.get("insert") or "") if r.get("insert") == r.get("insert") else "",
            "label": int(r["label"]),
            "weaknesses": str(r.get("weaknesses") or ""),
        }
    print(f"[data] {len(out)} (paper, condition) documents from "
          f"{os.path.basename(path)}")
    return out


def episode_for(pid: str, cond: str, docs: Dict, benign: Dict[str, str]) -> Episode:
    d = docs.get((pid, cond), {})
    doc = d.get("document", "")
    ins = d.get("insert", "")
    a, b = locate_insert(doc, ins) if (doc and ins) else (-1, -1)
    return Episode(
        doc_id=f"{pid}|{cond}", paper_id=pid,
        label=int(d.get("label", 0 if cond.endswith("clean") else 1)),
        condition=cond, document=doc, insert=ins, ins_start=a, ins_end=b,
        benign_document=benign.get(pid, ""),
        peer_weaknesses=d.get("weaknesses", ""), access_level="text_only")


# ======================================================================================
# rebuilding a trajectory from the k-th sample of each role
# ======================================================================================
def build_trajectory(per_role: Dict[str, Dict], ep: Episode, k: int) -> Trajectory:
    """
    One pseudo-trajectory: the k-th sample of every role that ran on this document.

    `shown` is taken from what ft_trajectories actually put in the prompt -- the chunk
    for CHUNK_VERIFIER, the limitations region for everyone else. Reconstructing it any
    other way would make the verbatim check disagree with the check that produced the
    sample's own score, and a reward that disagrees with the rollout's own scoring is
    worse than no reward.
    """
    t = Trajectory(doc_id=ep.doc_id, paper_id=ep.paper_id, group_index=k, escalated=1)
    for agent in ("CHUNK_VERIFIER", "INSTRUCTION", "PRESSURE", "VOICE", C.MASTER):
        rec = per_role.get(agent)
        if rec is None:
            continue
        shown = rec.get("chunk") or _region(rec, ep)
        st = RW.fill_step(AgentStep(agent, completion=rec.get("completion", ""),
                                    shown=shown))
        t.steps.append(st)
        t.n_calls += 1
    m = t.master
    if m is not None and m.verdict == "ATTACKED":
        shown = m.shown
        spans = [s for s in m.spans if RW.verbatim(s, shown)]
        # BOTH GATES, the same two v5's detect() applies:
        #   a quotable span, AND a confidence at or above the floor.
        # v6 applied only the first, counted every low-confidence ATTACKED verdict on a
        # clean paper as a flag, and reported FPR 0.797 against v5's 0.107.
        t.pred = 1 if (spans and m.confidence >= C.MIN_CONF) else 0
        t.pred_spans = spans[:6]
        o = m.obj or {}
        t.attack_type = str(o.get("attack_type") or "OTHER")[:40]
    return t


def _messages(system: str, user: str, *, system_role: bool) -> List[Dict[str, str]]:
    """
    Chat messages from the system/user strings ft_trajectories already recorded.

    Deliberately NOT routed through rl_prompts. The prompts are already in samples.jsonl
    -- ft_trajectories wrote the exact strings it sent -- so re-deriving them would mean
    importing the whole V4 prompt module for three lines, and would introduce the
    possibility of rebuilding a prompt that differs from the one the sample was drawn
    from. Reading what was recorded cannot drift.

    system_role=False merges the system text into the user turn: gemma's chat template
    has no system role and apply_chat_template RAISES on one.
    """
    if system_role:
        return [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    return [{"role": "user", "content": (system + "\n\n" + user).strip()}]


def _region(rec: Dict, ep: Episode) -> str:
    d = rec.get("document") or ep.document or ""
    return d if len(d) <= 6000 else d[-6000:]


# ======================================================================================
def main() -> int:
    p = argparse.ArgumentParser(
        description="score ft_trajectories samples into GRPO groups")
    p.add_argument("--model", required=True, choices=sorted(C.MODELS))
    p.add_argument("--ablation", required=True, choices=sorted(C.ABL))
    p.add_argument("--samples", default="",
                   help="trajectories/<model>/samples.jsonl written by "
                        "llim_agents_v4_SFT_DPO/ft_trajectories.py")
    p.add_argument("--documents", default="",
                   help="traindata_shared/train_documents.csv from the same run")
    p.add_argument("--out-dir", dest="out_dir", default="")
    p.add_argument("--iteration", type=int, default=0)
    p.add_argument("--reward", default="", help="override the ablation's reward kind")
    p.add_argument("--credit", default="", help="override the ablation's credit rule")
    p.add_argument("--mechint-csv", dest="mechint_csv", default="",
                   help="the FLASH fusion CSV. When given, the attention footprint adds "
                        "a MECHANISTIC term to the worker step rewards (see v7_mechint).")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    mk = a.model.replace("/", "_")
    ab = C.ABL[a.ablation]
    reward_kind = a.reward or ab.reward or "provenance"
    credit_rule = a.credit or ab.credit or "uniform"

    samples_path = a.samples or os.path.join(
        C.V4_FT_DIR, "trajectories", a.model, "samples.jsonl")
    docs_path = a.documents or os.path.join(
        C.V4_FT_DIR, "traindata_shared", "train_documents.csv")
    out_dir = a.out_dir or C.wpath(C.D_TRAJ, mk, f"{a.ablation}_iter{a.iteration}")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 92)
    print(f"SCORE ROLLOUTS — {a.model} / {a.ablation}  iter={a.iteration}")
    print("=" * 92)
    print(f"  samples    {samples_path}")
    print(f"  documents  {docs_path}")
    print(f"  reward={reward_kind}  credit={credit_rule}")
    print("  NOTE: nothing is generated here. These are the samples "
          "llim_agents_v4_SFT_DPO")
    print("  already drew from the served policy, with the prompts the adapters were")
    print("  trained on.")

    samples = load_samples(samples_path)
    docs = load_documents(docs_path)
    print(f"[data] {len(samples)} samples")

    mech = None
    if a.mechint_csv:
        import v7_mechint as MI
        mech = MI.AttentionPrior.from_csv(a.mechint_csv)
        print(mech.describe())

    # benign document per paper, for the excision arm later
    benign: Dict[str, str] = {}
    for (pid, cond), d in docs.items():
        if d["label"] == 0:
            benign.setdefault(pid, d["document"])

    # ---- index: (paper, condition, k) -> {role: sample} -----------------------------
    by_doc: Dict[Tuple[str, str], Dict[int, Dict[str, Dict]]] = defaultdict(
        lambda: defaultdict(dict))
    n_k = 0
    for s in samples:
        pid, cond, role = str(s["paper_id"]), str(s["condition"]), str(s["role"])
        k = int(s.get("k", 0))
        n_k = max(n_k, k + 1)
        # CHUNK_VERIFIER runs on several chunks per document; keep the first, because a
        # trajectory has one verifier turn and the reward is defined on a trajectory.
        if role == "CHUNK_VERIFIER" and role in by_doc[(pid, cond)][k]:
            continue
        by_doc[(pid, cond)][k][role] = s
    print(f"[data] {len(by_doc)} documents, group size k={n_k}")
    if n_k < 2:
        raise SystemExit(
            "[FATAL] only one sample per prompt. GRPO's baseline is the mean of the "
            "OTHER completions of the same prompt, so a group of one has zero advantage "
            "by construction. Re-run the V4 rollout with --n-samples 4 or more.")

    f_roll = open(os.path.join(out_dir, "rollouts.jsonl"), "w", buffering=1)
    f_turn = open(os.path.join(out_dir, "turns.jsonl"), "w", buffering=1)
    stats = {"n_groups": 0, "n_traj": 0, "n_dropped": 0, "reward_sum": 0.0,
             "tp": 0, "fp": 0, "tn": 0, "fn": 0, "n_mech": 0}

    for (pid, cond), per_k in sorted(by_doc.items()):
        ep = episode_for(pid, cond, docs, benign)
        if not ep.document.strip():
            continue
        g = Group(ep.doc_id, pid, int(ep.label), cond)
        trajs: List[Trajectory] = []
        for k in sorted(per_k):
            t = build_trajectory(per_k[k], ep, k)
            if not t.steps:
                continue
            r, bd, steps = RW.score_trajectory(t, ep, reward_kind)
            if mech is not None:
                # THE MECHANISTIC TERM. Pays a worker for quoting the span that actually
                # captured the reviewer's attention, not the span that merely reads like
                # an instruction. See v7_mechint for why that distinction is the one the
                # diagnosis says is missing.
                bonus = mech.step_bonus(t, ep)
                if bonus:
                    stats["n_mech"] += 1
                    for agent, v in bonus.items():
                        steps[agent] = steps.get(agent, 0.0) + v
                    r += sum(bonus.values()) / max(1, len(bonus))
            trajs.append(t)
            g.trajectories.append(t)
            g.rewards.append(r)
            g.step_rewards.append(steps)
        if len(g) < 2:
            continue

        cr = assign(g, credit_rule, episodes={ep.doc_id: ep})
        stats["n_groups"] += 1
        stats["n_dropped"] += int(cr.dropped)
        for i, t in enumerate(g.trajectories):
            stats["n_traj"] += 1
            stats["reward_sum"] += g.rewards[i]
            key = ("tp" if (ep.label == 1 and t.pred == 1) else
                   "fp" if (ep.label == 0 and t.pred == 1) else
                   "tn" if (ep.label == 0 and t.pred == 0) else "fn")
            stats[key] += 1
            rec = t.to_json(with_completions=True)
            rec.update({"model": a.model, "ablation": a.ablation,
                        "iteration": a.iteration, "label": ep.label,
                        "condition": cond, "reward": g.rewards[i],
                        "step_rewards": g.step_rewards[i], "credit_rule": cr.rule,
                        "credit_inter": {ag: cr.inter[ag][i] for ag in cr.inter},
                        "credit_intra": {ag: cr.intra[ag][i] for ag in cr.intra},
                        "credit_weight_intra": cr.weight_intra,
                        "group_dropped": int(cr.dropped)})
            f_roll.write(json.dumps(rec, ensure_ascii=False) + "\n")
            for agent, s in ((st.agent, per_k[t.group_index].get(st.agent))
                             for st in t.steps):
                if s is None:
                    continue
                spec = C.MODELS[a.model]
                msgs = _messages(s.get("system", ""), s.get("user", ""),
                                 system_role=spec.system_role)
                f_turn.write(json.dumps(
                    {"doc_id": ep.doc_id, "paper_id": pid,
                     "group_index": t.group_index, "agent": agent,
                     "messages": msgs, "completion": s.get("completion", "")},
                    ensure_ascii=False) + "\n")

    f_roll.close()
    f_turn.close()

    tp, fp, tn, fn = stats["tp"], stats["fp"], stats["tn"], stats["fn"]
    stats.update({
        "mean_reward": stats["reward_sum"] / max(1, stats["n_traj"]),
        "dropped_frac": stats["n_dropped"] / max(1, stats["n_groups"]),
        "recall": tp / (tp + fn) if tp + fn else None,
        "fpr": fp / (fp + tn) if fp + tn else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "reward_kind": reward_kind, "credit_rule": credit_rule,
        "group_size": n_k, "min_conf": C.MIN_CONF,
        "mechint": bool(mech)})
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    def _f(x):
        return "-" if x is None else f"{x:.3f}"

    print()
    print("=" * 92)
    print("SCORED")
    print("=" * 92)
    print(f"  {stats['n_groups']} groups, {stats['n_traj']} trajectories")
    print(f"  mean reward {stats['mean_reward']:+.3f}")
    print(f"  sampled: recall {_f(stats['recall'])}  FPR {_f(stats['fpr'])}  "
          f"precision {_f(stats['precision'])}   (min_conf {C.MIN_CONF})")
    print(f"  groups with NO reward variance: {stats['n_dropped']} "
          f"({100*stats['dropped_frac']:.1f}%)")
    if mech is not None:
        print(f"  mechanistic bonus applied to {stats['n_mech']} trajectories")
    if (stats["fpr"] or 0) > 0.4:
        print()
        print("  THE SAMPLED POLICY STILL FLAGS MOST CLEAN PAPERS.")
        print("  GRPO re-weights behaviours that APPEAR in the samples. With almost no")
        print("  CLEAN completions in the group there is nothing for the advantage to")
        print("  push toward, and no number of steps will teach restraint from this")
        print("  data. Raise C.MIN_CONF, or add clean documents to the rollout corpus,")
        print("  before spending GPU on training.")
    if stats["dropped_frac"] > 0.4:
        print()
        print("  OVER 40% OF GROUPS CARRY NO GRADIENT. Either the rollout temperature")
        print("  was too low, or the reward is too coarse for these episodes.")
    print(f"\n[saved] {out_dir}/rollouts.jsonl")
    print(f"[saved] {out_dir}/turns.jsonl")
    print(f"[saved] {out_dir}/stats.json")
    print(f"\nNEXT: python train_grpo.py --model {a.model} --ablation {a.ablation} \\")
    print(f"          --rollout-dir {out_dir} --init-adapter "
          f"{C.adapter_dir(a.model, 'sft')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
