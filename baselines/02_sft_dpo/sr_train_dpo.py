#!/usr/bin/env python3
"""
sr_train_dpo.py — the SafeReview defender update (Algorithm 1 line 22, Eq. 5):

    R_t <- DPO(R_{t-1}, D_pref_t; pi_ref = R_{t-1})
    L_DPO = -E log sigmoid( beta [ (log pi(y+|p) - log pi_ref(y+|p))
                                 - (log pi(y-|p) - log pi_ref(y-|p)) ] )

    p   = reviewer prompt + ATTACKED paper     y+ = review of the CLEAN paper
                                               y- = review of the attacked paper

Paper settings (App. A.2): beta 0.1, batch 8, 30 steps per iteration, linear warm-up over
the first 3% of steps, pi_ref reset to R_{t-1} at the start of every iteration. The
defender is the reviewer with a LoRA; the paper fine-tunes DeepReviewer-14B in full with
lr 8e-7, which a LoRA would not move, hence --lr 1e-5 by default (see README).

HAND-WRITTEN, LIKE train_grpo.py, AND FOR THE SAME REASON
    One 40 GiB card holds a 4-bit 27B policy but not a second copy as the reference. The
    reference here IS the policy at step 0 (R_{t-1}: the previous round's adapter, or a
    fresh LoRA, which is the identity), so its log-probs are cached in a no-grad pre-pass
    and the training loop runs one forward per sequence. That is v7's cached-reference
    trick, and it is exact: no approximation of pi_ref is involved.

OUTPUT, READABLE BY YOUR adapter_report.py
    adapters/<model>-<stage>/            the LoRA (+ tokenizer)
        ft_meta.json        stage "dpo_safereview", lr, beta, n_train, ref
        trainer_state.json  log_history with rewards/margins and rewards/accuracies
        dpo_summary.json    the same, summarised; FLAT if the margin never moved
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from typing import Dict, List, Tuple

import sr_config as S
import sr_torch as T


def load_pairs(paths: List[str]) -> List[Dict]:
    out = []
    for p in paths:
        for l in open(p, encoding="utf-8"):
            if l.strip():
                out.append(json.loads(l))
    return out


def prompt_messages(pair: Dict) -> List[Dict]:
    """The paper's pair format (App. G): chosen = [system?, user, assistant]. Everything
    before the assistant turn is the prompt, identical for chosen and rejected."""
    ch = pair["chosen"]
    return [m for m in ch if m["role"] != "assistant"]


def assistant_text(msgs: List[Dict]) -> str:
    return next(m["content"] for m in msgs if m["role"] == "assistant")


def main() -> int:
    ap = argparse.ArgumentParser(description="SafeReview defender: DPO on the reviewer")
    ap.add_argument("--model", required=True, choices=sorted(S.MODELS))
    ap.add_argument("--pairs", nargs="+", required=True, help="pairs.jsonl file(s)")
    ap.add_argument("--stage", required=True, help="adapter stage name: sr_static_dpo | sr_r<t>")
    ap.add_argument("--init-adapter", dest="init_adapter", default="",
                   help="R_{t-1}'s adapter dir ('' = base reviewer, fresh LoRA)")
    ap.add_argument("--out-dir", dest="out_dir", default="")
    ap.add_argument("--steps", type=int, default=S.SR.dpo_steps)
    ap.add_argument("--batch", type=int, default=S.SR.dpo_batch)
    ap.add_argument("--lr", type=float, default=S.SR.dpo_lr)
    ap.add_argument("--beta", type=float, default=S.SR.dpo_beta)
    ap.add_argument("--warmup-ratio", dest="warmup_ratio", type=float,
                    default=S.SR.dpo_warmup_ratio)
    ap.add_argument("--max-seq-len", dest="max_seq_len", type=int,
                    default=S.SR.dpo_max_seq_len)
    ap.add_argument("--seed", type=int, default=S.SR.seed)
    ap.add_argument("--dry-run", dest="dry_run", action="store_true",
                    help="load, tokenise, report, stop before the first forward")
    a = ap.parse_args()

    import torch
    from transformers import get_constant_schedule_with_warmup

    spec = S.MODELS[a.model]
    out_dir = a.out_dir or S.adapter_dir(a.model, a.stage)
    pairs = load_pairs(a.pairs)
    if not pairs:
        raise SystemExit(f"[FATAL] no pairs in {a.pairs}")
    need = a.steps * a.batch
    rng = random.Random(a.seed)
    order = list(range(len(pairs)))
    rng.shuffle(order)
    sched_idx: List[int] = []
    while len(sched_idx) < need:                     # cycle (reshuffled) if pairs < need
        sched_idx += order
        rng.shuffle(order)
    sched_idx = sched_idx[:need]
    used = sorted(set(sched_idx))
    print(f"[data] {len(pairs)} pairs from {len(a.pairs)} file(s); {a.steps} steps x batch "
          f"{a.batch} = {need} examples over {len(used)} distinct pairs"
          + ("  (fewer pairs than examples: cycling)" if len(pairs) < need else ""))

    model, tok = T.load_policy(spec.path, load_4bit=spec.load_4bit,
                               init_adapter=a.init_adapter,
                               lora_cfg=dict(r=S.SR.lora_r, alpha=S.SR.lora_alpha,
                                             dropout=S.SR.lora_dropout,
                                             targets=S.SR.target_modules))
    keep = T.probe_logits_to_keep(model)

    # ---- tokenise every used pair once ----------------------------------------------
    prep: Dict[int, Tuple] = {}
    n_drop = n_trunc = 0
    for i in used:
        pr = pairs[i]
        msgs = prompt_messages(pr)
        p_ids = T.prompt_ids(tok, msgs, spec.chat_kwargs)
        c = T.completion_ids(tok, msgs, assistant_text(pr["chosen"]),
                             chat_kwargs=spec.chat_kwargs, end_of_turn=True)
        r = T.completion_ids(tok, msgs, assistant_text(pr["rejected"]),
                             chat_kwargs=spec.chat_kwargs, end_of_turn=True)
        jc, jr = T.join(p_ids, c, a.max_seq_len), T.join(p_ids, r, a.max_seq_len)
        if jc is None or jr is None:
            n_drop += 1
            continue
        if len(p_ids) + max(len(c), len(r)) > a.max_seq_len:
            n_trunc += 1
        prep[i] = (jc, jr)
    sched_idx = [i for i in sched_idx if i in prep]
    if not sched_idx:
        raise SystemExit("[FATAL] nothing left to train on after tokenisation")
    lens = sorted(len(v[0][0]) for v in prep.values())
    print(f"[tok] {len(prep)} pairs tokenised, {n_drop} dropped, {n_trunc} left-truncated; "
          f"length median {lens[len(lens)//2]}, max {lens[-1]}")
    n_updates = math.ceil(len(sched_idx) / a.batch)
    if a.dry_run:
        print(f"[dry-run] {n_updates} optimiser steps would run. Stopping.")
        return 0

    # ---- PRE-PASS: reference log-probs = the policy as it is now = R_{t-1} -----------
    model.eval()
    ref: Dict[int, Tuple[float, float]] = {}
    t0 = time.time()
    with torch.no_grad():
        for k, (i, (jc, jr)) in enumerate(prep.items()):
            lc = T.seq_logprobs(model, jc[0], jc[1], keep=keep).sum().item()
            lr_ = T.seq_logprobs(model, jr[0], jr[1], keep=keep).sum().item()
            ref[i] = (lc, lr_)
            if (k + 1) % 50 == 0:
                print(f"[ref] {k+1}/{len(prep)}  ({(time.time()-t0)/60:.1f}m)", flush=True)
    print(f"[ref] cached reference log-probs for {len(ref)} pairs "
          f"in {(time.time()-t0)/60:.1f} min (pi_ref = {a.init_adapter or 'base reviewer'})")
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.999), weight_decay=0.0)
    sch = get_constant_schedule_with_warmup(opt, max(1, int(round(a.warmup_ratio * n_updates))))
    hist: List[Dict] = []
    t0 = time.time()
    for step in range(n_updates):
        batch = sched_idx[step * a.batch:(step + 1) * a.batch]
        m_list, acc_list, ch_list, rj_list, l_list = [], [], [], [], []
        opt.zero_grad(set_to_none=True)
        for i in batch:
            (jc, jr), (rc, rr) = prep[i], ref[i]
            # chosen and rejected get SEPARATE backward passes: one 6-8k-token sequence of
            # activations at a time is what fits next to a 4-bit 27B.
            lc = T.seq_logprobs(model, jc[0], jc[1], keep=keep).sum()
            with torch.no_grad():
                lr_det = T.seq_logprobs(model, jr[0], jr[1], keep=keep).sum()
            z = a.beta * ((lc - rc) - (lr_det - rr))
            loss_c = -torch.nn.functional.logsigmoid(z) / len(batch)
            loss_c.backward()
            lr_ = T.seq_logprobs(model, jr[0], jr[1], keep=keep).sum()
            z2 = a.beta * ((lc.detach() - rc) - (lr_ - rr))
            loss_r = -torch.nn.functional.logsigmoid(z2) / len(batch)
            loss_r.backward()
            with torch.no_grad():
                margin = float(z2)
                m_list.append(margin)
                acc_list.append(1.0 if margin > 0 else 0.0)
                ch_list.append(float(a.beta * (lc.detach() - rc)))
                rj_list.append(float(a.beta * (lr_.detach() - rr)))
                l_list.append(float(-torch.nn.functional.logsigmoid(z2)))
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sch.step()
        mean = lambda xs: sum(xs) / max(1, len(xs))
        h = {"step": step + 1, "loss": mean(l_list), "rewards/margins": mean(m_list),
             "rewards/accuracies": mean(acc_list), "rewards/chosen": mean(ch_list),
             "rewards/rejected": mean(rj_list), "learning_rate": sch.get_last_lr()[0]}
        hist.append(h)
        print(f"[dpo] step {step+1}/{n_updates}  loss {h['loss']:.4f}  margin "
              f"{h['rewards/margins']:+.4f}  acc {h['rewards/accuracies']:.2f}  "
              f"chosen {h['rewards/chosen']:+.4f}  rejected {h['rewards/rejected']:+.4f}  "
              f"{(time.time()-t0)/60:.1f}m", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    last = hist[-1]
    meta = {"stage": "dpo_safereview", "row_stage": a.stage, "model": a.model,
            "lr": a.lr, "beta": a.beta, "batch": a.batch, "steps": n_updates,
            "warmup_ratio": a.warmup_ratio, "n_train": len(prep),
            "from_adapter": a.init_adapter or "", "ref_is": a.init_adapter or "base",
            "pairs": [os.path.abspath(p) for p in a.pairs], "seed": a.seed,
            "lora_r": S.SR.lora_r, "max_seq_len": a.max_seq_len}
    with open(os.path.join(out_dir, "ft_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(out_dir, "trainer_state.json"), "w") as f:
        json.dump({"log_history": hist, "global_step": n_updates}, f, indent=2)
    summ = dict(meta, final_margin=last["rewards/margins"],
                final_accuracy=last["rewards/accuracies"], final_loss=last["loss"],
                minutes=(time.time() - t0) / 60)
    with open(os.path.join(out_dir, "dpo_summary.json"), "w") as f:
        json.dump(summ, f, indent=2)
    print("\n" + "=" * 80)
    print(f"DPO COMPLETE — {a.model} / {a.stage}")
    print("=" * 80)
    print(f"  {n_updates} steps, final loss {last['loss']:.4f} (log 2 = 0.693 at step 0)")
    print(f"  reward margin {last['rewards/margins']:+.4f}   accuracy "
          f"{last['rewards/accuracies']:.2f}")
    if abs(last["rewards/margins"]) < 0.05:
        print("  FLAT: the margin barely moved, so the adapter is close to R_{t-1}. Raise "
              "DPO_LR (e.g. 2e-5 -> 5e-5) or DPO_STEPS before evaluating this row.")
    print(f"[saved] {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
