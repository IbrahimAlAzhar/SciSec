#!/usr/bin/env python3
"""
sr_train_attacker.py — the SafeReview attacker update (Algorithm 1 line 15, Eq. 2-4):

    J_GRPO = E [ 1/G sum_g ( min(rho A, clip(rho, 1-eps, 1+eps) A) - beta D_KL(A_t || A_ref) ) ]
    D_KL   = pi_ref/pi - log(pi_ref/pi) - 1          (Eq. 3, the k3 estimator)
    A      = (r - mean_g r) / std_g r                (Eq. 4, computed by sr_attacker.py score)

"At iteration t, both the old policy and the KL reference A_ref_{t-1} are set to A_{t-1}."
The rollout was sampled from A_{t-1} and this script starts from A_{t-1}, so ONE cached
no-grad pass gives pi_old and pi_ref together -- exactly v7 train_grpo's design, for the
same memory reason, and exact rather than approximate.

Differences from the paper, all forced by one 40 GiB card or by LoRA, and listed in the
README: offline rollout->update cycles instead of online GRPO; a LoRA instead of full
fine-tuning (lr 5e-6 instead of 5e-7); token-level ratios (Eq. 2 writes a sequence-level
ratio; OpenRLHF, which the paper uses, applies it per token).

Writes adapters_attacker/<attacker>-sratk-<reviewer>-r<t>/ with grpo_summary.json, which
your adapter_report.py reads (mean_kl and the INERT / USABLE / DIVERGED bands).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict

import sr_config as S
import sr_torch as T


def main() -> int:
    ap = argparse.ArgumentParser(description="SafeReview attacker: offline GRPO")
    ap.add_argument("--model", required=True, help="the reviewer this attacker targets")
    ap.add_argument("--attacker", default=S.DEFAULT_ATTACKER, choices=sorted(S.ATTACKERS))
    ap.add_argument("--scored", nargs="+", required=True, help="scored.jsonl file(s)")
    ap.add_argument("--init-adapter", dest="init_adapter", default="",
                    help="A_{t-1} ('' = the base attacker, fresh LoRA)")
    ap.add_argument("--out-dir", dest="out_dir", required=True)
    ap.add_argument("--lr", type=float, default=S.SR.atk_lr)
    ap.add_argument("--beta-kl", dest="beta_kl", type=float, default=S.SR.atk_beta_kl)
    ap.add_argument("--clip-eps", dest="clip_eps", type=float, default=S.SR.atk_clip_eps)
    ap.add_argument("--grad-accum", dest="grad_accum", type=int, default=S.SR.atk_grad_accum)
    ap.add_argument("--epochs", type=int, default=S.SR.atk_epochs)
    ap.add_argument("--max-seq-len", dest="max_seq_len", type=int, default=4096)
    ap.add_argument("--kl-max", dest="kl_max", type=float, default=S.SR.atk_kl_max)
    ap.add_argument("--kl-clamp", dest="kl_clamp", type=float, default=S.SR.atk_kl_clamp)
    ap.add_argument("--seed", type=int, default=S.SR.seed)
    ap.add_argument("--dry-run", dest="dry_run", action="store_true")
    a = ap.parse_args()

    import torch
    from transformers import get_constant_schedule_with_warmup

    spec = S.ATTACKERS[a.attacker]
    rows = []
    for p in a.scored:
        rows += [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    ex = [r for r in rows if not r.get("group_dropped") and abs(r.get("advantage", 0)) > 1e-9
          and str(r.get("completion") or "").strip()]
    print(f"[data] {len(rows)} rollouts, {len(ex)} carry a non-zero advantage "
          f"({sum(1 for r in rows if r.get('group_dropped'))} in dropped groups)")
    if not ex:
        raise SystemExit("[FATAL] no rollout carries gradient. Every group had identical "
                         "rewards: raise ATK_TEMP or use --n-reviews with a review temperature.")

    model, tok = T.load_policy(spec.path, load_4bit=spec.load_4bit,
                               init_adapter=a.init_adapter,
                               lora_cfg=dict(r=S.SR.lora_r, alpha=S.SR.lora_alpha,
                                             dropout=S.SR.lora_dropout,
                                             targets=S.SR.target_modules))
    keep = T.probe_logits_to_keep(model)
    prep = []
    for r in ex:
        p_ids = T.prompt_ids(tok, r["messages"], spec.chat_kwargs)
        c_ids = T.completion_ids(tok, r["messages"], r["completion"], end_of_turn=False)
        j = T.join(p_ids, c_ids, a.max_seq_len)
        if j is not None:
            prep.append((r, j[0], j[1]))
    print(f"[tok] {len(prep)} examples; median length "
          f"{sorted(len(x[1]) for x in prep)[len(prep)//2]}")
    n_updates = max(1, (len(prep) * a.epochs) // a.grad_accum)
    if a.dry_run:
        print(f"[dry-run] {n_updates} optimiser steps would run. Stopping.")
        return 0

    # pi_old = pi_ref = A_{t-1}: the policy as it is right now
    model.eval()
    lp0 = []
    t0 = time.time()
    with torch.no_grad():
        for k, (_r, ids, cl) in enumerate(prep):
            lp0.append(T.seq_logprobs(model, ids, cl, keep=keep).detach().float().cpu())
            if (k + 1) % 100 == 0:
                print(f"[ref] {k+1}/{len(prep)} ({(time.time()-t0)/60:.1f}m)", flush=True)
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.999), weight_decay=0.01)
    sch = get_constant_schedule_with_warmup(opt, max(1, int(0.03 * n_updates)))
    order = [j for _ in range(a.epochs) for j in range(len(prep))]
    random.Random(a.seed).shuffle(order)
    hist = defaultdict(list)
    acc = step = 0
    opt.zero_grad(set_to_none=True)
    t0 = time.time()
    dev = next(model.parameters()).device
    for n, j in enumerate(order):
        r, ids, cl = prep[j]
        ref = lp0[j].to(dev)
        lp = T.seq_logprobs(model, ids, cl, keep=keep)
        ratio = torch.exp(lp - ref)
        A = torch.tensor(float(r["advantage"]), device=dev, dtype=ratio.dtype)
        pg = -torch.min(ratio * A, torch.clamp(ratio, 1 - a.clip_eps, 1 + a.clip_eps) * A).mean()
        d = ref - lp                                       # log(pi_ref / pi)
        with torch.no_grad():
            k3_all = torch.exp(d) - d - 1.0
        d_pen = d.clamp(max=a.kl_clamp) if a.kl_clamp > 0 else d
        kl = (torch.exp(d_pen) - d_pen - 1.0).mean()
        ((pg + a.beta_kl * kl) / a.grad_accum).backward()
        acc += 1
        hist["pg"].append(float(pg)); hist["kl"].append(float(k3_all.mean()))
        hist["ratio"].append(float(ratio.mean()))
        hist["clipfrac"].append(float(((ratio - 1).abs() > a.clip_eps).float().mean()))
        if acc >= a.grad_accum or n == len(order) - 1:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
            acc = 0
            step += 1
            w = max(1, a.grad_accum * 5)
            mk = sum(hist["kl"][-w:]) / len(hist["kl"][-w:])
            if step % 5 == 0 or step == n_updates:
                print(f"[grpo] step {step}/{n_updates}  pg {sum(hist['pg'][-w:])/len(hist['pg'][-w:]):+.4f}"
                      f"  KL {mk:.4f}  rho {sum(hist['ratio'][-w:])/len(hist['ratio'][-w:]):.3f}"
                      f"  clip {sum(hist['clipfrac'][-w:])/len(hist['clipfrac'][-w:]):.2%}"
                      f"  {(time.time()-t0)/60:.1f}m", flush=True)
            if a.kl_max > 0 and mk > a.kl_max:
                raise SystemExit(f"[FATAL] DIVERGED: windowed mean KL {mk:.2f} > --kl-max "
                                 f"{a.kl_max}. NOTHING SAVED. Lower ATK_LR.")

    os.makedirs(a.out_dir, exist_ok=True)
    model.save_pretrained(a.out_dir)
    tok.save_pretrained(a.out_dir)
    m = lambda xs: sum(xs) / max(1, len(xs))
    summ = {"stage": "grpo_safereview_attacker", "attacker": a.attacker,
            "target_reviewer": a.model, "n_examples": len(prep), "n_updates": step,
            "lr": a.lr, "beta_kl": a.beta_kl, "clip_eps": a.clip_eps,
            "grad_accum": a.grad_accum, "epochs": a.epochs,
            "mean_kl": m(hist["kl"]), "mean_ratio": m(hist["ratio"]),
            "mean_clipfrac": m(hist["clipfrac"]), "max_ratio": None,
            "mean_reward": m([r["reward"] for r, _, _ in prep]),
            "init_adapter": a.init_adapter or "base", "scored": a.scored,
            "minutes": (time.time() - t0) / 60}
    with open(os.path.join(a.out_dir, "grpo_summary.json"), "w") as f:
        json.dump(summ, f, indent=2)
    band = ("INERT" if summ["mean_kl"] < 0.005 else "USABLE" if summ["mean_kl"] <= 0.1
            else "HOT" if summ["mean_kl"] <= 1 else "TOO FAR")
    print(f"\n[grpo] {step} updates, mean KL {summ['mean_kl']:.4f} ({band}), "
          f"mean ratio {summ['mean_ratio']:.3f}\n[saved] {a.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
