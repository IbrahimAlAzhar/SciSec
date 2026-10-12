#!/usr/bin/env python3
"""
train_grpo_v8.py — GRPO on v8 rollout groups (the provenance-reward ablation).

Input: datasets/grpo/<model>/<task>/<reward>/train.jsonl, one line per sampled completion:
    {"messages": [...], "completion": "...", "adv": <group-normalised reward>, ...}

Everything v7 paid to learn is built in:
  * ONE base-model load (ft_model.load_base resolves the dtype kwarg from the version)
  * sdpa attention, not eager: gemma3 does not soft-cap logits, and eager materialises
    [heads, L, L] in fp32 -- 62 GiB at L=2048 on gemma3-27b
  * gradient checkpointing with use_reentrant=False + the input-grad hook a 4-bit
    embedding needs
  * logits restricted to the completion positions (262k-token vocabulary)
  * reference log-probs cached in ONE no-grad pass before any update, so pi_old = pi_ref =
    the initial (SFT) policy and the ratio measures real drift
  * k3 KL = exp(d) - d - 1 with d = lp_ref - lp. REPORTED unclamped, PENALISED with d
    clamped at --kl-clamp: its gradient is exponential in d and one collapsed token
    otherwise dominates the update
  * min_ratio / kl_p50 / kl_p95 / collapsed_frac recorded: in v7 mean_kl was a tail
    statistic (mean 0.05 with p95 0.0019) and ranking on it picked the wrong config
  * trust region: the run ABORTS WITHOUT SAVING above --kl-max, so a diverged adapter can
    never overwrite a good one under its derived name
  * prompts truncated from the LEFT: the schema contract and the answer are at the end
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import ft_config as C


def load_rows(path: str) -> List[Dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def render(tok, messages, completion: str, max_len: int) -> Optional[Tuple[List[int], int]]:
    try:
        p = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    except Exception as e:
        print(f"[tok] chat template failed: {type(e).__name__}: {e}")
        return None
    if hasattr(p, "input_ids"):
        p = p["input_ids"]
    c = tok(completion, add_special_tokens=False).input_ids
    if not c:
        return None
    if len(p) + len(c) > max_len:
        room = max_len - len(c)
        if room < 64:
            return None
        p = p[-room:]
    return list(p) + list(c), len(c)


def main() -> int:
    ap = argparse.ArgumentParser(description="GRPO for v8 agents")
    ap.add_argument("--model", required=True, choices=[k for k, v in C.MODELS.items() if not v.api])
    ap.add_argument("--task", required=True, choices=list(C.TASKS))
    ap.add_argument("--reward", default="provenance", choices=["standard", "provenance"])
    ap.add_argument("--train", default="")
    ap.add_argument("--init-adapter", dest="init_adapter", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--lr", type=float, default=C.GRPO.lr)
    ap.add_argument("--beta-kl", dest="beta_kl", type=float, default=C.GRPO.beta_kl)
    ap.add_argument("--clip-eps", dest="clip_eps", type=float, default=C.GRPO.clip_eps)
    ap.add_argument("--kl-clamp", dest="kl_clamp", type=float, default=C.GRPO.kl_clamp)
    ap.add_argument("--kl-max", dest="kl_max", type=float, default=C.GRPO.kl_max)
    ap.add_argument("--grad-accum", dest="grad_accum", type=int, default=C.GRPO.grad_accum)
    ap.add_argument("--max-seq-len", dest="max_seq_len", type=int, default=C.GRPO.max_seq_len)
    ap.add_argument("--epochs", type=float, default=C.GRPO.epochs)
    ap.add_argument("--max-steps", dest="max_steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", dest="dry_run", action="store_true")
    a = ap.parse_args()

    mk = a.model.replace("/", "_")
    path = a.train or C.wpath(C.D_GRPO, mk, a.task, a.reward, "train.jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"[FATAL] {path} not found -- run v8_pairs.py --reward {a.reward}")
    rows = [r for r in load_rows(path) if str(r.get("completion") or "").strip()]
    out = a.out or C.adapter_dir(a.model, f"grpo-{a.task}")
    spec = C.MODELS[a.model]
    print(f"[data] {len(rows)} turns from {path}")
    by = defaultdict(list)
    for r in rows:
        by[r.get("agent", "?")].append(float(r["adv"]))
    for ag, v in sorted(by.items()):
        mu = sum(v) / len(v)
        sd = math.sqrt(sum((x - mu) ** 2 for x in v) / max(1, len(v) - 1))
        print(f"  {ag:<10} {len(v):>6} turns   adv mean {mu:+.3f} sd {sd:.3f}")

    from ft_tokenizer import load_tokenizer
    tok = load_tokenizer(spec.path, a.model)
    prepared = []
    for r in rows:
        x = render(tok, r["messages"], r["completion"], a.max_seq_len)
        if x is not None:
            prepared.append((x[0], x[1], float(r["adv"])))
    print(f"[tok] {len(prepared)}/{len(rows)} tokenised")
    if not prepared:
        raise SystemExit("[FATAL] nothing to train on")
    if a.dry_run:
        L = sorted(len(p[0]) for p in prepared)
        print(f"[dry-run] length p50 {L[len(L)//2]} max {L[-1]}; stopping before the GPU")
        return 0

    import torch
    import ft_model as FM
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import BitsAndBytesConfig

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    quant = (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.bfloat16,
                                bnb_4bit_use_double_quant=True) if spec.load_4bit else None)
    base = FM.load_base(spec.path, quant=quant, dtype=torch.bfloat16,
                        device_map=FM.pick_device_map(),
                        attn_implementation=os.environ.get("V8_ATTN", "sdpa"))
    FM.set_use_cache(base, False)
    if a.init_adapter and os.path.isdir(a.init_adapter):
        print(f"[lora] continuing from {a.init_adapter}")
        model = PeftModel.from_pretrained(base, a.init_adapter, is_trainable=True)
    else:
        print("[lora] FRESH adapter -- this is GRPO from the base policy, not 'SFT then GRPO'")
        model = get_peft_model(base, LoraConfig(
            r=C.TRAIN.lora_r, lora_alpha=C.TRAIN.lora_alpha,
            lora_dropout=C.TRAIN.lora_dropout, bias="none", task_type="CAUSAL_LM",
            target_modules=list(FM.lora_targets(base, C.TRAIN.target_modules))))
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()
    try:
        model.enable_input_require_grads()
    except Exception:
        pass
    FM.set_use_cache(model, False)
    dev = next(model.parameters()).device

    import inspect
    keep_kw = None
    try:
        sig = inspect.signature(model.get_base_model().forward).parameters
        keep_kw = "logits_to_keep" if "logits_to_keep" in sig else (
            "num_logits_to_keep" if "num_logits_to_keep" in sig else None)
    except Exception:
        keep_kw = None
    print(f"[mem] logits restriction: {keep_kw or 'unavailable (full logits)'}")

    def seq_lp(ids: List[int], clen: int):
        x = torch.tensor([ids], device=dev)
        kw = {keep_kw: clen + 1} if keep_kw else {}
        o = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, **kw)
        lg = o.logits.float()
        lg = lg[:, :-1, :] if keep_kw else lg[:, -clen - 1:-1, :]
        tgt = x[:, -clen:]
        return torch.log_softmax(lg, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)

    print(f"[ref] caching reference log-probs for {len(prepared)} turns")
    t0 = time.time()
    ref = []
    model.eval()
    with torch.no_grad():
        for i, (ids, clen, _adv) in enumerate(prepared):
            ref.append(seq_lp(ids, clen).detach().to("cpu"))
            if (i + 1) % 200 == 0:
                print(f"[ref] {i+1}/{len(prepared)}  ({(time.time()-t0)/60:.1f}m)", flush=True)
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    order = list(range(len(prepared)))
    n_steps_total = max(1, int(len(order) * a.epochs) // a.grad_accum)
    if a.max_steps:
        n_steps_total = min(n_steps_total, a.max_steps)
    print(f"[train] {len(prepared)} turns, grad_accum {a.grad_accum} -> {n_steps_total} "
          f"steps, lr {a.lr:g}, beta {a.beta_kl}, clip {a.clip_eps}, kl_clamp {a.kl_clamp}")
    H = defaultdict(list)
    step = acc = 0
    t0 = time.time()
    ep = 0
    while step < n_steps_total:
        random.Random(a.seed + ep).shuffle(order)
        ep += 1
        for j in order:
            ids, clen, adv = prepared[j]
            lp = seq_lp(ids, clen)
            lr_ = ref[j].to(dev)
            ratio = torch.exp(lp - lr_)
            A_ = torch.tensor(adv, device=dev, dtype=ratio.dtype)
            pg = -torch.min(ratio * A_, torch.clamp(ratio, 1 - a.clip_eps, 1 + a.clip_eps) * A_).mean()
            d = lr_ - lp
            with torch.no_grad():
                k3 = torch.exp(d) - d - 1.0
            dp = d.clamp(max=a.kl_clamp) if a.kl_clamp > 0 else d
            kl = (torch.exp(dp) - dp - 1.0).mean()
            ((pg + a.beta_kl * kl) / a.grad_accum).backward()
            acc += 1
            with torch.no_grad():
                f = k3.flatten().float()
                H["kl"].append(float(f.mean()))
                H["kl_max"].append(float(f.max()))
                if f.numel() > 1:
                    q = torch.quantile(f, torch.tensor([0.5, 0.95], device=f.device))
                    H["kl_p50"].append(float(q[0]))
                    H["kl_p95"].append(float(q[1]))
                H["ratio"].append(float(ratio.mean()))
                H["ratio_min"].append(float(ratio.min()))
                H["ratio_max"].append(float(ratio.max()))
                H["collapsed"].append(float((ratio < 0.05).float().mean()))
                H["clip"].append(float(((ratio - 1).abs() > a.clip_eps).float().mean()))
                H["loss"].append(float(pg + a.beta_kl * kl))
            if acc >= a.grad_accum:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                acc = 0
                step += 1
                w = a.grad_accum * 5
                mk_ = sum(H["kl"][-w:]) / len(H["kl"][-w:])
                if step % 5 == 0:
                    p95 = H["kl_p95"][-w:]
                    print(f"[train] step {step}/{n_steps_total}  loss "
                          f"{sum(H['loss'][-w:])/len(H['loss'][-w:]):+.4f}  KL {mk_:.4f}  "
                          f"p95 {(sum(p95)/len(p95) if p95 else 0):.4f}  "
                          f"min_rho {min(H['ratio_min'][-w:]):.2e}  "
                          f"{(time.time()-t0)/60:.1f}m", flush=True)
                if a.kl_max > 0 and mk_ > a.kl_max:
                    raise SystemExit(
                        f"[FATAL] DIVERGED: windowed mean KL {mk_:.2f} > --kl-max "
                        f"{a.kl_max:g} at step {step}. NOTHING SAVED; {out} is untouched. "
                        f"Lower --lr (v7 measured: 6e-6 usable, 1.2e-5 diverged).")
                if step >= n_steps_total:
                    break

    m = lambda k: (sum(H[k]) / len(H[k])) if H[k] else None
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    summ = {"model": a.model, "task": a.task, "reward": a.reward, "out": out,
            "init_adapter": a.init_adapter, "n_turns": len(prepared), "n_updates": step,
            "minutes": (time.time() - t0) / 60, "mean_loss": m("loss"),
            "mean_kl": m("kl"), "kl_p50": m("kl_p50"), "kl_p95": m("kl_p95"),
            "kl_max": max(H["kl_max"] or [0]), "min_ratio": min(H["ratio_min"] or [0]),
            "max_ratio": max(H["ratio_max"] or [0]), "mean_ratio": m("ratio"),
            "collapsed_frac": m("collapsed"), "mean_clipfrac": m("clip"), "lr": a.lr,
            "beta_kl": a.beta_kl, "clip_eps": a.clip_eps, "kl_clamp": a.kl_clamp,
            "grad_accum": a.grad_accum}
    with open(os.path.join(out, "grpo_summary.json"), "w") as f:
        json.dump(summ, f, indent=2)
    p95 = summ["kl_p95"] or 0
    band = ("INERT" if p95 < 0.002 else "MOVING" if p95 < 0.05 else "FAR")
    print(f"\n[saved] {out}")
    print(f"  kl_p95 {p95:.4f} ({band})   mean_kl {summ['mean_kl']:.4f}   "
          f"min_rho {summ['min_ratio']:.2e}")
    print("  Judge movement on kl_p95, not mean_kl: the mean is set by a few collapsed "
          "tokens. INERT means the adapter barely differs from its initialisation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
