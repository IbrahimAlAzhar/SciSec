#!/usr/bin/env python3
"""
rl_reward_model.py — train the outcome reward model (ORM) and the process reward model
(PRM), and use them at inference.

THE QUESTION THIS FILE HAS TO ANSWER FIRST
=======================================================================================
If the reward is computable exactly from the gold insert, what is a learned reward model
FOR?

Three things, and none of them is "approximate a number we already have":

  1. PAPERS WITH NO GOLD. Every real submission. Also, in your own corpus, the attacked
     rows whose insert was truncated away by max_chars and the ARGAR-style repackaging
     rows where nothing was inserted at all -- `ins_start < 0`, so span IoU is undefined
     and the shaped reward collapses to the decision term. A reward model trained where
     gold exists supplies a signal where it does not.
  2. TEST-TIME SELECTION. Sample N cascade trajectories, score them with the PRM, keep
     the best. That converts extra inference compute into accuracy with no retraining,
     and it is the cheapest win in the package -- `best_of_n` at the bottom of this file.
  3. EARLY ABSTENTION. The PRM predicts a value after the SCREEN and after the
     CHUNK_VERIFIER, before the three workers and the leader have run. A trajectory
     predicted to end badly can be abandoned at step 2 instead of step 6, which is the
     `Calls` column in your table.

ORM vs PRM, WHICH IS THE ABLATION YOU ASKED FOR
    ORM   one scalar for the finished trajectory. Trained on (rendered trajectory,
          outcome_reward). Simple, and blind to WHERE the trajectory went wrong.
    PRM   one scalar per pipeline step, trained on (prefix, step reward). It can say
          "the INSTRUCTION worker's finding was the mistake" from the prefix alone,
          which is what makes it usable for abstention and for credit.

    Rows E and F of the ablation table are exactly this comparison.

A WARNING ABOUT WHAT A REWARD MODEL IS FOR HERE
    The ORM and PRM never decide the LABEL. The label came from the attack we applied.
    A reward model that starts deciding labels has become a second detector with no
    error bars, and its mistakes would be invisible because they would be baked into the
    reward that trained the detector being evaluated.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import rl_config as C

try:
    import ft_model as FM
except Exception as e:                                          # pragma: no cover
    raise SystemExit(f"[FATAL] cannot import ft_model from {C.V4_FT_DIR}: "
                     f"{type(e).__name__}: {e}")


# ======================================================================================
# rendering a trajectory (or a prefix of one) as text
# ======================================================================================
def render_prefix(rec: Dict, upto: Optional[str] = None, *,
                  doc_chars: int = 2400, out_chars: int = 900) -> str:
    """
    The trajectory, or its prefix, as a single string the reward model reads.

    The MANUSCRIPT REGION IS INCLUDED, truncated. Without it the model cannot tell a
    grounded quote from a fabricated one -- which is the single most important thing the
    reward is about -- and a reward model that cannot see the document learns to score
    style.

    `upto` stops after that step, which is what makes the PRM's prediction causal: a value
    at step k may not depend on what step k+1 said.
    """
    L: List[str] = []
    doc = str(rec.get("document") or rec.get("region") or "")
    if doc:
        L.append("MANUSCRIPT (limitations region, truncated):")
        L.append(doc[-doc_chars:])
        L.append("")
    s = rec.get("screen_score")
    L.append(f"SCREEN: {'n/a' if s is None else f'{float(s):+.3f}'}   "
             f"ESCALATED: {int(rec.get('escalated', 0))}")
    for st in rec.get("steps", []):
        L.append("")
        L.append(f"[{st['agent']}]")
        L.append(str(st.get("completion") or "")[:out_chars])
        if upto and st["agent"] == upto:
            break
    return "\n".join(L)


def steps_present(rec: Dict) -> List[str]:
    return [st["agent"] for st in rec.get("steps", [])]


# ======================================================================================
# the model
# ======================================================================================
class ValueHead:
    """
    A LoRA'd base model with a scalar head on the last non-padding hidden state.

    A separate head rather than AutoModelForSequenceClassification on purpose: gemma3's
    config maps to Gemma3ForConditionalGeneration and Mistral3Config is not in the
    sequence-classification mapping at all, so the AutoModel path fails on two of your
    three reviewers. ft_model.load_base already solves that for causal LM, and a linear
    layer on the final hidden state is the whole difference.

    MEMORY — WHY THIS CLASS IS SHAPED THE WAY IT IS
    A 4-bit gemma3-27b is ~16 GiB of a 39.5 GiB card, which leaves ~23 GiB for
    activations. Four things spent more than that, and all four are fixed here:

      1. eager attention materialised [1, 32, L, L] and softmaxed it in fp32, retained for
         backward: ~800 MB per layer x 62 layers ~= 50 GiB at L=2048. Now sdpa
         (see rl_load.py for why that is safe on gemma3 and not on gemma2).
      2. the base model was loaded TWICE by ft_model.load_base's dtype-kwarg retry
         (12 shards, then 12 shards again, first copy never freed). Now rl_load.
      3. the full CausalLM computed logits over 262,144 tokens — 2.1 GiB at L=2048,
         discarded unread — and output_hidden_states=True retained all 63 layer outputs.
         Now the decoder trunk only, which returns last_hidden_state and no logits.
      4. no gradient checkpointing at all. Now on, with the input-grad hook a quantised
         embedding needs.

    CORRECTNESS — THE TRUNCATION SIDE
    render_prefix puts the MANUSCRIPT first and the trajectory steps last, and the value is
    read off the LAST token. Truncating on the right (the tokenizer default) therefore
    deleted the trajectory and kept the document, so every long prefix scored the same
    paper text and the PRM's step-to-step differences came from nothing. truncation_side
    is set to "left": the document loses its head, the trajectory survives.
    """

    def __init__(self, model_key: str, *, r: int = 16, alpha: int = 32,
                 load_adapter: str = "", grad_ckpt: bool = True,
                 attn: Optional[str] = None):
        import torch
        import torch.nn as nn
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoTokenizer

        import rl_load as RL
        self.torch = torch
        spec = C.MODELS[model_key]
        self.spec = spec
        self.tok = AutoTokenizer.from_pretrained(spec.path, trust_remote_code=True)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        # See the class docstring: the value is the LAST token, so the trajectory is the
        # part that must survive truncation.
        self.tok.truncation_side = "left"

        base = RL.load_base_once(spec.path, quant=C.bnb_config(spec),
                                 dtype=C.compute_dtype(),
                                 device_map=FM.pick_device_map(),
                                 attn_implementation=(attn or RL.attn_default()))
        FM.set_use_cache(base, False)
        print(RL.cuda_report("after base load"))
        if grad_ckpt:
            base = RL.enable_kbit_training(base, grad_ckpt=True)
            FM.set_use_cache(base, False)

        if load_adapter and os.path.isdir(load_adapter):
            self.model = PeftModel.from_pretrained(base, load_adapter,
                                                   is_trainable=False)
        else:
            targets = FM.lora_targets(base, ("q_proj", "k_proj", "v_proj", "o_proj"))
            self.model = get_peft_model(base, LoraConfig(
                r=r, lora_alpha=alpha, lora_dropout=0.05, bias="none",
                task_type="CAUSAL_LM", target_modules=targets))
        self.dev = next(self.model.parameters()).device

        # The decoder stack, verified by probe rather than trusted. A wrong guess would
        # score the wrong tensor silently; a probe turns that into a startup message.
        self.trunk_name, self.trunk = RL.trunk_of(self.model)
        if self.trunk is not None and not self._probe_trunk():
            print("[value] the trunk did not return last_hidden_state — falling back to "
                  "the full model with output_hidden_states=True (costs the LM head)")
            self.trunk = None

        hidden = self._hidden_size()
        self.head = nn.Linear(hidden, 1, dtype=torch.float32).to(self.dev)
        hp = os.path.join(load_adapter or "", "value_head.pt")
        if load_adapter and os.path.exists(hp):
            self.head.load_state_dict(torch.load(hp, map_location=self.dev))
            print(f"[rm] value head loaded from {hp}")
        print(RL.cuda_report("ready"))

    def _hidden_size(self) -> int:
        cfg = FM.config_of(self.spec.path)
        for c in (getattr(cfg, "text_config", None), cfg):
            v = getattr(c, "hidden_size", None)
            if v:
                return int(v)
        return int(getattr(self.model.config, "hidden_size", 4096))

    def _probe_trunk(self) -> bool:
        """Four tokens through the trunk: does it give last_hidden_state of the right width?"""
        torch = self.torch
        try:
            ids = self.tok("probe", return_tensors="pt").to(self.dev)
            with torch.no_grad():
                out = self.trunk(input_ids=ids["input_ids"],
                                 attention_mask=ids.get("attention_mask"),
                                 use_cache=False)
            h = getattr(out, "last_hidden_state", None)
            if h is None and isinstance(out, (tuple, list)):
                h = out[0]
            ok = h is not None and h.dim() == 3 and h.shape[-1] == self._hidden_size()
            if ok:
                print(f"[value] trunk probe ok: {tuple(h.shape)}")
            return bool(ok)
        except Exception as e:
            print(f"[value] trunk probe failed: {type(e).__name__}: {e}")
            return False

    def parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad] + \
               list(self.head.parameters())

    def value(self, text: str, max_len: int = 1024):
        ids = self.tok(text, return_tensors="pt", truncation=True,
                       max_length=max_len).to(self.dev)
        if self.trunk is not None:
            out = self.trunk(input_ids=ids["input_ids"],
                             attention_mask=ids.get("attention_mask"),
                             use_cache=False)
            h = getattr(out, "last_hidden_state", None)
            if h is None:
                h = out[0]
        else:
            out = self.model(**ids, output_hidden_states=True, use_cache=False)
            h = out.hidden_states[-1]
        return self.head(h[:, -1, :].float()).squeeze(-1)

    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        self.model.save_pretrained(out_dir)
        self.tok.save_pretrained(out_dir)
        self.torch.save(self.head.state_dict(),
                        os.path.join(out_dir, "value_head.pt"))
        print(f"[saved] {out_dir}")


# ======================================================================================
# datasets
# ======================================================================================
def orm_examples(rollouts: Sequence[Dict]) -> List[Tuple[str, float]]:
    """(rendered full trajectory, outcome reward)."""
    return [(render_prefix(r), float(r.get("reward", 0.0))) for r in rollouts]


def prm_examples(rollouts: Sequence[Dict], *, gamma: float = 0.9
                 ) -> List[Tuple[str, float]]:
    """
    (rendered PREFIX up to step k, discounted return from step k).

    The target is the RETURN, not the immediate step reward. A PRM trained on immediate
    rewards learns to score the current utterance in isolation, which is exactly what SFT
    already does and what this package exists to move past. The discounted return makes
    the value at step k depend on what that step makes possible later, which is what turns
    the PRM into something usable for early abstention: a worker's finding is good if the
    leader can act on it, not if it reads well.
    """
    out: List[Tuple[str, float]] = []
    for r in rollouts:
        agents = steps_present(r)
        sr = r.get("step_rewards") or {}
        if not agents:
            continue
        # returns, computed backwards over the steps that actually happened
        rets: Dict[str, float] = {}
        acc = float(sr.get("EXCISE", 0.0))
        for ag in reversed(agents):
            acc = float(sr.get(ag, 0.0)) + gamma * acc
            rets[ag] = acc
        for ag in agents:
            out.append((render_prefix(r, upto=ag), rets[ag]))
    return out


def load_rollouts(paths: Sequence[str]) -> List[Dict]:
    recs: List[Dict] = []
    for p in paths:
        fp = os.path.join(p, "rollouts.jsonl") if os.path.isdir(p) else p
        if not os.path.exists(fp):
            print(f"[rm] skipping missing {fp}")
            continue
        n = 0
        for line in open(fp):
            if line.strip():
                recs.append(json.loads(line))
                n += 1
        print(f"[rm] {n} rollouts from {fp}")
    return recs


# ======================================================================================
# training
# ======================================================================================
def fit(examples: List[Tuple[str, float]], model_key: str, out_dir: str, *,
        epochs: float = 1.0, lr: float = 1e-4, grad_accum: int = 8,
        max_len: int = 1024, val_frac: float = 0.1, seed: int = 0,
        dry_run: bool = False, grad_ckpt: bool = True,
        attn: Optional[str] = None) -> Dict:
    import torch
    import rl_load as RL

    rng = random.Random(seed)
    rng.shuffle(examples)
    n_val = max(1, int(val_frac * len(examples)))
    val, tr = examples[:n_val], examples[n_val:]

    ys = [y for _x, y in tr]
    mu = sum(ys) / max(1, len(ys))
    sd = math.sqrt(sum((y - mu) ** 2 for y in ys) / max(1, len(ys) - 1)) or 1.0
    print(f"[rm] {len(tr)} train / {len(val)} val   target mean {mu:+.3f} sd {sd:.3f}")
    if sd < 1e-3:
        raise SystemExit(
            "[FATAL] every training target is the same value. A reward model fitted to a "
            "constant predicts that constant, and the ablation row would report a "
            "reward model that is really a bias term. Roll out at a higher temperature, "
            "or use a reward with continuous terms (`provenance`) rather than `outcome`.")

    # Standardised targets. Reward scales differ by an order of magnitude between the
    # outcome reward (4 discrete values) and the discounted process return, and an MSE
    # head fitted on the raw scale learns the mean and stops.
    tr = [(x, (y - mu) / sd) for x, y in tr]
    val = [(x, (y - mu) / sd) for x, y in val]

    if dry_run:
        print("[rm] --dry-run: not loading the model.")
        return {"dry_run": True, "n_train": len(tr), "mean": mu, "sd": sd}

    vh = ValueHead(model_key, grad_ckpt=grad_ckpt, attn=attn)
    opt = torch.optim.AdamW(vh.parameters(), lr=lr)
    n_steps = int(len(tr) * epochs)
    t0 = time.time()
    opt.zero_grad(set_to_none=True)
    losses: List[float] = []

    def _oom(step: int, e: Exception) -> "SystemExit":
        return SystemExit(
            f"[FATAL] CUDA OOM at step {step}/{n_steps} with max_len={max_len}.\n"
            f"        {RL.cuda_report('at failure')}\n"
            f"        {str(e).splitlines()[0]}\n"
            f"        In order of effect, and none of these needs a second GPU:\n"
            f"          --max-len {max(256, max_len // 2)}   (attention is linear in L "
            f"under sdpa, the hidden stream is too)\n"
            f"          check the log for 'Loading checkpoint shards' appearing TWICE — "
            f"that is two\n"
            f"            copies of the base model and it is a load bug, not an "
            f"activation problem\n"
            f"          export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   "
            f"(fragmentation)\n"
            f"          confirm the startup line says attn_implementation='sdpa'; with "
            f"'eager'\n"
            f"            gemma3-27b needs ~50 GiB of attention alone at L=2048\n"
            f"        If you do want two cards: select=1:ngpus=2 in ONE chunk, and "
            f"pick_device_map()\n"
            f"        switches to device_map='auto' by itself. Ray and tensor parallelism "
            f"are a\n"
            f"        vLLM serving feature and cannot train anything.")

    for i in range(n_steps):
        x, y = tr[i % len(tr)]
        try:
            pred = vh.value(x, max_len=max_len)
            loss = torch.nn.functional.mse_loss(
                pred, torch.tensor([y], device=vh.dev, dtype=pred.dtype))
            (loss / grad_accum).backward()
        except torch.OutOfMemoryError as e:
            opt.zero_grad(set_to_none=True)
            RL.free_cuda()
            raise _oom(i + 1, e)
        losses.append(float(loss))
        if (i + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(vh.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
        if (i + 1) % (grad_accum * 20) == 0:
            w = losses[-grad_accum * 20:]
            print(f"[rm] {i+1}/{n_steps}  mse {sum(w)/len(w):.4f}  "
                  f"{(time.time()-t0)/60:.1f}m  {RL.cuda_report('').strip()}",
                  flush=True)

    # ---- validation: the only number that says whether this model is worth anything ---
    preds, gts = [], []
    with torch.no_grad():
        for x, y in val:
            preds.append(float(vh.value(x, max_len=max_len)))
            gts.append(y)
    n = len(gts)
    mp, mg = sum(preds) / n, sum(gts) / n
    num = sum((a - mp) * (b - mg) for a, b in zip(preds, gts))
    dp = math.sqrt(sum((a - mp) ** 2 for a in preds))
    dg = math.sqrt(sum((b - mg) ** 2 for b in gts))
    r = num / (dp * dg) if dp and dg else float("nan")
    mse = sum((a - b) ** 2 for a, b in zip(preds, gts)) / n
    # the trivial competitor: predict the mean
    mse0 = sum(b ** 2 for b in gts) / n

    vh.save(out_dir)
    meta = {"model": model_key, "n_train": len(tr), "n_val": n,
            "target_mean": mu, "target_sd": sd,
            "val_pearson": r, "val_mse": mse, "val_mse_predict_mean": mse0,
            "minutes": (time.time() - t0) / 60}
    with open(os.path.join(out_dir, "rm_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 88)
    print(f"REWARD MODEL — {os.path.basename(out_dir)}")
    print("=" * 88)
    print(f"  validation Pearson r  {r:+.3f}")
    print(f"  validation MSE        {mse:.4f}   (predicting the mean: {mse0:.4f})")
    if mse >= mse0 * 0.95:
        print()
        print("  THIS MODEL IS NOT BETTER THAN PREDICTING THE MEAN. Do not use it as a")
        print("  reward: the ablation row would then measure noise with a label on it.")
        print("  Usual causes, in order: too few rollouts, targets with no variance, or")
        print("  the rendered prefix not containing the manuscript (so the model cannot")
        print("  tell a grounded quote from an invented one).")
    elif r < 0.3:
        print()
        print(f"  r = {r:+.3f} is weak. Usable for best-of-N reranking, where only the")
        print("  ORDER matters, and NOT usable as a training reward, where the magnitude")
        print("  does. Say which of the two you used.")
    print(f"\n[saved] {out_dir}/rm_meta.json")
    return meta


# ======================================================================================
# inference-time use
# ======================================================================================
def best_of_n(candidates: Sequence[Dict], vh: "ValueHead", *,
              upto: Optional[str] = None, max_len: int = 2048) -> Tuple[int, List[float]]:
    """
    Pick the best of N sampled trajectories by predicted value.

    The cheapest thing in this package: no retraining, and it converts inference compute
    into accuracy directly. Report it as a separate row rather than folding it into the
    policy's number, because it costs N times the calls and the `Calls` column has to
    stay honest.
    """
    scores = [float(vh.value(render_prefix(c, upto=upto), max_len=max_len))
              for c in candidates]
    return (max(range(len(scores)), key=lambda i: scores[i]) if scores else -1), scores


def should_abstain(rec: Dict, vh: "ValueHead", *, after: str = "CHUNK_VERIFIER",
                   threshold: float = -0.5, max_len: int = 2048) -> Tuple[bool, float]:
    """
    Early abstention: after `after`, is this trajectory predicted to end badly?

    Returns (abstain, value). Abstaining means STOP AND SAY CLEAN, which on this task is
    the safe direction -- the cost of a miss is a manipulated review, the cost of a false
    positive is an accused author, and the reward weights say the second is worse. A
    threshold tuned on the val split and frozen; tuned on test it is not a result.
    """
    v = float(vh.value(render_prefix(rec, upto=after), max_len=max_len))
    return v < threshold, v


# ======================================================================================
def main() -> int:
    p = argparse.ArgumentParser(description="train the ORM or the PRM")
    p.add_argument("--model", required=True, choices=sorted(C.MODELS))
    p.add_argument("--kind", required=True, choices=["orm", "prm"])
    p.add_argument("--rollouts", nargs="+", required=True,
                   help="rollout directories (or rollouts.jsonl paths) to train on")
    p.add_argument("--out-dir", dest="out_dir", default="")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--grad-accum", dest="grad_accum", type=int, default=8)
    p.add_argument("--max-len", dest="max_len", type=int, default=1024,
                   help="tokens the value head reads. Truncation keeps the TAIL (the "
                        "trajectory), not the head (the manuscript).")
    p.add_argument("--attn", default=None,
                   choices=["sdpa", "eager", "flash_attention_2"],
                   help="default sdpa. 'eager' materialises [heads, L, L] and needs "
                        "~50 GiB on gemma3-27b at L=2048; it is only required for "
                        "Gemma 2, which soft-caps attention logits.")
    p.add_argument("--no-grad-ckpt", dest="grad_ckpt", action="store_false",
                   help="disable gradient checkpointing (faster, much more memory)")
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", dest="dry_run", action="store_true")
    a = p.parse_args()

    recs = load_rollouts(a.rollouts)
    if not recs:
        raise SystemExit("[FATAL] no rollouts found.")
    ex = orm_examples(recs) if a.kind == "orm" else prm_examples(recs, gamma=a.gamma)
    print(f"[rm] {a.kind.upper()}: {len(ex)} examples from {len(recs)} rollouts")

    mk = a.model.replace("/", "_")
    out_dir = a.out_dir or C.wpath(C.D_RM, mk, a.kind)
    fit(ex, a.model, out_dir, epochs=a.epochs, lr=a.lr, grad_accum=a.grad_accum,
        max_len=a.max_len, seed=a.seed, dry_run=a.dry_run,
        grad_ckpt=a.grad_ckpt, attn=a.attn)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
