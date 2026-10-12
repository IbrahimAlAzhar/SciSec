#!/usr/bin/env python3
"""
sr_torch.py — what the two SafeReview trainers share: loading a (4-bit) base with a LoRA,
rendering chat prompts exactly as vLLM does, and completion log-probabilities.

Everything that v7 already solved is REUSED, not re-solved:
    rl_load.load_base_once     one from_pretrained call, sdpa attention (rl_load.py says
                               why both matter on a 40 GiB card)
    ft_model.pick_device_map / lora_targets / set_use_cache   (V4), when importable
    train_grpo.seq_logprobs    re-stated below with the same logits_to_keep trick: on
                               gemma3's 262k vocabulary it is the difference between ~1 GiB
                               and ~6 GiB of logits
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import ft_model as FM                     # V4 (on the cluster)
except Exception:                             # pragma: no cover
    FM = None


def pick_device_map():
    if FM is not None and hasattr(FM, "pick_device_map"):
        return FM.pick_device_map()
    import torch
    if torch.cuda.is_available():
        return "auto" if torch.cuda.device_count() > 1 else {"": 0}
    return None


def lora_targets(model, names: Sequence[str]):
    """V4's resolver when present (the one your served adapters were built with). The
    fallback is a regex that keeps LoRA off the vision tower of gemma3 / mistral3: their
    vision encoders also have q_proj/k_proj, and an adapter on them is one vLLM refuses."""
    if FM is not None and hasattr(FM, "lora_targets"):
        return FM.lora_targets(model, names)
    alt = "|".join(re.escape(n) for n in names)
    return rf"^(?!.*(?:vision|multi_modal|mm_projector)).*\.(?:{alt})$"


def set_use_cache(model, flag: bool) -> None:
    if FM is not None and hasattr(FM, "set_use_cache"):
        FM.set_use_cache(model, flag)
        return
    for m in (model, getattr(model, "config", None)):
        try:
            if m is not None and hasattr(m, "config"):
                m.config.use_cache = flag
            elif m is not None:
                m.use_cache = flag
        except Exception:
            pass


def load_policy(path: str, *, load_4bit: bool, init_adapter: str, lora_cfg: Dict,
                grad_ckpt: bool = True):
    """(model, tokenizer). Continues from `init_adapter` when it exists (R_{t-1} / A_{t-1}),
    otherwise a FRESH LoRA, which is the identity at step 0 (PEFT initialises lora_B = 0),
    so the policy at the start IS the base model and the cached reference is correct."""
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoTokenizer
    import rl_load as RLL

    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    quant = None
    if load_4bit:
        from transformers import BitsAndBytesConfig
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.bfloat16,
                                   bnb_4bit_use_double_quant=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base = RLL.load_base_once(path, quant=quant, dtype=dtype,
                              device_map=pick_device_map(),
                              attn_implementation=RLL.attn_default())
    set_use_cache(base, False)
    print(RLL.cuda_report("after base load"))
    if init_adapter and os.path.isdir(init_adapter):
        print(f"[lora] continuing from {init_adapter}")
        model = PeftModel.from_pretrained(base, init_adapter, is_trainable=True)
    else:
        if init_adapter:
            print(f"[lora] {init_adapter} not found -- starting a FRESH adapter")
        model = get_peft_model(base, LoraConfig(
            r=lora_cfg["r"], lora_alpha=lora_cfg["alpha"], lora_dropout=lora_cfg["dropout"],
            bias="none", task_type="CAUSAL_LM",
            target_modules=lora_targets(base, lora_cfg["targets"])))
    try:
        model.print_trainable_parameters()
    except Exception:
        pass
    if grad_ckpt and torch.cuda.is_available():
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
        except Exception as e:
            print(f"[mem] gradient checkpointing unavailable: {type(e).__name__}: {e}")
        try:
            model.enable_input_require_grads()
        except Exception:
            pass
        set_use_cache(model, False)
    model.train()
    return model, tok


# ======================================================================================
# rendering
# ======================================================================================
def _template(tok, messages: List[Dict], *, gen: bool, chat_kwargs: Optional[Dict]) -> str:
    kw = dict(chat_kwargs or {})
    try:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=gen,
                                       **kw)
    except TypeError:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=gen)


def _ids(tok, text: str) -> List[int]:
    return list(tok(text, add_special_tokens=False).input_ids)


def prompt_ids(tok, messages: List[Dict], chat_kwargs: Optional[Dict] = None) -> List[int]:
    """The prompt as vLLM's /chat/completions renders it: template + generation prompt.
    Rendered to TEXT and then tokenised, which behaves the same on transformers 4.x and
    5.x (5.x changed what apply_chat_template(tokenize=True) returns)."""
    return _ids(tok, _template(tok, messages, gen=True, chat_kwargs=chat_kwargs))


def completion_ids(tok, messages: List[Dict], completion: str, *,
                   chat_kwargs: Optional[Dict] = None, end_of_turn: bool = True) -> List[int]:
    """The assistant turn's tokens. With end_of_turn=True (DPO) they include the template's
    end-of-turn marker, taken from the template itself: full conversation minus the
    generation prompt. With end_of_turn=False (GRPO) they are the raw sampled text, as in
    v7's train_grpo.render -- no token the policy did not emit."""
    if not end_of_turn:
        return _ids(tok, completion)
    p = _template(tok, messages, gen=True, chat_kwargs=chat_kwargs)
    full = _template(tok, list(messages) + [{"role": "assistant", "content": completion}],
                     gen=False, chat_kwargs=chat_kwargs)
    if full.startswith(p):
        return _ids(tok, full[len(p):].rstrip())
    ids = _ids(tok, completion)
    if tok.eos_token_id is not None:
        ids.append(int(tok.eos_token_id))
    return ids


def join(p_ids: List[int], c_ids: List[int], max_len: int) -> Optional[Tuple[List[int], int]]:
    """LEFT truncation of the prompt (the completion and the end of the prompt survive),
    as in v7's train_grpo.render. None when the completion alone fills the window."""
    if not c_ids:
        return None
    if len(p_ids) + len(c_ids) > max_len:
        room = max_len - len(c_ids)
        if room < 64:
            return None
        p_ids = p_ids[-room:]
    return list(p_ids) + list(c_ids), len(c_ids)


# ======================================================================================
# log-probabilities
# ======================================================================================
def probe_logits_to_keep(model) -> bool:
    """Ask the CausalLM class, not the PEFT wrapper: PeftModel.forward is (*args, **kwargs)
    (or a fixed list without logits_to_keep), and walking `.base_model` overshoots to the
    decoder stack, which has no LM head. get_base_model() is the CausalLM; the kwarg is
    passed through the wrapper's **kwargs."""
    import inspect
    cands = []
    if hasattr(model, "get_base_model"):
        try:
            cands.append(model.get_base_model())
        except Exception:
            pass
    cands.append(model)
    for m in cands:
        try:
            if "logits_to_keep" in inspect.signature(m.forward).parameters:
                return True
        except (TypeError, ValueError):
            pass
    print("[mem] no logits_to_keep on this model: full logits are materialised")
    return False


def seq_logprobs(model, ids: List[int], comp_len: int, *, keep: bool):
    """Per-token log-probs of the last comp_len tokens, shape (1, comp_len)."""
    import torch
    dev = next(model.parameters()).device
    x = torch.tensor([ids], dtype=torch.long, device=dev)
    kw = {"logits_to_keep": comp_len + 1} if keep else {}
    out = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, **kw)
    logits = out.logits.float()
    logits = logits[:, :-1, :] if keep else logits[:, -comp_len - 1:-1, :]
    tgt = x[:, -comp_len:]
    return torch.log_softmax(logits, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
