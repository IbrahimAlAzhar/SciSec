#!/usr/bin/env python3
"""
rl_load.py — load a 27B base model ONCE, with attention that does not materialise L x L.

WHY THIS FILE EXISTS
=======================================================================================
Two separate defects put ~40 GiB on one card before a single training step ran.

DEFECT 1 — THE BASE MODEL WAS LOADED TWICE.
    `ft_model.load_base` tries `dtype=` and then `torch_dtype=` because transformers
    renamed the kwarg. On a transformers that wants `torch_dtype`, the `dtype=` attempt
    does not fail early: from_pretrained reads all 12 shards onto the GPU and only then
    raises TypeError. The retry reads all 12 again, and the first copy is still held by
    the caching allocator. Two 4-bit copies of gemma3-27b is ~31 GiB of a 39.5 GiB card,
    and the job then OOMs on the first forward — which looks like an activation problem
    and is not one.

    The log signature is unmistakable: `Loading checkpoint shards: 100%|...| 12/12`
    printed TWICE for one model. Yours printed it twice.

    Fixed by resolving the kwarg name from the transformers version BEFORE loading, so
    from_pretrained is called once. The fallback retry still exists for the case where the
    version check is wrong, but it frees the GPU first, so the worst case is slow rather
    than fatal.

DEFECT 2 — attn_implementation="eager".
    Eager attention builds the full [batch, heads, L, L] score matrix, and gemma3's eager
    path then runs `softmax(attn_weights, dtype=torch.float32)`, which allocates a second
    copy in fp32. Autograd keeps the softmax output for the backward pass, so the cost is
    paid per layer and never released until backward.

    gemma3-27b at L=2048: 32 heads x 2048 x 2048 = 134M elements.
        bf16 scores        268 MB
        fp32 softmax       537 MB   <- retained for backward
        bf16 cast back     268 MB
    x 62 layers ~= 50 GiB. On a 40 GiB card this cannot work at any batch size, and
    dropping to L=1024 only buys a factor of 4 on the term that dominates.

    With "sdpa" PyTorch dispatches to the memory-efficient / flash kernel and the L x L
    matrix is never materialised: memory becomes linear in L instead of quadratic.

    Eager is REQUIRED for Gemma **2**, which soft-caps attention logits — a fused kernel
    computes the wrong numbers there. Gemma **3** removed logit soft-capping (it uses
    QK-norm and alternating sliding-window attention instead), so sdpa is not a
    correctness compromise on gemma3. That is the reason this file is allowed to change
    the setting rather than just documenting it.

DEFECT 3 — the value head paid for the LM head.
    A value head needs the last hidden state. Calling the full CausalLM computes logits
    over gemma3's 262,144-token vocabulary: at L=2048 that is 2.1 GiB in fp32, discarded
    unread, plus `output_hidden_states=True` retaining all 63 layer outputs. `trunk_of()`
    returns the decoder stack, whose forward returns `last_hidden_state` and no logits.

This module is imported by rl_reward_model.py and train_grpo.py. It loads nothing at
import time and needs no GPU to import, so rl_selftest can exercise the pure parts.
"""

from __future__ import annotations

import gc
import os
from typing import Any, Optional, Tuple

# The transformers release that renamed `torch_dtype` -> `dtype` on from_pretrained.
_DTYPE_RENAME = (4, 56)


def transformers_version() -> Tuple[int, ...]:
    import transformers
    parts = []
    for tok in str(transformers.__version__).split(".")[:3]:
        num = "".join(ch for ch in tok if ch.isdigit())
        parts.append(int(num) if num else 0)
    return tuple(parts)


def dtype_kwarg() -> str:
    """
    Which spelling this transformers accepts, decided from the version instead of by
    calling from_pretrained twice. Overridable with RL_DTYPE_KWARG when a version lies.
    """
    forced = os.environ.get("RL_DTYPE_KWARG", "").strip()
    if forced in ("dtype", "torch_dtype"):
        return forced
    return "dtype" if transformers_version() >= _DTYPE_RENAME else "torch_dtype"


def free_cuda() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def _classes() -> list:
    import transformers
    out = []
    for name in ("AutoModelForCausalLM", "AutoModelForImageTextToText",
                 "AutoModelForVision2Seq"):
        cls = getattr(transformers, name, None)
        if cls is not None:
            out.append((name, cls))
    return out


def load_base_once(path: str, *, quant=None, dtype=None, device_map=None,
                   attn_implementation: str = "sdpa", trust_remote_code: bool = True,
                   verbose: bool = True) -> Any:
    """
    One call to from_pretrained in the normal case. Same class-resolution order as
    ft_model.load_base (AutoModelForCausalLM first — for gemma3 it already returns the
    multimodal class; the image-text classes are the mistral3 case), but the dtype kwarg
    is decided up front and the GPU is freed before any retry.
    """
    import transformers

    key = dtype_kwarg()
    other = "torch_dtype" if key == "dtype" else "dtype"
    if verbose:
        print(f"[load] transformers {transformers.__version__} -> dtype kwarg {key!r}; "
              f"attn_implementation={attn_implementation!r}")

    base_kw = dict(trust_remote_code=trust_remote_code)
    if quant is not None:
        base_kw["quantization_config"] = quant
    if device_map is not None:
        base_kw["device_map"] = device_map
    if attn_implementation:
        base_kw["attn_implementation"] = attn_implementation

    candidates = list(_classes())
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(path, trust_remote_code=trust_remote_code)
        for arch in (getattr(cfg, "architectures", None) or []):
            cls = getattr(transformers, arch, None)
            if cls is not None:
                candidates.append((arch, cls))
    except Exception:
        pass

    errors = []
    for name, cls in candidates:
        for spelling in ((key, other) if dtype is not None else (None,)):
            kw = dict(base_kw)
            if spelling is not None:
                kw[spelling] = dtype
            try:
                model = cls.from_pretrained(path, **kw)
                if verbose:
                    print(f"[load] {name} -> {type(model).__name__}"
                          + (f"  ({spelling}=)" if spelling else ""))
                return model
            except TypeError as e:
                msg = str(e)
                if spelling == key and ("dtype" in msg or "torch_dtype" in msg):
                    # The version check was wrong. Release the partially built model
                    # BEFORE retrying — this is the 31 GiB that used to accumulate.
                    print(f"[load] {key}= rejected ({msg.splitlines()[0]}); freeing the "
                          f"GPU and retrying with {other}=")
                    free_cuda()
                    continue
                errors.append(f"{name}: TypeError: {msg.splitlines()[0]}")
                free_cuda()
                break
            except (ValueError, KeyError) as e:
                errors.append(f"{name}: {type(e).__name__}: {str(e).splitlines()[0]}")
                free_cuda()
                break
            except Exception as e:                       # OOM, missing kernel, ...
                errors.append(f"{name}: {type(e).__name__}: {str(e).splitlines()[0]}")
                free_cuda()
                break

    raise SystemExit(
        "[FATAL] could not load the base model.\n"
        f"        path: {path}\n"
        + "".join(f"        - {e}\n" for e in errors)
        + f"        dtype kwarg tried first: {key} (transformers "
          f"{transformers.__version__}); override with RL_DTYPE_KWARG.\n")


# ======================================================================================
# the decoder stack, without the LM head
# ======================================================================================
def trunk_of(model: Any, *, verbose: bool = True) -> Tuple[str, Any]:
    """
    Return (name, module) for the submodule whose forward gives `last_hidden_state`.

    Resolution order, all of which exist on some model in this project:
        get_decoder()             the transformers API; correct when implemented
        .model.language_model     gemma3 / mistral3, where the text stack is nested
        .model                    plain text CausalLM
    PEFT wrappers are unwrapped first. The caller VERIFIES the choice with a probe forward
    rather than trusting it, because a wrong guess here would silently score the wrong
    tensor instead of crashing.
    """
    m = model
    for _ in range(3):
        if hasattr(m, "get_base_model"):
            try:
                m = m.get_base_model()
                continue
            except Exception:
                pass
        if type(m).__name__ in ("PeftModel", "PeftModelForCausalLM", "LoraModel") and \
                hasattr(m, "base_model"):
            m = m.base_model
            continue
        break

    cands = []
    if hasattr(m, "get_decoder"):
        try:
            d = m.get_decoder()
            if d is not None:
                cands.append(("get_decoder()", d))
        except Exception:
            pass
    inner = getattr(m, "model", None)
    if inner is not None:
        lm = getattr(inner, "language_model", None)
        if lm is not None:
            cands.append((".model.language_model", lm))
        cands.append((".model", inner))
    lm2 = getattr(m, "language_model", None)
    if lm2 is not None:
        cands.append((".language_model", lm2))

    for name, c in cands:
        if c is not None and callable(getattr(c, "forward", None)):
            if verbose:
                print(f"[value] hidden states from {name} -> {type(c).__name__} "
                      f"(no LM head, no {262144:,}-way logits)")
            return name, c
    return "", None


def enable_kbit_training(model: Any, *, grad_ckpt: bool = True,
                         verbose: bool = True) -> Any:
    """
    Gradient checkpointing plus the input-embedding grad hook, in the order that works
    with a 4-bit base.

    Both are needed and for different reasons. Checkpointing trades one recomputation for
    the whole activation stack, which is what makes 62 layers fit. The hook exists because
    a quantised embedding has requires_grad=False, so with checkpointing the recomputed
    segment has no input requiring grad and backward raises "element 0 of tensors does not
    require grad" — a confusing error that has nothing to do with the LoRA config.
    """
    if not grad_ckpt:
        return model
    try:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False})
        if verbose:
            print("[mem] gradient checkpointing on (prepare_model_for_kbit_training, "
                  "use_reentrant=False)")
        return model
    except Exception as e:
        if verbose:
            print(f"[mem] prepare_model_for_kbit_training unavailable "
                  f"({type(e).__name__}: {e}); falling back")
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()
    except Exception as e:
        print(f"[mem] gradient checkpointing unavailable: {type(e).__name__}: {e}")
        return model
    try:
        model.enable_input_require_grads()
    except Exception:
        pass
    if verbose:
        print("[mem] gradient checkpointing on")
    return model


def cuda_report(tag: str = "") -> str:
    try:
        import torch
        if not torch.cuda.is_available():
            return f"[mem] {tag} no CUDA"
        a = torch.cuda.memory_allocated() / 2 ** 30
        r = torch.cuda.memory_reserved() / 2 ** 30
        t = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        return f"[mem] {tag} allocated {a:.2f} GiB / reserved {r:.2f} GiB / card {t:.1f} GiB"
    except Exception as e:
        return f"[mem] {tag} unavailable ({type(e).__name__})"


def attn_default() -> str:
    """
    sdpa, unless RL_ATTN says otherwise.

    Set RL_ATTN=eager only for a Gemma-2 checkpoint, which soft-caps attention logits and
    therefore needs the unfused path to be numerically correct. Gemma 3 does not soft-cap.
    """
    v = os.environ.get("RL_ATTN", "").strip().lower()
    return v if v in ("sdpa", "eager", "flash_attention_2") else "sdpa"


if __name__ == "__main__":
    # A 3-line preflight worth running on the node before queueing stage B: it says which
    # dtype kwarg will be used (one shard load vs two) and which attention path.
    try:
        print(f"transformers version tuple : {transformers_version()}")
    except Exception as e:
        print(f"transformers version tuple : unavailable ({type(e).__name__}: {e})")
    try:
        print(f"dtype kwarg                : {dtype_kwarg()}")
    except Exception as e:
        print(f"dtype kwarg                : unavailable ({type(e).__name__}: {e})")
    print(f"attn_implementation        : {attn_default()}")
    print(cuda_report("now"))
