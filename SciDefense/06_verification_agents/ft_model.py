#!/usr/bin/env python3
"""
ft_model.py — load a base model for TEXT-ONLY training, even when the checkpoint is
secretly multimodal; place it across the GPUs you actually have; and print the one number
that decides whether the run survives.

WHY THIS FILE EXISTS
=======================================================================================
Two of the three reviewers here ship as vision-language checkpoints even though this
project never shows them an image:

    gemma3-27b-it          config class Gemma3Config       -> Gemma3ForConditionalGeneration
    mistral-small-3.1-24b  config class Mistral3Config     -> Mistral3ForConditionalGeneration
    qwen3-8b               config class Qwen3Config        -> Qwen3ForCausalLM   (text, fine)

That produced two failures that look unrelated:

  1. MISTRAL, at load time
         ValueError: Unrecognized configuration class Mistral3Config for this kind of
         AutoModel: AutoModelForCausalLM
     Mistral3Config is simply not in the CausalLM mapping. Gemma3Config *is* (it maps to
     the conditional-generation class), which is why gemma loaded and mistral did not.

  2. GEMMA, four stages later, inside TRL
         AttributeError: GemmaTokenizerFast has no attribute tokenizer
     TRL decides how to tokenize a row from the MODEL's config, not from the dataset:
         dataset.map(self.tokenize_row if not self.is_vision_model else self.process_row)
     and `process_row` starts with `processor, tokenizer = processing_class,
     processing_class.tokenizer`. The error names the tokenizer, which sends you looking in
     exactly the wrong place.

And then a third, which is pure arithmetic rather than a config quirk: see logits_report().

WHAT THIS FILE PROVIDES
    load_base()        tries the AutoModel classes in order and says which one worked
    lora_targets()     keeps LoRA off the vision tower
    text_only()        pins TRL's vision flag off so the text path is always taken
    set_use_cache()    reaches config.text_config, where a multimodal config hides it
    pick_device_map()  one GPU or several -- and why that is not Ray
    report_device_map()refuses to train through silent CPU offload
    supported_kwargs() survives TRL renaming config fields between releases
    logits_report()    prints the logits memory BEFORE the GPU is asked for it
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Bumped whenever a function is added that the training scripts call. They check it on
# import, so a half-copied directory fails in the preflight with a message naming the file,
# instead of 90 seconds into a GPU allocation with an AttributeError.
VERSION = 3

# Substrings that mark a submodule as part of the non-text stack. Matched against the
# dotted module path, case-insensitively.
VISION_MARKERS: Tuple[str, ...] = (
    "vision_tower", "vision_model", "vision_encoder", "visual",
    "multi_modal_projector", "mm_projector", "multimodal_projector",
    "patch_merger", "patch_conv", "image_newline", "img_projection",
    "audio_tower", "audio_encoder",
)

# Attribute names TRL (and friends) use for "this is a vision-language run". Matched with
# fullmatch, case-insensitively. Kept broad on purpose: the name has changed between TRL
# releases, and a flag we fail to pin is a crash four stages downstream.
_VISION_FLAG = re.compile(r"(is_)?(vision|vlm|multi_?modal|mm)(_model)?", re.I)


# =====================================================================================
# config inspection
# =====================================================================================
def config_of(path: str, trust_remote_code: bool = True):
    from transformers import AutoConfig
    return AutoConfig.from_pretrained(path, trust_remote_code=trust_remote_code)


def is_multimodal(cfg: Any) -> bool:
    """True when the checkpoint carries a non-text stack.

    `text_config` is the reliable signal: transformers gives a multimodal config a nested
    text config and keeps the language hyper-parameters there. Checking model_type against
    a hardcoded list instead would miss the next model you add.
    """
    if getattr(cfg, "text_config", None) is not None:
        return True
    for attr in ("vision_config", "audio_config"):
        if getattr(cfg, attr, None) is not None:
            return True
    return False


def vocab_size(model: Any) -> int:
    """The number that drives logits memory. On a multimodal config it lives on
    text_config, and reading the outer one gives a wrong (often much smaller) answer."""
    cfg = getattr(model, "config", None)
    for c in (getattr(cfg, "text_config", None), cfg):
        v = getattr(c, "vocab_size", None)
        if v:
            return int(v)
    try:
        return int(model.get_output_embeddings().weight.shape[0])
    except Exception:
        return 0


def set_use_cache(model: Any, flag: bool) -> None:
    """Turn the KV cache off for training, on BOTH configs.

    A multimodal config keeps `use_cache` on `config.text_config`. Setting it only on the
    outer config creates a new attribute that nothing reads, gradient checkpointing then
    warns and silently re-enables caching, and memory use goes up for no reason.
    """
    for c in (getattr(model, "config", None), getattr(getattr(model, "config", None),
                                                      "text_config", None)):
        if c is not None:
            try:
                c.use_cache = flag
            except Exception:
                pass


# =====================================================================================
# device placement
# =====================================================================================
def pick_device_map(verbose: bool = True):
    """One GPU -> everything on cuda:0. More than one -> split the model across them.

    THIS IS HOW YOU GET TWO CARDS, AND IT IS NOT RAY.
    Ray and tensor parallelism are a vLLM thing, for SERVING. A HuggingFace Trainer on two
    cards of one node uses `device_map="auto"`: accelerate cuts the layer stack in half,
    puts the first half on cuda:0 and the second half plus the LM head on cuda:1, and the
    forward pass hops between them. One process, no torchrun, no Ray, no accelerate launch.
    Launching under torchrun instead gives each rank a FULL copy of the model, which is
    DDP -- that improves throughput and does nothing at all for memory, which is the
    problem here.

    It also has to be ONE PBS chunk: `select=1:ngpus=2`, not `select=2`. Two chunks can
    land on two different nodes, and accelerate cannot split a model across nodes.
    """
    import torch
    n = torch.cuda.device_count()
    if n <= 1:
        if verbose:
            print("[device] 1 GPU visible -> everything on cuda:0")
        return {"": 0}
    if verbose:
        print(f"[device] {n} GPUs visible -> device_map='auto' (model parallel, one "
              f"process; NOT ray, NOT torchrun)")
    return "auto"


def report_device_map(model: Any) -> None:
    """Print where the layers landed, and refuse to train through CPU offload.

    When accelerate cannot fit a model it does not fail -- it puts the overflow on the CPU
    and keeps going. Training then works, at roughly fifty times slower, and the symptom is
    "the job is taking days", which nobody reads as an error.
    """
    dm = getattr(model, "hf_device_map", None)
    if not dm:
        return
    from collections import Counter
    counts = Counter(str(v) for v in dm.values())
    print("[device] module placement: "
          + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    bad = sorted({str(v) for v in dm.values()} & {"cpu", "disk"})
    if bad:
        raise SystemExit(
            f"[FATAL] accelerate placed part of the model on {bad}. Training would still\n"
            f"        run, at roughly 50x slower, and would look like a hang rather than\n"
            f"        an error. Ask for more GPU memory (ngpus=2 in ONE chunk:\n"
            f"        select=1:ngpus=2) or lower --max-seq-len.")


# =====================================================================================
# loading
# =====================================================================================
def _dtype_kwarg() -> str:
    """'dtype' on transformers >= 4.56, else 'torch_dtype'. RL_DTYPE_KWARG overrides."""
    forced = os.environ.get("RL_DTYPE_KWARG", "").strip()
    if forced in ("dtype", "torch_dtype"):
        return forced
    try:
        import transformers
        parts = []
        for tok in str(transformers.__version__).split(".")[:2]:
            num = "".join(ch for ch in tok if ch.isdigit())
            parts.append(int(num) if num else 0)
        return "dtype" if tuple(parts) >= (4, 56) else "torch_dtype"
    except Exception:
        return "torch_dtype"


def _free_cuda() -> None:
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def load_base(path: str, *, quant=None, dtype=None, device_map=None,
              attn_implementation: str = "sdpa", trust_remote_code: bool = True,
              verbose: bool = True) -> Any:
    """Load a base model for causal-LM training, whatever family it belongs to.

    The order matters: AutoModelForCausalLM first, because for a text model it is the only
    correct answer and for gemma3 it already returns the right (multimodal) class. The
    image-text classes are tried only when that fails, which is the mistral3 case.
    """
    import transformers

    kw: Dict[str, Any] = dict(trust_remote_code=trust_remote_code)
    if quant is not None:
        kw["quantization_config"] = quant
    if device_map is not None:
        kw["device_map"] = device_map
    if attn_implementation:
        kw["attn_implementation"] = attn_implementation

    candidates: List[Tuple[str, Any]] = []
    for name in ("AutoModelForCausalLM", "AutoModelForImageTextToText",
                 "AutoModelForVision2Seq"):
        cls = getattr(transformers, name, None)
        if cls is not None:
            candidates.append((name, cls))

    # Last resort: the exact class the checkpoint names in config.architectures.
    try:
        cfg = config_of(path, trust_remote_code)
        for arch in (getattr(cfg, "architectures", None) or []):
            cls = getattr(transformers, arch, None)
            if cls is not None:
                candidates.append((arch, cls))
    except Exception:
        pass

    errors: List[str] = []
    # v8: the spelling is resolved from the transformers VERSION first. Trying `dtype=`
    # then `torch_dtype=` blind reads all shards onto the GPU, fails AFTER loading, and
    # reads them again without freeing the first copy -- two 4-bit gemma3-27b copies is
    # 31 of 39.5 GiB, and the job then OOMs on the first forward pass looking like an
    # activation problem. (Measured in v7: "Loading checkpoint shards 12/12" printed twice.)
    first = _dtype_kwarg()
    order = (first, "torch_dtype" if first == "dtype" else "dtype")
    for name, cls in candidates:
        for dtype_key in order:
            kwargs = dict(kw)
            if dtype is not None:
                kwargs[dtype_key] = dtype
            try:
                model = cls.from_pretrained(path, **kwargs)
                if verbose:
                    print(f"[model] loaded with {name} -> {type(model).__name__}"
                          + (f"   (dtype kwarg: {dtype_key})" if dtype is not None else ""))
                return model
            except TypeError as e:
                if dtype is not None and ("dtype" in str(e) or "torch_dtype" in str(e)):
                    print(f"[model] {dtype_key}= rejected; freeing the GPU before retrying")
                    _free_cuda()
                    continue                      # try the other spelling
                errors.append(f"{name}: TypeError: {e}")
                _free_cuda()
                break
            except (ValueError, KeyError) as e:
                errors.append(f"{name}: {type(e).__name__}: {str(e).splitlines()[0]}")
                break
            if dtype is None:
                break

    raise SystemExit(
        "[FATAL] could not load the base model with any AutoModel class.\n"
        f"        path: {path}\n"
        + "".join(f"        - {e}\n" for e in errors)
        + "        If every line says 'Unrecognized configuration class', this transformers\n"
          "        version predates the model. Check `pip show transformers` inside the\n"
          "        image_lim_vllm_final environment against the model card.")


# =====================================================================================
# LoRA targeting
# =====================================================================================
def _is_vision_path(name: str) -> bool:
    low = name.lower()
    return any(m in low for m in VISION_MARKERS)


def lora_targets(model: Any, suffixes: Sequence[str], *,
                 verbose: bool = True) -> Sequence[str]:
    """Return what to pass as LoRA target_modules.

    For a text-only model: the suffix tuple unchanged, which is what PEFT expects and what
    every tutorial shows. For a multimodal model: the explicit full names of the matching
    Linear modules inside the language stack only.

    target_modules=("q_proj", ...) matches by NAME SUFFIX, and a vision tower has q_proj,
    k_proj, v_proj and o_proj too -- so the plain suffix list silently attaches adapters to
    it. Those adapters never see a gradient (no image ever enters), so they stay at their
    init value. The result is not wrong output; it is a larger adapter file, a slower serve,
    and a `trainable params` count that overstates what you trained. On gemma3-27b that is
    81 vision modules out of 515.

    Explicit names rather than a regex, because the count is then printable and checkable --
    and a silent zero-match here produces a LoRA with no trainable parameters, a loss that
    never moves, and an adapter that serves as a no-op.
    """
    import torch.nn as nn

    sfx = tuple(suffixes)
    if not is_multimodal(getattr(model, "config", object())):
        if verbose:
            print(f"[lora] text-only checkpoint; target_modules = {list(sfx)}")
        return sfx

    keep: List[str] = []
    skipped: List[str] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear) and type(mod).__name__ not in (
                "Linear4bit", "Linear8bitLt", "Params4bit", "QuantLinear"):
            continue
        short = name.rsplit(".", 1)[-1]
        if short not in sfx:
            continue
        (skipped if _is_vision_path(name) else keep).append(name)

    if verbose:
        print(f"[lora] multimodal checkpoint: {len(keep)} language modules targeted, "
              f"{len(skipped)} vision/projector modules EXCLUDED")
        if keep:
            print(f"[lora]   e.g. {keep[0]}")
        if skipped:
            print(f"[lora]   e.g. excluded {skipped[0]}")

    if not keep:
        raise SystemExit(
            "[FATAL] no LoRA target modules matched inside the language stack.\n"
            f"        looked for name suffixes: {list(sfx)}\n"
            "        Update TRAIN.target_modules in ft_config.py to the real names.")
    return keep


# =====================================================================================
# keeping TRL on the text path
# =====================================================================================
def text_only(trainer_cls: Any) -> Any:
    """Wrap a TRL trainer class so its vision flag can never become True.

    The flag is set inside Trainer.__init__, BEFORE __init__ calls _prepare_dataset, so a
    subclass that fixes it after super().__init__() is already too late -- the dataset has
    been mapped through the vision tokenizer and the AttributeError has been raised.
    Overriding __setattr__ is the only hook that runs early enough, and it also survives
    TRL renaming the attribute, which it has done before.
    """
    class _TextOnlyTrainer(trainer_cls):                     # type: ignore[misc]
        _pinned: List[str] = []

        def __setattr__(self, key: str, value: Any) -> None:
            if value and _VISION_FLAG.fullmatch(key):
                if key not in type(self)._pinned:
                    type(self)._pinned.append(key)
                    print(f"[trl] pinning {trainer_cls.__name__}.{key} = False "
                          f"(this run is text-only; the checkpoint is multimodal)")
                value = False
            super().__setattr__(key, value)

    # Belt and braces: if the class carries a vision row-processor with the same signature
    # as the text one, point it at the text one. Covers TRL versions that branch on the
    # dataset columns instead of on a flag.
    tr = getattr(trainer_cls, "tokenize_row", None)
    pr = getattr(trainer_cls, "process_row", None)
    if tr is not None and pr is not None:
        try:
            import inspect
            if (inspect.signature(inspect.unwrap(tr)).parameters.keys() ==
                    inspect.signature(inspect.unwrap(pr)).parameters.keys()):
                _TextOnlyTrainer.process_row = staticmethod(inspect.unwrap(tr))
                print(f"[trl] {trainer_cls.__name__}.process_row redirected to "
                      f"tokenize_row (identical signature; text-only run)")
        except Exception:
            pass

    _TextOnlyTrainer.__name__ = f"TextOnly{trainer_cls.__name__}"
    _TextOnlyTrainer.__qualname__ = _TextOnlyTrainer.__name__
    return _TextOnlyTrainer


def assert_text_path(trainer: Any) -> None:
    """Fail loudly if a vision flag survived. Cheap, and the alternative is a crash whose
    message names the tokenizer."""
    bad = [k for k, v in vars(trainer).items() if v and _VISION_FLAG.fullmatch(k)]
    if bad:
        raise SystemExit(
            f"[FATAL] TRL is still on its vision code path ({bad}). "
            f"ft_model.text_only() did not catch this attribute name — add it to "
            f"_VISION_FLAG in ft_model.py.")


# =====================================================================================
# TRL / transformers version drift
# =====================================================================================
def supported_kwargs(cls: Any, kwargs: Dict[str, Any], *, label: str = "",
                     verbose: bool = True) -> Dict[str, Any]:
    """Drop config keys this installed version does not accept.

    SFTConfig and DPOConfig gain and lose fields between TRL releases. Passing one that no
    longer exists is a TypeError at construction -- raised AFTER the model has loaded, which
    on a 27B checkpoint costs 90 seconds per attempt. Filtering keeps the file working
    across versions and, more usefully, PRINTS what was dropped, so a memory knob that
    silently stopped applying is visible rather than mysterious.
    """
    import inspect
    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return dict(kwargs)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    ok = {k: v for k, v in kwargs.items() if k in params}
    dropped = sorted(set(kwargs) - set(ok))
    if dropped and verbose:
        print(f"[trl] {label or cls.__name__} does not accept {dropped} in this version — "
              f"dropped. If one of those was a memory setting, it is NOT in effect.")
    return ok


# =====================================================================================
# memory arithmetic, printed before the GPU is asked for anything
# =====================================================================================
def logits_report(vocab: int, seq_len: int, batch: int, *, concat: int = 1,
                  weights_gib: float = 0.0, label: str = "") -> float:
    """Print the size of the logits tensor and what it implies. Returns GiB.

    THIS IS THE NUMBER THAT KILLED THE DPO RUN, and it appears in no traceback.

        vocab 262,144 (gemma3) x 4096 tokens x 2 sequences
        (DPO concatenates chosen and rejected into ONE forward, so the batch is doubled)

        held three times over:
            bf16 logits                        4 GiB
            accelerate's fp32 copy of them     8 GiB   <- convert_to_fp32 in the traceback
            the fp32 log-softmax               8 GiB
                                              -------
                                              20 GiB

        plus 16 GiB of 4-bit weights = 36 GiB on a 40 GiB card. The job died reporting
        38.47 GiB allocated, trying to add 114 MiB, inside compute_ref_log_probs.

    Nothing about the data was wrong. The same pipeline ran SFT to completion because SFT
    does ONE sequence, not two, and never held the reference forward at the same time.
    """
    n = concat * batch * seq_len * vocab
    gib = n * (2 + 4 + 4) / 2 ** 30
    print(f"[mem] {label or 'logits'}: vocab={vocab:,} x len={seq_len} x "
          f"{concat * batch} sequence(s)")
    print(f"[mem]   bf16 {n*2/2**30:.1f} + fp32 copy {n*4/2**30:.1f} + log-softmax "
          f"{n*4/2**30:.1f} = {gib:.1f} GiB of logits")
    try:
        import torch
        if not torch.cuda.is_available():
            return gib
        ndev = max(1, torch.cuda.device_count())
        per = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        # With a split model the logits live wherever the LM head is, so the card that
        # matters holds its share of the weights plus all of the logits.
        share = weights_gib / ndev
        need = share + gib
        print(f"[mem]   the card holding the LM head needs ~{share:.1f} GiB of weights + "
              f"{gib:.1f} GiB of logits = {need:.1f} GiB of {per:.0f} GiB   "
              f"({ndev} GPU(s) visible)")
        if need > 0.88 * per:
            budget = max(0.0, 0.80 * per - share)
            safe = int(budget * 2 ** 30 / max(1, concat * batch * vocab * 10))
            safe = max(512, (safe // 256) * 256)
            print(f"[mem]   WARNING: too close to the limit. Either give the job a second")
            print(f"[mem]            GPU (select=1:ngpus=2 -- ONE chunk, not select=2),")
            print(f"[mem]            or lower the sequence length to about {safe}.")
    except Exception:
        pass
    return gib


def weights_gib_estimate(model: Any, four_bit: bool) -> float:
    """Rough resident size of the base weights, for the report above."""
    try:
        n = sum(p.numel() for p in model.parameters())
    except Exception:
        return 0.0
    # A 4-bit model reports a packed parameter count, so bytes-per-param is close to 1.
    return n * (1.0 if four_bit else 2.0) / 2 ** 30


# =====================================================================================
# a small report, printed once per training run
# =====================================================================================
def describe(model: Any, path: str = "") -> None:
    cfg = getattr(model, "config", None)
    mt = getattr(cfg, "model_type", "?")
    tc = getattr(cfg, "text_config", None)
    print(f"[model] class {type(model).__name__}   model_type {mt}"
          + (f"   text_config {getattr(tc, 'model_type', '?')}" if tc is not None else "")
          + ("   MULTIMODAL -> training text only" if is_multimodal(cfg) else ""))
    try:
        n = sum(p.numel() for p in model.parameters())
        print(f"[model] {n/1e9:.1f}B parameters loaded   vocab {vocab_size(model):,}")
    except Exception:
        pass
    report_device_map(model)