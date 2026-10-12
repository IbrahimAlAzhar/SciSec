#!/usr/bin/env python3
"""
train_sft.py — LoRA supervised fine-tuning, one adapter, on one or two 40GB cards.

WHAT THIS TRAINS ON
    The `completion` of each record: the model is taught to produce the accepted answer
    given the role's prompt. Loss is on the COMPLETION ONLY — TRL's `prompt`/`completion`
    dataset format masks the prompt automatically. That matters more than usual here: the
    prompt contains a 6,000-character manuscript region, so training on it would spend
    almost all the gradient teaching the model to reproduce papers.

WHY 4-BIT BASE + bf16 ADAPTER
    A 27B model in bf16 is 54GB of weights before optimiser state. Under 4-bit quantisation
    it is ~16GB, leaving room for activations at 4k sequence length and the LoRA's own
    parameters in bf16. qwen3-8b fits without quantisation and is trained that way, because
    quantising a model that fits costs accuracy for memory that is not scarce.

THE ONE SETTING PEOPLE GET WRONG
    LoRA on attention only. The task's hard constraint is STRICT JSON, and the MLP is where
    formatting behaviour lives; attention-only adapters keep producing prose preambles that
    fail the parse. target_modules therefore covers both, which is why r=32 is enough.

MEMORY, COMPARED WITH DPO
    SFT runs ONE sequence per example and has no reference model, so its logits tensor is a
    quarter the size of DPO's. That is why SFT completed at max_seq_len 4096 on a single
    40GB card while DPO OOM'd on the same settings — see the arithmetic in train_dpo.py.
    If a second GPU is allocated this script will use it automatically (model parallel, one
    process), which is what ft_model.pick_device_map does.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import ft_config as C


def load_jsonl(path: str) -> List[Dict]:
    if not os.path.exists(path):
        raise SystemExit(f"[FATAL] {path} not found — run ft_datasets.py first")
    return [json.loads(l) for l in open(path) if l.strip()]


def preflight(model_key: str) -> None:
    import torch
    spec = C.MODELS[model_key]
    print("=" * 92)
    print(f"SFT — {model_key}")
    print("=" * 92)
    print(f"  weights      {spec.path}")
    print(f"  4-bit base   {spec.load_4bit}")
    print(f"  system role  {spec.system_role}")
    if not os.path.isdir(spec.path):
        raise SystemExit(f"[FATAL] model directory not found: {spec.path}")
    if not torch.cuda.is_available():
        raise SystemExit("[FATAL] no CUDA device visible")
    n = torch.cuda.device_count()
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        print(f"  gpu{i}        {p.name}  {p.total_memory/2**30:.0f} GiB")



# =========================================================================================
# ft_model.py freshness check.
#
# A stale ft_model.py is the one failure mode that survives every other guard in this
# pipeline: the job queues, starts, loads a 27B checkpoint, and only then raises
# AttributeError on a helper that the copy on disk does not have. This check runs at
# IMPORT, so --dry-run catches it in the preflight, before a GPU is held.
#
# It also catches a stale __pycache__, which looks identical from the traceback: Python
# will happily import a .pyc compiled from a previous version of the file.
# =========================================================================================
_FT_MODEL_NEEDS = 3
_FT_MODEL_API = ("config_of", "vocab_size", "set_use_cache", "pick_device_map",
                 "report_device_map", "load_base", "lora_targets", "text_only",
                 "assert_text_path", "supported_kwargs", "logits_report",
                 "weights_gib_estimate", "describe")


def _check_ft_model(FM) -> None:
    missing = [n for n in _FT_MODEL_API if not hasattr(FM, n)]
    ver = getattr(FM, "VERSION", 0)
    if not missing and ver >= _FT_MODEL_NEEDS:
        return
    import os
    where = getattr(FM, "__file__", "?")
    raise SystemExit(
        "[FATAL] ft_model.py on disk is an OLD copy.\n"
        f"        file    : {where}\n"
        f"        version : {ver}   (this script needs {_FT_MODEL_NEEDS})\n"
        + (f"        missing : {missing}\n" if missing else "")
        + "        Copy the current ft_model.py into this directory, then delete the\n"
          "        compiled cache, which can otherwise keep serving the old module:\n"
          f"            rm -rf {os.path.join(os.path.dirname(where) or '.', '__pycache__')}\n")

def main() -> int:
    p = argparse.ArgumentParser(description="LoRA SFT for one role or all roles")
    p.add_argument("--model", required=True, choices=sorted(C.MODELS))
    p.add_argument("--role", default="", help="a role name, or empty for the shared "
                                              "single adapter over all roles")
    p.add_argument("--train", default="", help="override the train jsonl")
    p.add_argument("--val", default="", help="override the val jsonl")
    p.add_argument("--out", default="", help="adapter output directory")
    p.add_argument("--epochs", type=float, default=C.TRAIN.sft_epochs)
    p.add_argument("--lr", type=float, default=C.TRAIN.sft_lr)
    p.add_argument("--lora-r", dest="lora_r", type=int, default=C.TRAIN.lora_r)
    p.add_argument("--lora-alpha", dest="lora_alpha", type=int, default=C.TRAIN.lora_alpha)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=C.TRAIN.batch_size)
    p.add_argument("--eval-batch-size", dest="eval_batch_size", type=int, default=1,
                   help="MUST stay small. Evaluation materialises logits over the whole "
                        "vocabulary for every sequence in the batch; at batch 8 that is "
                        "~20 GiB for a 262k-vocab model and the job OOMs in the eval step, "
                        "not in training.")
    p.add_argument("--no-eval", dest="do_eval", action="store_false",
                   help="skip evaluation entirely. On a tiny smoke set the eval loss says "
                        "nothing anyway, and this removes the largest memory spike.")
    p.set_defaults(do_eval=True)
    p.add_argument("--grad-accum", dest="grad_accum", type=int, default=C.TRAIN.grad_accum)
    p.add_argument("--max-seq-len", dest="max_seq_len", type=int,
                   default=C.TRAIN.max_seq_len)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="load the data, report shapes and token statistics, and stop "
                        "before touching the GPU")
    a = p.parse_args()

    mk = a.model.replace("/", "_")
    stage_dir = C.wpath(C.D_SFT, mk)
    base = a.role if a.role else "all"
    train_path = a.train or os.path.join(stage_dir, f"{base}.train.jsonl")
    val_path = a.val or os.path.join(stage_dir, f"{base}.val.jsonl")
    out_dir = a.out or C.adapter_dir(a.model, "sft", a.role or None)

    train = load_jsonl(train_path)
    val = load_jsonl(val_path) if os.path.exists(val_path) else []
    print(f"[data] train {len(train)} from {train_path}")
    print(f"[data] val   {len(val)} from {val_path}")
    if not train:
        raise SystemExit("[FATAL] the training set is empty")

    from ft_tokenizer import load_tokenizer
    import ft_model as FM
    _check_ft_model(FM)
    spec = C.MODELS[a.model]
    # NOT AutoTokenizer directly: your Mistral checkpoint has no chat_template, and
    # apply_chat_template then raises four stages into the pipeline. load_tokenizer finds
    # the model's own chat_template.jinja, or installs a known-correct one.
    tok = load_tokenizer(spec.path, a.model)

    # ---- token budget, before anything is loaded -----------------------------------
    # A silently truncated completion is the worst failure here: the JSON loses its closing
    # brace, every target in the batch becomes unparseable, and the adapter learns to emit
    # truncated JSON. So the length distribution is reported and over-long records are
    # counted, not discovered later.
    lens = []
    over = 0
    for r in train[: min(len(train), 500)]:
        try:
            text = tok.apply_chat_template(r["prompt"] + r["completion"], tokenize=False)
            n = len(tok(text).input_ids)
        except Exception as e:
            raise SystemExit(
                f"[FATAL] apply_chat_template failed: {type(e).__name__}: {e}\n"
                f"        If this mentions a system role, set system_role=False for "
                f"{a.model} in ft_config.py and REBUILD the datasets — the message shape "
                f"is baked into the jsonl.")
        lens.append(n)
        if n > a.max_seq_len:
            over += 1
    lens.sort()
    pct = lambda q: lens[min(len(lens) - 1, int(q * len(lens)))] if lens else 0
    print(f"[len] tokens per example: p50={pct(0.5)} p90={pct(0.9)} max={lens[-1] if lens else 0}"
          f"   over --max-seq-len {a.max_seq_len}: {over}/{len(lens)} sampled")
    if over > 0.1 * max(1, len(lens)):
        print(f"[len] WARNING: {100*over/len(lens):.0f}% of examples exceed the sequence "
              f"limit and will be TRUNCATED FROM THE RIGHT — which cuts the completion, "
              f"not the manuscript. Lower --region-chars in ft_trajectories.py and "
              f"regenerate, or raise --max-seq-len if memory allows.")

    if a.dry_run:
        print("\n[dry-run] data is loadable and the chat template applies. Stopping "
              "before the GPU.")
        return 0

    preflight(a.model)

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    quant = None
    if spec.load_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

    device_map = FM.pick_device_map()
    # NOT AutoModelForCausalLM directly. mistral-small-3.1-24b is a Mistral3Config, which
    # that class refuses outright ("Unrecognized configuration class"), and gemma3-27b-it
    # is a multimodal config that it accepts but returns a vision-language class for.
    model = FM.load_base(
        spec.path, quant=quant, dtype=torch.bfloat16, device_map=device_map,
        attn_implementation="eager" if "gemma" in a.model else "sdpa")
    FM.describe(model, spec.path)
    # use_cache lives on config.text_config for a multimodal config; setting only the
    # outer one creates an attribute nothing reads.
    FM.set_use_cache(model, False)

    v = FM.vocab_size(model)
    if v:
        FM.logits_report(v, a.max_seq_len, a.batch_size, concat=1,
                         weights_gib=FM.weights_gib_estimate(model, spec.load_4bit),
                         label="SFT forward")

    peft_cfg = LoraConfig(
        r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=C.TRAIN.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        # On a multimodal checkpoint the bare suffix list also matches the vision tower's
        # q_proj/k_proj/v_proj/o_proj, and those adapters never receive a gradient.
        target_modules=list(FM.lora_targets(model, C.TRAIN.target_modules)))

    keep = ("prompt", "completion")
    ds_tr = Dataset.from_list([{k: r[k] for k in keep} for r in train])
    ds_va = (Dataset.from_list([{k: r[k] for k in keep} for r in val])
             if (val and a.do_eval) else None)

    cfg_kwargs = dict(
        output_dir=out_dir,
        num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch_size,
        # THE EVAL BATCH IS WHY THE FIRST RUN OOM'd. It is a SEPARATE setting and its
        # HuggingFace default is 8 — so training ran at batch 1 and then evaluation tried
        # to materialise logits for 8 sequences at once. In fp32 after logits.float() that
        # is  8 x 2600 tokens x 262144 vocab x 4 bytes ~= 20 GiB for gemma3-27b, on a card
        # that had ~11 GiB left. The traceback points at logits.float(), which reads like a
        # model bug; it is a batch-size default.
        per_device_eval_batch_size=a.eval_batch_size,
        eval_accumulation_steps=1,
        gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=C.TRAIN.warmup_ratio,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if ds_va is not None else "no",
        bf16=C.TRAIN.bf16,
        gradient_checkpointing=C.TRAIN.grad_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=a.max_seq_len,
        seed=a.seed,
        report_to=[],
        save_total_limit=1,
    )
    cfg = SFTConfig(**FM.supported_kwargs(SFTConfig, cfg_kwargs, label="SFTConfig"))

    # TRL picks its row tokenizer from the MODEL's config, not from the data: a multimodal
    # config sends it down a path that expects an AutoProcessor and does
    # `processing_class.tokenizer`, which raises "GemmaTokenizerFast has no attribute
    # tokenizer". text_only() pins that flag off before __init__ maps the dataset.
    tr = FM.text_only(SFTTrainer)(
        model=model, args=cfg, train_dataset=ds_tr, eval_dataset=ds_va,
        processing_class=tok, peft_config=peft_cfg)
    FM.assert_text_path(tr)
    tr.train()
    tr.save_model(out_dir)
    tok.save_pretrained(out_dir)

    meta = {"model": a.model, "stage": "sft", "role": a.role or None,
            "adapter_name": C.lora_name(a.model, "sft", a.role or None),
            "n_train": len(train), "n_val": len(val), "epochs": a.epochs, "lr": a.lr,
            "lora_r": a.lora_r, "lora_alpha": a.lora_alpha,
            "max_seq_len": a.max_seq_len, "base_path": spec.path}
    with open(os.path.join(out_dir, "ft_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[saved] adapter -> {out_dir}")
    print(f"[serve] name it {meta['adapter_name']!r} in --lora-modules")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())