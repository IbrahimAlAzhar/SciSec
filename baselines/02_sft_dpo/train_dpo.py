#!/usr/bin/env python3
"""
train_dpo.py — LoRA DPO on same-prompt preference pairs, starting from the SFT adapter.

WHY THE FIRST RUN OOM'd, AND WHAT CHANGED
=======================================================================================
The gemma3-27b DPO run died at step 0 inside compute_ref_log_probs:

    torch.OutOfMemoryError: Tried to allocate 114.00 MiB.
    GPU 0 has 39.52 GiB of which 44.88 MiB is free; 38.47 GiB in use.

That is not a data problem and not a bug in the pipeline -- it is arithmetic that no
traceback shows you:

    DPO concatenates chosen and rejected into ONE forward, so a batch of 1 is 2 sequences.
    gemma3's vocabulary is 262,144 tokens. At max_length 4096:

        logits, bf16                       2 x 4096 x 262144 x 2 B  =  4 GiB
        accelerate's fp32 copy             ... x 4 B                =  8 GiB
        the fp32 log-softmax               ... x 4 B                =  8 GiB
                                                                      ------
                                                                      20 GiB
        4-bit weights                                                 16 GiB
                                                                      ------
                                                                      36 GiB   on a 40 GiB card

SFT survived the same settings because SFT runs ONE sequence, not two, and never holds a
reference forward at the same time. So "SFT worked, DPO OOM'd" is expected, not surprising.

THE FIX, and why one 40GB card is enough after it:

  1. LOGITS OVER THE COMPLETION ONLY. The DPO loss reads log-probabilities of the
     COMPLETION tokens and nothing else. With a ~3300-token prompt and a ~700-token
     completion, 80% of that 20 GiB is built, converted to fp32, log-softmaxed and thrown
     away. TRL can pass `logits_to_keep` to the model so only the tail is materialised:
     20 GiB becomes 3.8, and 16 + 3.8 = 19.8 GiB of 40. This is the whole fix.
  2. precompute_ref_log_probs. The reference pass runs once, up front, under no_grad,
     instead of inside every training step next to the policy forward. That is the exact
     call the traceback died in.
  3. If a second GPU is allocated anyway, ft_model.pick_device_map splits the model across
     both. That is accelerate's naive model parallelism -- NOT Ray, NOT torchrun -- and it
     needs select=1:ngpus=2, one chunk. It is no longer required.

For scale: the qwen2.5-3b worker script that runs DPO on one card needs 5.8 GiB of weights
and 11.6 GiB of logits, 17.4 total. It is not doing anything cleverer; it is nine times
smaller with a 1.7x smaller vocabulary.

THE TWO THINGS THAT DECIDE WHETHER DPO WORKS AT ALL
=======================================================================================
1. THE LEARNING RATE. DPO at SFT's learning rate diverges: the implicit reward grows
   without bound, the policy drifts off the reference, and the model emits degenerate text
   that satisfies the preference and nothing else. 5e-6 is two orders of magnitude below
   the SFT default here, and that is not a typo.

2. WHICH ADAPTER IS THE REFERENCE. With PEFT, TRL uses the BASE MODEL WITH THE ADAPTER
   DISABLED as the implicit reference -- so continuing to train the SFT adapter means the
   KL term pulls back toward the *pre-SFT* model, not toward the SFT model. That is usually
   what you want (it keeps total drift bounded), but it must be a decision rather than an
   accident, so --ref-is-sft makes the alternative explicit.

--from-sft is the default and is how ablation (c) is built. --no-from-sft trains DPO on the
raw base model, which is the `dpo_only` control row -- the one that says whether the
supervised stage was worth its compute.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import ft_config as C


def load_jsonl(path: str) -> List[Dict]:
    if not os.path.exists(path):
        raise SystemExit(f"[FATAL] {path} not found — run ft_datasets.py first")
    return [json.loads(l) for l in open(path) if l.strip()]


def pair_report(rows: List[Dict]) -> None:
    from collections import Counter
    n = len(rows)
    lab = Counter(int(r.get("label", 0)) for r in rows)
    rej = Counter((r.get("rejected_source") or "?").split(":")[0] for r in rows)
    syn = sum(v for k, v in rej.items() if k == "synth")
    print(f"[pairs] {n} pairs   attacked={lab.get(1,0)} clean={lab.get(0,0)}")
    print(f"[pairs] rejected side: " +
          ", ".join(f"{k}={v}" for k, v in rej.most_common()))
    if n and syn / n > 0.5:
        print(f"[pairs] WARNING: {100*syn/n:.0f}% of the rejected samples are SYNTHETIC. "
              f"DPO will mostly teach the model to avoid mistakes it was not making. "
              f"Raise --n-samples in ft_trajectories.py so more real errors are captured.")
    same = sum(1 for r in rows
               if (r.get("chosen") or [{}])[-1].get("content") ==
                  (r.get("rejected") or [{}])[-1].get("content"))
    if same:
        print(f"[pairs] {same} pair(s) have IDENTICAL chosen and rejected text — those "
              f"contribute nothing and are dropped.")


def length_report(rows: List[Dict], tok, max_len: int, max_prompt: int,
                  sample: int = 200) -> None:
    """How many pairs will be truncated, and where.

    Printed because truncation here is silent and asymmetric: TRL cuts the PROMPT from the
    LEFT (keeping the instructions nearest the answer and dropping the start of the
    manuscript region) and the COMPLETION from the right. A completion cut from the right
    loses its closing brace, so the JSON target becomes unparseable and the pair teaches
    the model to emit truncated JSON. Knowing the rate is the difference between an informed
    trade and a silent one.
    """
    lens_full: List[int] = []
    lens_prompt: List[int] = []
    for r in rows[: min(len(rows), sample)]:
        try:
            p = tok.apply_chat_template(r["prompt"], tokenize=False)
            np_ = len(tok(p).input_ids)
            best = 0
            for side in ("chosen", "rejected"):
                t = tok.apply_chat_template(r["prompt"] + r[side], tokenize=False)
                best = max(best, len(tok(t).input_ids))
        except Exception as e:
            print(f"[len] could not measure lengths ({type(e).__name__}: {e}) — skipping "
                  f"the report, training is unaffected")
            return
        lens_prompt.append(np_)
        lens_full.append(best)
    if not lens_full:
        return
    lens_full.sort()
    lens_prompt.sort()
    q = lambda xs, p: xs[min(len(xs) - 1, int(p * len(xs)))]
    over = sum(1 for n in lens_full if n > max_len)
    over_p = sum(1 for n in lens_prompt if n > max_prompt)
    print(f"[len] pair length  p50={q(lens_full,0.5)} p90={q(lens_full,0.9)} "
          f"max={lens_full[-1]}   over --max-seq-len {max_len}: {over}/{len(lens_full)}")
    print(f"[len] prompt only  p50={q(lens_prompt,0.5)} p90={q(lens_prompt,0.9)} "
          f"max={lens_prompt[-1]}   over max_prompt_length {max_prompt}: "
          f"{over_p}/{len(lens_prompt)}")
    if over > 0.25 * len(lens_full):
        print(f"[len] NOTE: {100*over/len(lens_full):.0f}% of pairs are longer than the "
              f"limit. The prompt is cut from the LEFT, so what is lost is the start of the "
              f"manuscript region, not the instructions. If that share looks too high, "
              f"raise --max-seq-len and give the job a second GPU rather than accepting it.")



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
    p = argparse.ArgumentParser(description="LoRA DPO for one role or all roles")
    p.add_argument("--model", required=True, choices=sorted(C.MODELS))
    p.add_argument("--role", default="")
    p.add_argument("--train", default="")
    p.add_argument("--val", default="")
    p.add_argument("--out", default="")
    p.add_argument("--sft-adapter", dest="sft_adapter", default="",
                   help="adapter to continue from (default: the SFT adapter for this "
                        "model/role)")
    p.add_argument("--no-from-sft", dest="from_sft", action="store_false",
                   help="train DPO on the RAW base model. This is the dpo_only control "
                        "row that prices the supervised stage.")
    p.set_defaults(from_sft=True)
    p.add_argument("--ref-is-sft", dest="ref_is_sft", action="store_true",
                   help="use the SFT adapter as the KL reference instead of the base "
                        "model. NOTE: this loads a SECOND full copy of the base model and "
                        "roughly doubles weight memory — on a 27B checkpoint that alone "
                        "needs two cards.")
    p.add_argument("--stage-name", dest="stage_name", default="",
                   help="adapter stage label; defaults to 'dpo', or 'dpo_only' with "
                        "--no-from-sft")
    p.add_argument("--epochs", type=float, default=C.TRAIN.dpo_epochs)
    p.add_argument("--lr", type=float, default=C.TRAIN.dpo_lr)
    p.add_argument("--beta", type=float, default=C.TRAIN.dpo_beta)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=C.TRAIN.batch_size)
    p.add_argument("--eval-batch-size", dest="eval_batch_size", type=int, default=1)
    p.add_argument("--no-eval", dest="do_eval", action="store_false")
    p.set_defaults(do_eval=True)
    p.add_argument("--grad-accum", dest="grad_accum", type=int, default=C.TRAIN.grad_accum)
    # DEFAULTS LOWER THAN SFT, DELIBERATELY. DPO holds two sequences and a reference
    # forward; at gemma3's 262k vocabulary each 1024 tokens of length costs 5 GiB of logits.
    p.add_argument("--max-seq-len", dest="max_seq_len", type=int, default=4096,
                   help="total prompt+completion length. With logits-to-keep on (the "
                        "default) this no longer drives memory, because logits are "
                        "materialised over the completion only. With --no-logits-to-keep "
                        "it does, linearly.")
    p.add_argument("--max-prompt-len", dest="max_prompt_len", type=int, default=0,
                   help="0 = max_seq_len minus 768 for the completion")
    p.add_argument("--no-precompute-ref", dest="precompute_ref", action="store_false",
                   help="compute reference log-probs inside each training step instead of "
                        "once up front. Uses less disk and more peak GPU memory.")
    p.set_defaults(precompute_ref=True)
    p.add_argument("--no-logits-to-keep", dest="logits_to_keep", action="store_false",
                   help="compute logits over the WHOLE sequence instead of only the "
                        "completion. Costs ~4x the logits memory; only useful if your TRL "
                        "or the model's forward() mishandles the option.")
    p.set_defaults(logits_to_keep=True)
    p.add_argument("--merge-sft", dest="merge_sft", action="store_true",
                   help="merge the SFT adapter into the base weights and train a FRESH "
                        "LoRA on top. This moves the KL reference from the pre-SFT base "
                        "model to the SFT model. On a 4-bit base it dequantises and "
                        "requantises every merged layer, which adds rounding error to "
                        "weights you just spent three hours training — see the note in "
                        "the module docstring before using it on gemma3-27b.")
    p.add_argument("--lora-r", dest="lora_r", type=int, default=C.TRAIN.lora_r)
    p.add_argument("--lora-alpha", dest="lora_alpha", type=int, default=C.TRAIN.lora_alpha)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", dest="dry_run", action="store_true")
    a = p.parse_args()

    # Before anything else: a stale ft_model.py must not be discovered three
    # screens of output later, or after a GPU has been handed over.
    import ft_model as FM
    _check_ft_model(FM)

    max_prompt = a.max_prompt_len or max(512, a.max_seq_len - 768)

    stage = a.stage_name or ("dpo" if a.from_sft else "dpo_only")
    mk = a.model.replace("/", "_")
    base = a.role if a.role else "all"
    train_path = a.train or C.wpath(C.D_DPO, mk, f"{base}.train.jsonl")
    val_path = a.val or C.wpath(C.D_DPO, mk, f"{base}.val.jsonl")
    out_dir = a.out or C.adapter_dir(a.model, stage, a.role or None)
    sft_dir = a.sft_adapter or C.adapter_dir(a.model, "sft", a.role or None)

    train = load_jsonl(train_path)
    val = load_jsonl(val_path) if os.path.exists(val_path) else []
    print("=" * 92)
    print(f"DPO — {a.model}  role={a.role or 'ALL'}  stage={stage}")
    print("=" * 92)
    print(f"[data] train {len(train)}  val {len(val)}")
    print(f"[cfg]  max_seq_len={a.max_seq_len}  max_prompt_length={max_prompt}  "
          f"batch={a.batch_size}  grad_accum={a.grad_accum}")
    pair_report(train)
    train = [r for r in train
             if (r.get("chosen") or [{}])[-1].get("content") !=
                (r.get("rejected") or [{}])[-1].get("content")]
    if not train:
        raise SystemExit("[FATAL] no usable pairs")

    if a.from_sft:
        if not os.path.isdir(sft_dir):
            raise SystemExit(
                f"[FATAL] SFT adapter not found: {sft_dir}\n"
                f"        Run train_sft.py for this model/role first, or pass "
                f"--no-from-sft to train the dpo_only control instead.")
        print(f"[init] continuing from the SFT adapter: {sft_dir}")
    else:
        print("[init] training from the RAW base model (dpo_only control)")

    from ft_tokenizer import load_tokenizer

    spec = C.MODELS[a.model]
    tok = load_tokenizer(spec.path, a.model)
    length_report(train, tok, a.max_seq_len, max_prompt)

    # The memory verdict, BEFORE the model is loaded — so a run that cannot fit says so in
    # 5 seconds instead of 90 seconds of checkpoint loading followed by an OOM.
    try:
        cfg = FM.config_of(spec.path)
        v = int(getattr(getattr(cfg, "text_config", None), "vocab_size", 0)
                or getattr(cfg, "vocab_size", 0) or 0)
    except Exception:
        v = 0
    if v:
        # 27B in nf4 ~= 16 GiB (measured: the run that died reported 38.47 GiB with
        # 20 GiB of logits). Close enough for a warning.
        eff_len = 768 if a.logits_to_keep else a.max_seq_len
        FM.logits_report(v, eff_len, a.batch_size, concat=2, weights_gib=16.0,
                         label=("DPO forward, logits over the COMPLETION only"
                                if a.logits_to_keep else
                                "DPO forward (chosen+rejected, FULL sequence)"))
        if a.logits_to_keep:
            print(f"[mem]   (--no-logits-to-keep would compute them over all "
                  f"{a.max_seq_len} tokens instead: "
                  f"{2*a.max_seq_len*v*10/2**30:.1f} GiB)")

    if a.dry_run:
        print("\n[dry-run] pairs load, lengths measured, the adapter path resolves. "
              "Stopping before the GPU.")
        return 0

    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer

    quant = None
    if spec.load_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

    device_map = FM.pick_device_map()
    # See ft_model.py: mistral3 is rejected by AutoModelForCausalLM, gemma3 is accepted but
    # comes back as a vision-language class.
    model = FM.load_base(
        spec.path, quant=quant, dtype=torch.bfloat16, device_map=device_map,
        attn_implementation="eager" if "gemma" in a.model else "sdpa")
    FM.describe(model, spec.path)
    FM.set_use_cache(model, False)

    peft_cfg = None
    if a.from_sft:
        if a.merge_sft:
            # YOUR WORKER SCRIPT'S APPROACH: fold the SFT adapter into the weights, then
            # train a fresh LoRA. The effect that matters is on the KL REFERENCE. TRL's
            # implicit PEFT reference is "this model with the adapter disabled", so once
            # SFT is merged in, disabling the new adapter lands on the SFT model rather
            # than the pre-SFT base — an SFT-anchored reference for free, without the
            # second copy of the weights that --ref-is-sft needs.
            #
            # THE CAVEAT, and it is why this is not the default here. Your worker script
            # loads qwen2.5-3b in bf16 (USE_4BIT = False), where merging is exact. This
            # pipeline loads gemma3-27b in nf4, and merging into a 4-bit layer means
            # dequantise -> add the delta -> requantise, once per module. PEFT warns about
            # it, and the rounding lands on weights you just spent three hours training.
            from peft import PeftModel as _PM
            m = _PM.from_pretrained(model, sft_dir, is_trainable=False)
            print("[init] merging the SFT adapter into the base weights "
                  + ("(4-bit base: this requantises every merged layer and introduces "
                     "rounding error)" if spec.load_4bit else "(bf16 base: exact)"))
            model = m.merge_and_unload()
            peft_cfg = LoraConfig(
                r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=C.TRAIN.lora_dropout,
                bias="none", task_type="CAUSAL_LM",
                target_modules=list(FM.lora_targets(model, C.TRAIN.target_modules)))
            print("[ref] KL reference = the SFT model (it is now the base)")
        else:
            # is_trainable=True so DPO continues the SAME adapter rather than stacking a
            # second one on top of a frozen first. Two stacked adapters cannot be merged
            # into one file for serving, and vLLM loads one adapter per request.
            model = PeftModel.from_pretrained(model, sft_dir, is_trainable=True)
    else:
        peft_cfg = LoraConfig(
            r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=C.TRAIN.lora_dropout,
            bias="none", task_type="CAUSAL_LM",
            target_modules=list(FM.lora_targets(model, C.TRAIN.target_modules)))

    ref = None
    precompute = a.precompute_ref
    if a.ref_is_sft and a.from_sft and not a.merge_sft:
        ref_base = FM.load_base(
            spec.path, quant=quant, dtype=torch.bfloat16, device_map=device_map,
            attn_implementation="eager" if "gemma" in a.model else "sdpa")
        ref = PeftModel.from_pretrained(ref_base, sft_dir, is_trainable=False)
        print("[ref] KL reference = the frozen SFT adapter (a SECOND copy of the weights)")
    elif not a.merge_sft:
        print("[ref] KL reference = the base model with the adapter disabled (TRL's "
              "implicit PEFT reference)")
    if precompute:
        print("[ref] precompute_ref_log_probs=ON — the reference pass runs once, up front, "
              "under no_grad, instead of inside every training step next to the policy "
              "forward. That is where the first run OOM'd.")

    keep = ("prompt", "chosen", "rejected")
    ds_tr = Dataset.from_list([{k: r[k] for k in keep} for r in train])
    ds_va = (Dataset.from_list([{k: r[k] for k in keep} for r in val])
             if (val and a.do_eval) else None)

    cfg_kwargs = dict(
        output_dir=out_dir,
        num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch_size,
        # The eval batch defaults to 8 in HuggingFace and evaluation materialises
        # full-vocabulary logits. DPO scores TWO completions per example, so the spike is
        # twice the SFT one. Keep it at 1.
        per_device_eval_batch_size=a.eval_batch_size,
        eval_accumulation_steps=1,
        gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr,
        beta=a.beta,
        lr_scheduler_type="cosine",
        warmup_ratio=C.TRAIN.warmup_ratio,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if ds_va is not None else "no",
        bf16=C.TRAIN.bf16,
        gradient_checkpointing=C.TRAIN.grad_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=a.max_seq_len,
        max_prompt_length=max_prompt,
        precompute_ref_log_probs=precompute,
        seed=a.seed,
        report_to=[],
        save_total_limit=1,
    )
    if a.logits_to_keep:
        # THE SETTING THAT MAKES ONE 40GB CARD ENOUGH.
        #
        # The DPO loss only ever reads log-probabilities of the COMPLETION tokens. Logits
        # over the prompt are computed, converted to fp32, log-softmaxed -- and thrown
        # away. With a 3328-token prompt and a ~700-token completion, that is 80% of the
        # single largest tensor in the run, allocated for nothing.
        #
        # TRL can pass `logits_to_keep` down to the model's forward so only the tail is
        # materialised. For gemma3 (vocab 262,144) at length 4096 that turns 20 GiB of
        # logits into under 4 GiB -- which is the whole difference between the OOM and a
        # run that finishes on the single card you already have.
        #
        # The option has been spelled two ways across TRL releases. Both are offered and
        # supported_kwargs() keeps whichever this version defines; if it keeps NEITHER, the
        # check below says so loudly rather than letting the job discover it at step 0.
        cfg_kwargs["use_logits_to_keep"] = True
        cfg_kwargs["use_num_logits_to_keep"] = True

    kept = FM.supported_kwargs(DPOConfig, cfg_kwargs, label="DPOConfig")
    if a.logits_to_keep:
        got = [k for k in ("use_logits_to_keep", "use_num_logits_to_keep") if k in kept]
        if got:
            print(f"[mem] {got[0]}=True — logits are materialised over the completion "
                  f"only, not all {a.max_seq_len} tokens.")
        else:
            print("[mem] WARNING: this TRL version has no logits_to_keep option, so logits "
                  "will cover the FULL sequence.")
            print(f"[mem]          On one 40GB card that means gemma3-27b needs "
                  f"max_seq_len <= 2560. Either lower --max-seq-len or allocate a second "
                  f"GPU (select=1:ngpus=2).")
    cfg = DPOConfig(**kept)

    # THE GEMMA CRASH (the first one). TRL chooses its row processor from the model config:
    #     dataset.map(self.tokenize_row if not self.is_vision_model else self.process_row)
    # and process_row opens with `processor, tokenizer = processing_class,
    # processing_class.tokenizer`. gemma3-27b-it is a multimodal config, so the flag was
    # True, and the error came back as "GemmaTokenizerFast has no attribute tokenizer" --
    # which names the tokenizer and hides the cause. text_only() pins the flag off inside
    # __setattr__, the only hook that runs before __init__ maps the dataset.
    tr = FM.text_only(DPOTrainer)(
        model=model, ref_model=ref, args=cfg, train_dataset=ds_tr,
        eval_dataset=ds_va, processing_class=tok, peft_config=peft_cfg)
    FM.assert_text_path(tr)
    tr.train()
    tr.save_model(out_dir)
    tok.save_pretrained(out_dir)

    meta = {"model": a.model, "stage": stage, "role": a.role or None,
            "adapter_name": C.lora_name(a.model, stage, a.role or None),
            "from_sft": bool(a.from_sft), "sft_adapter": sft_dir if a.from_sft else None,
            "ref_is_sft": bool(a.ref_is_sft), "n_train": len(train), "n_val": len(val),
            "epochs": a.epochs, "lr": a.lr, "beta": a.beta,
            "max_seq_len": a.max_seq_len, "max_prompt_length": max_prompt,
            "precompute_ref_log_probs": bool(precompute), "base_path": spec.path}
    with open(os.path.join(out_dir, "ft_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[saved] adapter -> {out_dir}")
    print(f"[serve] name it {meta['adapter_name']!r} in --lora-modules")
    print("\n[read] In the training log, watch `rewards/margins`. It should rise and then "
          "flatten. If `rewards/chosen` and `rewards/rejected` BOTH fall steeply, the "
          "policy is drifting off the reference rather than learning a preference — lower "
          "--lr or raise --beta.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())