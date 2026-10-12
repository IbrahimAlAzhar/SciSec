#!/usr/bin/env python3
"""
v8_lora_concat.py — make the `sft_dpo_merged` adapter SERVABLE on the raw base model.

THE PROBLEM
    train_dpo.py --merge-sft folds the SFT adapter into the base weights and trains a fresh
    LoRA on top. That LoRA is a delta relative to (base + SFT), so serving it on the raw
    base -- which is what vLLM loads -- silently drops the SFT stage, and the row labelled
    "SFT merged, then DPO" would really be "DPO without SFT".

THE FIX, EXACT
    Two LoRAs on one module add:  W + s1 B1 A1 + s2 B2 A2  =  W + [s1 B1 | s2 B2] [A1 ; A2]
    so one LoRA of rank r1 + r2 with scaling 1 reproduces both. This file writes that
    adapter (rank 64 from two rank-32 ones). Serve with --max-lora-rank >= 64.
    The one approximation is upstream and stated in train_dpo.py: on a 4-bit base the
    DPO LoRA was trained against requantised merged weights.

    python v8_lora_concat.py --first adapters/_raw/<sft> --second adapters/_raw/<dpo_merged> \
                             --out adapters/<served name>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from typing import Dict, Tuple

import torch


def _load(d: str) -> Tuple[Dict[str, torch.Tensor], Dict]:
    cfg = json.load(open(os.path.join(d, "adapter_config.json")))
    st = os.path.join(d, "adapter_model.safetensors")
    if os.path.exists(st):
        from safetensors.torch import load_file
        w = load_file(st)
    else:
        w = torch.load(os.path.join(d, "adapter_model.bin"), map_location="cpu")
    return w, cfg


def _scaling(cfg: Dict, r: int) -> float:
    alpha = float(cfg.get("lora_alpha", r))
    return alpha / (r ** 0.5) if cfg.get("use_rslora") else alpha / r


_KEY = re.compile(r"^(.*)\.lora_([AB])(?:\.default)?\.weight$")


def _split(w: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
    out: Dict[str, Dict[str, torch.Tensor]] = {}
    for k, v in w.items():
        m = _KEY.match(k)
        if not m:
            raise SystemExit(f"[FATAL] unexpected adapter key {k!r}: only plain LoRA A/B "
                             f"weights can be concatenated (no DoRA, no modules_to_save)")
        out.setdefault(m.group(1), {})[m.group(2)] = v
    return out


def concat(first: str, second: str, out: str) -> Dict:
    w1, c1 = _load(first)
    w2, c2 = _load(second)
    m1, m2 = _split(w1), _split(w2)
    r1, r2 = int(c1["r"]), int(c2["r"])
    s1, s2 = _scaling(c1, r1), _scaling(c2, r2)
    if c1.get("rank_pattern") or c2.get("rank_pattern"):
        raise SystemExit("[FATAL] rank_pattern adapters are not supported")
    new: Dict[str, torch.Tensor] = {}
    mods = sorted(set(m1) | set(m2))
    for mod in mods:
        a1, b1 = (m1.get(mod) or {}).get("A"), (m1.get(mod) or {}).get("B")
        a2, b2 = (m2.get(mod) or {}).get("A"), (m2.get(mod) or {}).get("B")
        ref_a, ref_b = (a1 if a1 is not None else a2), (b1 if b1 is not None else b2)
        dt = ref_b.dtype
        if a1 is None:                                   # module only in the second
            a1 = torch.zeros(r1, ref_a.shape[1], dtype=ref_a.dtype)
            b1 = torch.zeros(ref_b.shape[0], r1, dtype=dt)
        if a2 is None:
            a2 = torch.zeros(r2, ref_a.shape[1], dtype=ref_a.dtype)
            b2 = torch.zeros(ref_b.shape[0], r2, dtype=dt)
        A = torch.cat([a1.float(), a2.float()], 0)
        B = torch.cat([b1.float() * s1, b2.float() * s2], 1)
        new[f"{mod}.lora_A.weight"] = A.to(ref_a.dtype).contiguous()
        new[f"{mod}.lora_B.weight"] = B.to(dt).contiguous()
    cfg = dict(c1)
    cfg["r"] = r1 + r2
    cfg["lora_alpha"] = r1 + r2                       # scaling 1: s1, s2 live in B
    cfg["use_rslora"] = False
    tm = set(c1.get("target_modules") or []) | set(c2.get("target_modules") or [])
    if tm:
        cfg["target_modules"] = sorted(tm)
    os.makedirs(out, exist_ok=True)
    from safetensors.torch import save_file
    save_file(new, os.path.join(out, "adapter_model.safetensors"))
    json.dump(cfg, open(os.path.join(out, "adapter_config.json"), "w"), indent=2)
    for f in os.listdir(second):                        # tokenizer files, for completeness
        if f.startswith("tokenizer") or f in ("special_tokens_map.json", "chat_template.jinja"):
            shutil.copy2(os.path.join(second, f), os.path.join(out, f))
    meta = {"adapter_name": os.path.basename(os.path.normpath(out)),
            "built_by": "v8_lora_concat.py", "first": os.path.realpath(first),
            "second": os.path.realpath(second), "r": r1 + r2, "scalings": [s1, s2],
            "modules": len(mods), "note": "W + s1*B1A1 + s2*B2A2 as one rank-(r1+r2) LoRA"}
    json.dump(meta, open(os.path.join(out, "ft_meta.json"), "w"), indent=2)
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description="concatenate two LoRAs into one exact adapter")
    ap.add_argument("--first", required=True, help="the SFT adapter (applied first)")
    ap.add_argument("--second", required=True, help="the LoRA trained on top of the merge")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    m = concat(a.first, a.second, a.out)
    print(json.dumps(m, indent=2))
    print(f"[serve] rank {m['r']}: vLLM needs --max-lora-rank >= {m['r']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
