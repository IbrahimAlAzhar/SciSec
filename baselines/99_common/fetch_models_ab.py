#!/usr/bin/env python3
"""
fetch_models_ab.py — download the small models the ARGAR/BTR baselines need.
Run on a LOGIN node (compute nodes have no network). Skips anything already on disk.

    python fetch_models_ab.py
    python fetch_models_ab.py --only piguard

    piguard   leolee99/PIGuard                          PIGuard classifier (custom code)
    deberta   protectai/deberta-v3-base-prompt-injection-v2   D6 (probably already there)
    gpt2      gpt2                                      D3 perplexity (probably already there)
    mpnet     sentence-transformers/all-mpnet-base-v2   PaperGuard chunk retrieval (optional;
                                                        TF-IDF is used without it)

gemma3-27b-it is the only LLM and is already in path/to/anonymous/models.
Uses snapshot_download directly (the huggingface-cli wrapper on your PATH is stale).
"""

from __future__ import annotations

import argparse
import os
import sys

MODELS = {
    "piguard": ("leolee99/PIGuard", "piguard"),
    "deberta": ("protectai/deberta-v3-base-prompt-injection-v2", "protectai-deberta-pi-v2"),
    "gpt2": ("gpt2", "gpt2"),
    "mpnet": ("sentence-transformers/all-mpnet-base-v2", "all-mpnet-base-v2"),
}
IGNORE = ["*.h5", "*.ot", "*.msgpack", "*.onnx", "*.onnx_data", "*.tflite", "*.gguf",
          "onnx/*", "openvino/*"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default="path/to/anonymous/models")
    ap.add_argument("--only", default="", choices=[""] + list(MODELS))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for v in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ.pop(v, None)
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        sys.exit(f"[FATAL] huggingface_hub not importable: {e}")
    ok = True
    for key in ([a.only] if a.only else MODELS):
        repo, name = MODELS[key]
        out = os.path.join(a.dest, name)
        have = os.path.isdir(out) and any(os.path.exists(os.path.join(out, w)) for w in
                                          ("model.safetensors", "pytorch_model.bin"))
        if have and not a.force:
            print(f"[skip] {key}: {out}")
            continue
        print(f"[get ] {key}: {repo} -> {out}")
        try:
            snapshot_download(repo_id=repo, local_dir=out, ignore_patterns=IGNORE)
            print(f"[ ok ] {key}")
        except Exception as e:
            ok = False
            print(f"[FAIL] {key}: {type(e).__name__}: {e}")
            print("       On a login node? Proxy exported (HTTPS_PROXY)? huggingface.co reachable?")
    print("\nModel directories are found automatically under --dest by the ab_*.py scripts.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
