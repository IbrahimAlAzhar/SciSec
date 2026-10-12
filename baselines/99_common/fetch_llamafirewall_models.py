#!/usr/bin/env python3
"""
fetch_llamafirewall_models.py -- download Llama Prompt Guard 2 (LlamaFirewall's
PromptGuard scanner) to local disk, for the offline compute nodes.

    # on a LOGIN node (compute nodes have no network)
    #   1. accept the licence (once) at
    #        https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M
    #   2. create a read token at https://huggingface.co/settings/tokens
    export HF_TOKEN=hf_...
    cd $AD_ROOT/defense_baselines/llama_firewall
    python fetch_llamafirewall_models.py                 # 86M (the one the paper uses)
    python fetch_llamafirewall_models.py --also-22m      # + the 22M variant

WHY A SCRIPT
    Same reason as ../fetch_baseline_models.py: the huggingface-cli wrapper on your PATH is
    stale (huggingface_hub v1 removed it). snapshot_download() works in every version.
    The PG2 repos are GATED: without an accepted licence and a token you get 401/403,
    and this script says so instead of printing a stack trace.

The gemma-3-27b-it guard model is already on disk; nothing to fetch for it.
"""

from __future__ import annotations

import argparse
import os
import sys

MODELS = {
    "86m": dict(repo="meta-llama/Llama-Prompt-Guard-2-86M",
                dirname="Llama-Prompt-Guard-2-86M"),
    "22m": dict(repo="meta-llama/Llama-Prompt-Guard-2-22M",
                dirname="Llama-Prompt-Guard-2-22M"),
}
IGNORE = ["*.h5", "*.ot", "*.msgpack", "*.onnx", "*.onnx_data", "*.tflite", "*.gguf"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default="path/to/anonymous/models")
    ap.add_argument("--also-22m", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    for v in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        if os.environ.pop(v, None):
            print(f"[env] cleared {v} for this download (correct inside the job, not here)")
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        sys.exit(f"[FATAL] huggingface_hub not importable: {type(e).__name__}: {e}")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("[warn] no HF_TOKEN in the environment. The PromptGuard 2 repos are gated;")
        print("       unless you ran `hf auth login` before, the download will be refused.")

    want = ["86m"] + (["22m"] if a.also_22m else [])
    ok = True
    for k in want:
        spec = MODELS[k]
        out = os.path.join(a.dest, spec["dirname"])
        have = os.path.exists(os.path.join(out, "config.json")) and any(
            os.path.exists(os.path.join(out, w)) for w in ("model.safetensors",
                                                           "pytorch_model.bin"))
        if have and not a.force:
            print(f"[skip] {spec['repo']}: already at {out}")
            continue
        print(f"[get ] {spec['repo']} -> {out}")
        try:
            snapshot_download(repo_id=spec["repo"], local_dir=out, token=token,
                              ignore_patterns=IGNORE)
        except Exception as e:
            ok = False
            msg = f"{type(e).__name__}: {e}"
            print(f"[FAIL] {msg[:300]}")
            low = msg.lower()
            if any(s in low for s in ("401", "403", "gated", "access", "restricted",
                                      "authorized", "authenticat")):
                print("       GATED REPO: open the model page while logged in, accept the")
                print("       Llama 4 Community License, wait for approval (usually minutes),")
                print("       and export HF_TOKEN=<a read token> before re-running.")
            elif any(s in low for s in ("connection", "proxy", "timeout", "resolve",
                                        "network", "ssl")):
                print("       NETWORK: run this on a login node, with your site proxy exported.")
            continue
        miss = [f for f in ("config.json",) if not os.path.exists(os.path.join(out, f))]
        if not any(os.path.exists(os.path.join(out, w)) for w in ("model.safetensors",
                                                                  "pytorch_model.bin")):
            miss.append("model.safetensors")
        if miss:
            ok = False
            print(f"[FAIL] downloaded but missing {miss}")
        else:
            size = sum(os.path.getsize(os.path.join(dp, f))
                       for dp, _, fs in os.walk(out) for f in fs)
            print(f"[ ok ] {size/2**20:.0f} MiB at {out}")

    print("\n" + "=" * 78)
    if not ok:
        print("Some downloads failed (see above). D7 still runs the AlignmentCheck rows.")
        return 1
    print("Done. The job's default PG2_MODEL already points at the 86M folder:")
    print(f"  qsub -v PG2_MODEL={os.path.join(a.dest, MODELS['86m']['dirname'])} "
          f"job_d7_llamafirewall.pbs")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
