#!/usr/bin/env python3
"""
fetch_baseline_models.py — download the two models the defense baselines need, without
touching the huggingface-cli, which is broken on your cluster.

    # on a LOGIN node (compute nodes have no network)
    cd $AD_ROOT/defense_baselines
    python fetch_baseline_models.py

    python fetch_baseline_models.py --dest path/to/anonymous/models
    python fetch_baseline_models.py --only deberta        # skip gpt2

WHY NOT huggingface-cli
    huggingface_hub v1.0 REMOVED the `huggingface-cli` entry point and the
    `huggingface_hub.commands` module with it; the CLI is now called `hf`. Your PATH still
    has a wrapper script from an older install:

        path/to/bin/huggingface-cli

    That wrapper does `from huggingface_hub.commands.huggingface_cli import main`, the
    module is gone in v1.x, and you get ModuleNotFoundError. The wrapper is stale; the
    library is fine.

    This script calls snapshot_download() directly, which exists in every version from
    0.x to 1.x, so it does not care which CLI you have or whether you have one at all.

WHAT IT DOWNLOADS
    protectai/deberta-v3-base-prompt-injection-v2   ~740 MB   baseline D6
    gpt2                                            ~550 MB   the perplexity screen in D3

    Only model weights, config and tokenizer -- no .h5, .ot, .msgpack or .onnx, which
    roughly halves the transfer.
"""

from __future__ import annotations

import argparse
import os
import sys

MODELS = {
    "deberta": dict(
        repo="protectai/deberta-v3-base-prompt-injection-v2",
        dirname="protectai-deberta-pi-v2",
        needs=("config.json",),
        weights=("model.safetensors", "pytorch_model.bin"),
        note="baseline D6. The ProtectAI project is ARCHIVED; cite the model card."),
    "gpt2": dict(
        repo="gpt2",
        dirname="gpt2",
        needs=("config.json",),
        weights=("model.safetensors", "pytorch_model.bin"),
        note="perplexity screen in D3. Optional -- run with NO_PPL=1 to skip it."),
}

# Skip the framework duplicates. A repo often ships the same weights four times over.
IGNORE = ["*.h5", "*.ot", "*.msgpack", "*.onnx", "*.onnx_data", "*.tflite",
          "rust_model.ot", "flax_model.msgpack", "*.gguf"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default="path/to/anonymous/models",
                    help="where to put them. Defaults to the directory that already "
                         "holds gemma3_27b_it / qwen3_8b / mistral_small_3_1_24b.")
    ap.add_argument("--only", default="", choices=["", "deberta", "gpt2"])
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    a = ap.parse_args()

    # A download cannot work with the offline flags set. They are correct inside a PBS
    # job and wrong here, so clear them for this process only.
    for v in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        if os.environ.pop(v, None):
            print(f"[env] cleared {v} for this download (it is correct inside the job, "
                  f"not here)")

    try:
        import huggingface_hub as H
        from huggingface_hub import snapshot_download
    except Exception as e:
        sys.exit(f"[FATAL] huggingface_hub is not importable: {type(e).__name__}: {e}\n"
                 f"        pip install -U huggingface_hub")
    print(f"[hub] huggingface_hub {H.__version__} at {os.path.dirname(H.__file__)}")
    try:
        import huggingface_hub.commands  # noqa: F401
        print("[hub] this version still has the old `commands` module, so "
              "`huggingface-cli` would work too")
    except Exception:
        print("[hub] no `commands` module -- v1.x. The CLI is now `hf`, and the stale")
        print("[hub] `huggingface-cli` wrapper on your PATH will keep failing. Ignore it;")
        print("[hub] this script does not use it.")

    os.makedirs(a.dest, exist_ok=True)
    print(f"[dest] {a.dest}")

    want = [a.only] if a.only else list(MODELS)
    ok = True
    for key in want:
        spec = MODELS[key]
        out = os.path.join(a.dest, spec["dirname"])
        have = os.path.isdir(out) and any(
            os.path.exists(os.path.join(out, w)) for w in spec["weights"])
        if have and not a.force:
            print(f"\n[skip] {key}: already at {out}")
            continue

        print(f"\n[get ] {key}  {spec['repo']}")
        print(f"       {spec['note']}")
        try:
            snapshot_download(repo_id=spec["repo"], local_dir=out,
                              ignore_patterns=IGNORE)
        except Exception as e:
            ok = False
            msg = f"{type(e).__name__}: {e}"
            print(f"[FAIL] {key}: {msg}")
            low = msg.lower()
            if any(k in low for k in ("offline", "connection", "proxy", "403",
                                      "forbidden", "timeout", "timed out", "resolve",
                                      "network", "ssl", "certificate")):
                print("       That is a NETWORK failure, not a bad repo name. Check, in")
                print("       this order:")
                print("         1. you are on a LOGIN node — compute nodes have no")
                print("            outbound access, which is why the jobs set")
                print("            HF_HUB_OFFLINE=1")
                print("         2. your site proxy is exported, e.g.")
                print("              export HTTPS_PROXY=http://<proxy>:<port>")
                print("              export HTTP_PROXY=$HTTPS_PROXY")
                print("         3. huggingface.co is reachable at all:")
                print("              curl -sS -o /dev/null -w '%{http_code}\\n' \\")
                print("                   https://huggingface.co/api/models/gpt2")
                print("       If your cluster blocks it outright, copy the two model")
                print("       directories in from a machine that can reach it, or run")
                print("       the detector job with NO_PPL=1 and skip D6 — the regex")
                print("       half of D3 needs no download and is the more informative")
                print("       half anyway.")
            continue

        # verify rather than trust: a partial snapshot loads as a missing-file error
        # three hours into a job.
        miss = [f for f in spec["needs"] if not os.path.exists(os.path.join(out, f))]
        if not any(os.path.exists(os.path.join(out, w)) for w in spec["weights"]):
            miss.append("model.safetensors or pytorch_model.bin")
        if miss:
            ok = False
            print(f"[FAIL] {key}: downloaded but missing {miss}")
        else:
            size = sum(os.path.getsize(os.path.join(dp, f))
                       for dp, _, fs in os.walk(out) for f in fs)
            print(f"[ ok ] {key}: {size/2**20:.0f} MiB at {out}")

    print("\n" + "=" * 78)
    if not ok:
        print("Some downloads failed. See above.")
        return 1
    d_de = os.path.join(a.dest, MODELS['deberta']['dirname'])
    d_gp = os.path.join(a.dest, MODELS['gpt2']['dirname'])
    print("Done. Submit the detector job with these paths:\n")
    print(f"  qsub -v DEBERTA={d_de},PPL_MODEL={d_gp} job_def_detectors.pbs\n")
    print("Or edit the two default lines near the top of job_def_detectors.pbs:")
    print(f"  PPL_MODEL=${{PPL_MODEL:-{d_gp}}}")
    print(f"  DEBERTA=${{DEBERTA:-{d_de}}}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())