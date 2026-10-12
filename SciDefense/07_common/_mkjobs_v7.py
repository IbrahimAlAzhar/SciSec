#!/usr/bin/env python3
"""
_mkjobs_v7.py — write the v7 PBS files.

    python _mkjobs_v7.py

v7 has THREE stages and two of them do less work than v6's, because v6's versions were
re-implementations of code that already existed:

    A  score     CPU ONLY. Reads llim_agents_v4_SFT_DPO's samples.jsonl and adds reward
                 + credit. Generates nothing.
    B  train     GRPO on those groups. The only GPU stage that is genuinely new.
    C  evaluate  runs v5's OWN runner_def.py with --vllm-model <adapter>. No inference
                 code of our own at all.
    D  mechint   CPU. The attention prior, the abstention gate and the layer report.

The shared preamble, locator, server lifecycle and log-directory machinery are imported
from _mkjobs.py so there is one copy of each.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _mkjobs import (MODELS, PREAMBLE, HELPERS, LOCATOR, SERVER, GPU_CHECK,
                     header, model_block, write)

JOBS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jobs")

LOG = r'''# =====================================================================================
# PER-RUN LOG DIRECTORY. stdout and stderr in SEPARATE files, created automatically.
#
#     job_logs/@@STAGE@@__<...>__<jobid>/{output.txt, error.txt, meta.txt}
#
# An EMPTY error.txt means the job had no errors — the fastest check there is. The jobid
# is in the directory name so a re-run never overwrites the attempt before it.
# =====================================================================================
JOBID="${PBS_JOBID:-local$$}"
JOBID="${JOBID%%.*}"
RUN_DIR="$CODE_DIR/job_logs/@@RUNNAME@@"
mkdir -p "$RUN_DIR"
OUT_FILE="$RUN_DIR/output.txt"; ERR_FILE="$RUN_DIR/error.txt"
META_FILE="$RUN_DIR/meta.txt"
: > "$ERR_FILE"
{
    echo "stage        @@STAGE@@"
    echo "jobid        ${PBS_JOBID:-<interactive>}"
    echo "node         $(hostname -s)"
    echo "started      $(date)"
    echo "code_dir     $CODE_DIR"
    echo "model        ${MODEL:-n/a}"
    echo "ablation     ${ABL:-n/a}"
    echo "gpu          $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
} > "$META_FILE"
exec > >(tee -a "$OUT_FILE") 2> >(tee -a "$ERR_FILE" >&2)
ln -sfn "$RUN_DIR" "$CODE_DIR/job_logs/latest_@@STAGE@@_${TAG}"
type stop_server >/dev/null 2>&1 || stop_server () { return 0; }
finish () {
    local rc=$?
    stop_server
    { echo "exit_code    $rc"; echo "ended        $(date)"; } >> "$META_FILE"
    echo; echo "==================================================================="
    if [[ "$rc" -eq 0 ]]; then echo "OK (exit 0)"; else echo "FAILED (exit $rc)"; fi
    echo "  output  $OUT_FILE"
    echo "  errors  $ERR_FILE   ($(wc -l < "$ERR_FILE" 2>/dev/null || echo 0) lines)"
    echo "==================================================================="
    sleep 2
}
trap finish EXIT
'''


def log(stage: str, runname: str) -> str:
    return LOG.replace("@@STAGE@@", stage).replace("@@RUNNAME@@", runname)


# ======================================================================================
A = r'''
# =====================================================================================
# STAGE A — SCORE V4's ROLLOUTS. Nothing is generated here. CPU ONLY.
#
#     qsub -v ABL=G_grpo_prov jobs/job_v7_a_@@SHORT@@_score.pbs
#
# llim_agents_v4_SFT_DPO already samples k completions per (document, role) from the
# served policy, at temperature, using the prompts the adapters were trained on, and
# scores each against gold. That IS a GRPO rollout: same prompt, same policy, sampled
# together. v6 re-implemented it and the re-implementation dropped v5's leader
# confidence gate, reporting FPR 0.797 against the published 0.107.
#
# So v7 reads what V4 wrote and adds only reward and credit — the parts that are new.
#
# CONTROLS
#   ABL=<key>     which ablation (fixes the reward kind and the credit rule)
#   ITER=0        GRPO iteration
#   MECHINT=1     add the attention-grounded process term (rows P, Q, M)
#   SAMPLES=...   override V4's trajectories/<model>/samples.jsonl
#   DOCS=...      override V4's traindata_shared/train_documents.csv
# =====================================================================================
ABL=${ABL:-G_grpo_prov}
ITER=${ITER:-0}
MECHINT=${MECHINT:-0}
SEED=${SEED:-0}
RUNTAG="$TAG"
SAMPLES=${SAMPLES:-$V4_FT_DIR/trajectories/$MODEL/samples.jsonl}
DOCS=${DOCS:-$V4_FT_DIR/traindata_shared/train_documents.csv}
FLASH_CSV=${FLASH_CSV:-$AD_ROOT/lim_flash_igcg_wb_bb/output_fusion_gemma3-27b-it/seed2/dataset_primary_500_fusion_gemma3-27b-it_seed2.csv}
MECHFLAG=""
[[ "$MECHINT" == "1" ]] && MECHFLAG="--mechint-csv $FLASH_CSV"

@@LOG@@
@@HELPERS@@
@@LOCATOR@@

run python -u rl_selftest.py

echo "[input] samples   $SAMPLES"
echo "[input] documents $DOCS"
have "$SAMPLES" || {
    echo "[FATAL] $SAMPLES not found."
    echo "        v7 does NOT generate rollouts. Run V4's rollout stage first:"
    echo "            cd $V4_FT_DIR && qsub -v MODEL=$MODEL job_ft_a_rollout.pbs"
    exit 1; }

run python -u v7_score_rollouts.py --model "$MODEL" --ablation "$ABL" \
    --samples "$SAMPLES" --documents "$DOCS" --iteration "$ITER" \
    $MECHFLAG --seed "$SEED"

echo
echo "==================================================================="
echo "SCORED. Next:"
echo "    qsub -v ABL=$ABL,ITER=$ITER jobs/job_v7_b_@@SHORT@@_train.pbs"
echo "==================================================================="
'''

Cc = r'''
# =====================================================================================
# STAGE C — EVALUATE with v5's OWN runner_def.py. There is no inference code in v7.
#
#     qsub -v ABL=G_grpo_prov jobs/job_v7_c_@@SHORT@@_infer.pbs
#
# vLLM's LoRA selection IS the `model` field, and runner_def.py already takes
# --vllm-model. So an adapter is evaluated by v5's own defense.detect() -- same gates,
# same calibration, same memory, same corpus -- with ONE argument changed. The row then
# differs from your published baseline by the adapter and by nothing else.
#
# Rows A/B/C/D are IMPORTED by v7_table.py from results you already have. This job
# refuses to run them: re-running an experiment whose numbers are in your paper can only
# move them.
#
# CONTROLS
#   ABL=<key>      which ablation row this is. Sets the adapter name AND the out dir.
#   ADAPTER=<name> evaluate a DIFFERENT adapter under this ablation's row. For comparing
#                  two training runs of the same ablation -- e.g. an inert lr=1e-6
#                  adapter you renamed against its lr=2e-5 replacement. The out dir gets
#                  the adapter's name appended so the two rows cannot overwrite one
#                  another, and v7_table.py keeps them apart.
#   MAX_LORAS=5    vLLM preallocates a GPU slot per LoRA at startup, from the KV cache
#   LIMIT=0        cap the documents for a pilot (runner_def calls this --max-papers)
# =====================================================================================
ABL=${ABL:-G_grpo_prov}
ADAPTER=${ADAPTER:-}
MAX_LORAS=${MAX_LORAS:-5}
LIMIT=${LIMIT:-0}
SEED=${SEED:-0}
RUNTAG="$TAG"
V5_RUNNER="$V5_DIR/runner_def.py"
# =====================================================================================
# THESE DEFAULTS ARE COPIED FROM $V5_DIR/job_01_gemma3_27b.pbs, WHICH PRODUCED YOUR
# PUBLISHED ROWS. Every one of them was wrong here before, and none of them fails loudly.
#
#   MEMORY_DIR    job_01: $ROOT/defense/llm_agents_v4/shared   ("the same original V4
#                 shared/memory.json used by SFT/DPO"). This job used $V5_DIR/shared,
#                 which has no memory.json at all -- hence the error you saw. Building a
#                 fresh one there would NOT have fixed it: a different memory means
#                 different retrieval margins on every document, so the row would have
#                 differed from the baseline by the memory AND the adapter, and the
#                 comparison would measure their sum.
#   ATTACKED_CSV  job_01 pins SEED2, not seed0. The payload text differs per seed, so the
#                 documents differ, so the corpus differs.
#   REFERENCE_DETAIL  job_01 verifies the corpus against the SFT/DPO zeroshot run and
#                 ABORTS on a mismatch. Without it nothing checks that this row and the
#                 published row scored the same documents in the same order.
#
# Change any of these only with a reason, and expect the reference check to refuse.
# =====================================================================================
BENCH=${BENCH:-$AD_ROOT/lim_bb_multimodel_500_data/output_primary500}
BENIGN=${BENIGN:-$AD_ROOT/data/final_data/dataset_primary_500.csv}
ATTACK_SEED=${ATTACK_SEED:-seed2}
ATTACKED_CSV=${ATTACKED_CSV:-$BENCH/$MODEL/$ATTACK_SEED/metrics_detail.csv}
MEMORY_DIR=${MEMORY_DIR:-$AD_ROOT/defense/llm_agents_v4/shared}
REFERENCE_DETAIL=${REFERENCE_DETAIL:-$V4_FT_DIR/runs/$MODEL/zeroshot/stage1_detail.csv}
# job_01's own switches. CALIBRATE=1 and STAGE2=0 are ITS defaults, so they are the
# settings behind the published numbers.
CALIBRATE=${CALIBRATE:-1}
STAGE2=${STAGE2:-0}
BTR_SIGNATURES=${BTR_SIGNATURES:-1}

@@LOG@@
@@HELPERS@@
@@GPU_CHECK@@
@@LOCATOR@@

run python -u rl_selftest.py
[[ -f "$V5_RUNNER" ]] || { echo "[FATAL] $V5_RUNNER not found"; exit 1; }

ADAPTER_NAME=$(python - "$MODEL" "$ABL" <<'PYN'
import sys, rl_config as C
ab = C.ABL[sys.argv[2]]
print("" if (ab.stage is None or ab.imported) else C.lora_name(sys.argv[1], ab.stage))
PYN
)
if [[ -z "$ADAPTER_NAME" ]]; then
    echo "[FATAL] $ABL is an IMPORTED row. v7_table.py reads it from disk; it must"
    echo "        never be re-run. Re-running a result that is already in your paper"
    echo "        can only move it."
    exit 1
fi
ROW_SUFFIX=""
if [[ -n "$ADAPTER" ]]; then
    echo "[abl] ADAPTER override: $ADAPTER_NAME -> $ADAPTER"
    ADAPTER_NAME="$ADAPTER"
    # The out dir has to differ too, or the override row silently overwrites the real
    # one's stage1_detail.csv and the table reports whichever ran last.
    ROW_SUFFIX="__$ADAPTER"
fi
echo "[abl] $ABL -> adapter $ADAPTER_NAME"

# `find -L` FOLLOWS SYMLINKS. Without -L this loop returns nothing on the symlinked
# view, the server starts with ZERO adapters, and the run silently reports the base
# model under a fine-tuned label.
VIEW="$RL_WORK/adapters_view_$TAG"
rm -rf "$VIEW"; mkdir -p "$VIEW"
N_ADAPT=0
while IFS= read -r cfg; do
    d=$(dirname "$cfg")
    case "$(basename "$d")" in
        "$(echo "$MODEL" | tr '/' '_')"*) ln -sfn "$d" "$VIEW/$(basename "$d")"; N_ADAPT=$((N_ADAPT+1)) ;;
    esac
done < <(find -L "$RL_WORK/adapters" -maxdepth 2 -name adapter_config.json 2>/dev/null | sort)
echo "[serve] $N_ADAPT adapter(s) in $VIEW"
[[ "$N_ADAPT" -eq 0 ]] && { echo "[FATAL] no adapters found. Run stage B first."; exit 1; }

export VLLM_MAX_LORAS="$MAX_LORAS"
start_server "$VIEW" "c_infer"

run python - "$VLLM_URL" "$ADAPTER_NAME" <<'PYS'
import json, sys, urllib.request
url, want = sys.argv[1], sys.argv[2]
with urllib.request.urlopen(url + "/models", timeout=60) as r:
    served = [m["id"] for m in json.load(r).get("data", [])]
print("[serve] served:", served)
if want not in served:
    print(f"[FATAL] {want!r} is NOT served. Every document would be scored by the BASE")
    print( "        model and this row would report the baseline under a fine-tuned")
    print( "        label -- which is not recoverable after the fact.")
    sys.exit(1)
print(f"[serve] {want} confirmed")
PYS

OUT_DIR="$RL_WORK/runs/$(echo "$MODEL" | tr '/' '_')/${ABL}${ROW_SUFFIX}"
mkdir -p "$OUT_DIR"

# ---- every input job_01 requires, checked before the 27B server starts ---------------
for f in "$BENIGN" "$ATTACKED_CSV" "$MEMORY_DIR/memory.json" "$REFERENCE_DETAIL"; do
    [[ -s "$f" ]] || {
        echo "[FATAL] missing input: $f"
        echo
        echo "        These are the inputs $V5_DIR/job_01_gemma3_27b.pbs used for your"
        echo "        published rows. If memory.json is the missing one, do NOT rebuild it"
        echo "        with --mode index: a freshly built memory has different payloads and"
        echo "        a different paper split, so the row would differ from the baseline by"
        echo "        the MEMORY as well as the adapter and the comparison would be void."
        echo "        Find the memory your published runs used:"
        echo "            ls -l \$AD_ROOT/defense/llm_agents_v4/shared/memory.json"
        echo "            ls -l \$AD_ROOT/defense/llm_agents_v5_final/shared/memory.json"
        echo "        and pass it:  qsub -v MEMORY_DIR=/that/dir ..."
        exit 1; }
done
echo "[input] benign        $BENIGN"
echo "[input] attacked      $ATTACKED_CSV"
echo "[input] memory        $MEMORY_DIR  (read-only; NO index rebuild)"
echo "[input] reference     $REFERENCE_DETAIL"

# ---- the corpus, built by V5's own prepare_comparison.py (CPU) -----------------------
# job_01 does this, and it is what makes --reference-detail able to prove the documents
# match the published run. Building the corpus a different way is how v6 became
# incomparable.
run python -u "$V5_DIR/prepare_comparison.py" --project-root "$AD_ROOT" \
    --benign-csv "$BENIGN" --attacked-csv "$ATTACKED_CSV" \
    --reference-detail "$REFERENCE_DETAIL" --sft-dir "$V4_FT_DIR" \
    --out-dir "$OUT_DIR/prepared" --id-column id --max-papers 1000000 --seed 0
# runner_def.py has NO --limit. Its document cap is --max-papers. `--limit 50` aborted
# the run with an argparse error after the 27B server had already come up, which is the
# most expensive possible way to learn a flag name.
LIMFLAG=""
[[ "$LIMIT" -gt 0 ]] && LIMFLAG="--max-papers $LIMIT"

# --memory-dir IN FULL, not --memory. runner_def has BOTH --memory-dir and
# --memory-models, so `--memory` is an ambiguous prefix and argparse refuses it:
#     runner_def.py: error: ambiguous option: --memory could match --memory-dir, --memory-models
# Abbreviated flags are a trap in general here: an option added to runner_def later can
# turn a prefix that works today into an error, so every flag below is spelled out.
# job_01's own switches, assembled the same way it does.
FLAGS=""
[[ "$CALIBRATE" == "1" ]]      && FLAGS="$FLAGS --calibrate --fpr-budget 0.02"
[[ "$BTR_SIGNATURES" == "0" ]] && FLAGS="$FLAGS --no-btr-signatures"
[[ "$STAGE2" == "0" ]]         && FLAGS="$FLAGS --stage1-only"

# ======================================================================================
# THIS COMMAND IS job_01's, WITH ONE ARGUMENT CHANGED: --vllm-model.
#
# job_01 serves the BASE model and passes --vllm-model "$MODEL_DIR". This serves the same
# base with LoRAs attached and passes the ADAPTER's name, which is how vLLM selects an
# adapter. Everything else -- corpus, memory, calibration, cascade depth, workers,
# min_conf, seed -- is identical, so the row differs from the published baseline by the
# adapter and by nothing else. That is the entire premise of v7, and it only holds if this
# command stays a copy of job_01's.
# ======================================================================================
run python -u "$V5_RUNNER" --mode defend --input-mode sft-dpo \
    --eval-records "$OUT_DIR/prepared/eval_records.jsonl" \
    --reference-detail "$REFERENCE_DETAIL" \
    --memory-dir "$MEMORY_DIR" --benign-csv "$BENIGN" \
    --attacked-csv "$ATTACKED_CSV" --id-column id --threat insertion \
    --model "$MODEL" --backend vllm \
    --vllm-url "$VLLM_URL" --vllm-model "$ADAPTER_NAME" \
    --out-dir "$OUT_DIR" --max-rounds 3 --workers INSTRUCTION PRESSURE VOICE \
    --min-conf 0.5 --keep-transcripts --seed "$SEED" --save-every 5 $FLAGS $LIMFLAG
stop_server

have "$OUT_DIR/stage1_detail.csv" || { echo "[FATAL] no stage1_detail.csv written"; exit 1; }
soft python -u v7_table.py --model "$MODEL"

echo
echo "==================================================================="
echo "DONE.  $OUT_DIR/stage1_detail.csv"
echo "Table: $RL_WORK/final/V7_TABLE.txt   (imported rows + this one)"
echo "==================================================================="
'''

MECH = r'''
# =====================================================================================
# STAGE D — MECHANISTIC PRIOR. CPU ONLY, minutes. Run this FIRST.
#
#     qsub jobs/job_v7_d_mechint.pbs
#
# Everything it needs is already recorded: insert_attention__<cond> and
# sink_attention__<cond> in the fusion CSV. No forward passes.
#
# It answers, before any GPU is spent, whether the attention signal can carry a reward
# at all: if attacks do not attract materially more attention than LENGTH-MATCHED FILLER,
# rows P and Q have nothing to stand on and that is the finding.
# =====================================================================================
FLASH_CSV=${FLASH_CSV:-$AD_ROOT/lim_flash_igcg_wb_bb/output_fusion_gemma3-27b-it/seed2/dataset_primary_500_fusion_gemma3-27b-it_seed2.csv}
FPR_BUDGET=${FPR_BUDGET:-0.05}
LAYER_SCAN=${LAYER_SCAN:-}
TAG=cpu
MODEL=n/a
ABL=mechint

@@LOG@@
@@HELPERS@@
@@LOCATOR@@

[[ -f "$FLASH_CSV" ]] || { echo "[FATAL] $FLASH_CSV not found"; exit 1; }
run python -u rl_selftest.py
run python -u v7_mechint.py --flash-csv "$FLASH_CSV" ${LAYER_SCAN:+--layer-scan "$LAYER_SCAN"}
soft python -u rl_flash.py --flash-csv "$FLASH_CSV" --fpr-budget "$FPR_BUDGET"
for LVL in text_only query white_box; do
    soft python -u rl_invariance.py --flash-csv "$FLASH_CSV" --access "$LVL"
done

echo
echo "==================================================================="
echo "Outputs in $RL_WORK/mechint/ and $RL_WORK/final/"
echo "Read the SEPARATION line first: under 1 filler sd means the attention"
echo "signal cannot carry a reward on this corpus, and rows P/Q should be"
echo "reported as a negative result rather than trained."
echo "==================================================================="
'''


SWEEP = r'''
# =====================================================================================
# STAGE E — HYPERPARAMETER SWEEP. ONE job, the whole grid, sequentially.
#
#     qsub -v ABL=G_grpo_prov jobs/job_v7_e_@@SHORT@@_sweep.pbs
#
# WHY ONE JOB AND NOT SIXTEEN
#   Sixteen qsubs chained with -W depend hold the queue for the same wall time, mail you
#   sixteen times, and leave sixteen log directories to read. The grid is sequential
#   anyway -- one GPU, one config at a time -- so it belongs inside one allocation with
#   one transcript. rl_sweep.py is then only a REPORT tool; it submits nothing.
#
# CONTROLS
#   ABL=<key>       the ablation whose rollouts to train on (stage A must have run)
#   MAX_STEPS=40    updates per point. Divergence shows up in the first few dozen, so a
#                   short run rules a config out for 1/7th of the cost of a full one.
#   LRS / BETAS / ACCUMS   override the grid, space-separated
#   SKIP_DONE=1     leave points that already have a grpo_summary.json alone, so a job
#                   that hit walltime can be resubmitted and will continue
#
# EVERY POINT WRITES ITS OWN DIRECTORY
#   adapters/sweep/<ABL>__<tag>/  plus hparams.json. The canonical
#   adapters/<model>-<stage>/ is NOT touched: a sweep must never overwrite the adapter
#   stage C evaluates, which is exactly how the lr=1e-6 run was lost.
# =====================================================================================
ABL=${ABL:-G_grpo_prov}
ITER=${ITER:-0}
MAX_STEPS=${MAX_STEPS:-40}
CLIP=${CLIP:-0.2}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-4096}
INTRA=${INTRA:-1}
INIT_STAGE=${INIT_STAGE:-sft}
SEED=${SEED:-0}
KL_MAX=${KL_MAX:-0}
KL_CLAMP=${KL_CLAMP:-5.0}
SKIP_DONE=${SKIP_DONE:-1}
RUNTAG="$TAG"

# The grid. Measured anchors: lr 1e-6 -> mean_kl 0.0009 (inert); 2e-5 -> 321 (diverged).
LRS=${LRS:-"1e-6 3e-6 6e-6 1.2e-5"}
BETAS=${BETAS:-"0.04 0.15"}
ACCUMS=${ACCUMS:-"4 16"}

@@LOG@@
@@HELPERS@@
@@GPU_CHECK@@
@@LOCATOR@@

run python -u rl_selftest.py

# -------------------------------------------------------------------------------------
# VERSION PREFLIGHT. The file-completeness check at the top proves the files EXIST; it
# says nothing about which VERSION of them is on disk. A stale train_grpo.py without
# --kl-max/--kl-clamp makes argparse reject every one of the 16 points with rc=2, and the
# whole sweep "finishes" in seven seconds with 16 failures and nothing to show. Asking
# --help which flags it has costs one second and turns that into one clear message.
# -------------------------------------------------------------------------------------
TG_HELP=$(python train_grpo.py --help 2>&1)
MISSING_FLAGS=""
for fl in --kl-max --kl-clamp --out-dir --max-steps; do
    case "$TG_HELP" in *"$fl"*) ;; *) MISSING_FLAGS="$MISSING_FLAGS $fl" ;; esac
done
if [[ -n "$MISSING_FLAGS" ]]; then
    echo "[FATAL] the train_grpo.py on disk does not accept:$MISSING_FLAGS"
    echo "        $CODE_DIR/train_grpo.py is an OLDER version than this job expects."
    echo "        Copy the updated train_grpo.py AND rl_config.py into $CODE_DIR."
    echo "        (rl_config.py carries the kl_max / kl_clamp defaults train_grpo reads,"
    echo "         so copying only one of the two fails on an AttributeError instead.)"
    echo
    echo "        Nothing was trained. Without this check all 16 points would have"
    echo "        failed with argparse rc=2 and the job would have exited 0."
    exit 1
fi
echo "[preflight] train_grpo.py accepts --kl-max, --kl-clamp, --out-dir, --max-steps"

ROLL_DIR="$RL_WORK/rl_trajectories/$MODEL/${ABL}_iter${ITER}"
have "$ROLL_DIR/rollouts.jsonl" || {
    echo "[FATAL] $ROLL_DIR/rollouts.jsonl not found. Run stage A for $ABL first:"
    echo "            qsub -v ABL=$ABL,ITER=$ITER jobs/job_v7_a_@@SHORT@@_score.pbs"
    exit 1; }

INIT_DIR="$RL_WORK/adapters/$(echo "$MODEL" | tr '/' '_')-$INIT_STAGE"
INIT_FLAG=""
[[ -d "$INIT_DIR" ]] && INIT_FLAG="--init-adapter $INIT_DIR"
echo "[init] ${INIT_FLAG:-FRESH adapter (RL from base, NOT the 'SFT then RL' row)}"

SWEEP_ROOT="$RL_WORK/adapters/sweep"
mkdir -p "$SWEEP_ROOT"

# KL_MAX defaults to 0 (the trust-region abort DISABLED) for a sweep only. The whole point
# is to find out which settings diverge, so aborting on divergence would throw away the
# measurement that the sweep exists to make. A normal stage B run keeps it on at 10.
echo "[sweep] ABL=$ABL  MAX_STEPS=$MAX_STEPS  kl_max=$KL_MAX (0 = record divergence, do not abort)"
N_PTS=0
for LR in $LRS; do for BETA in $BETAS; do for GA in $ACCUMS; do
    N_PTS=$((N_PTS+1))
done; done; done
echo "[sweep] $N_PTS points"

I=0
N_OK=0
N_SKIP=0
N_FAIL=0
T_START=$(date +%s)
for LR in $LRS; do
 for BETA in $BETAS; do
  for GA in $ACCUMS; do
    I=$((I+1))
    # Tag built with printf %g so 1.2e-5 and 1e-6 cannot collapse to the same string.
    LRT=$(printf '%g' "$LR" | tr -d '+' | sed 's/e-0/e-/; s/\./p/')
    BT=$(printf '%g' "$BETA" | sed 's/\./p/')
    TAGP="lr${LRT}_b${BT}_ga${GA}"
    OUT="$SWEEP_ROOT/${ABL}__${TAGP}"

    echo
    echo "################################################################"
    echo "# [$I/$N_PTS] $TAGP   lr=$LR beta_kl=$BETA grad_accum=$GA"
    echo "# $(date)   elapsed $(( ($(date +%s)-T_START)/60 )) min"
    echo "################################################################"

    if [[ "$SKIP_DONE" == "1" && -s "$OUT/grpo_summary.json" ]]; then
        echo "[skip] already done: $OUT/grpo_summary.json"
        N_SKIP=$((N_SKIP+1))
        continue
    fi
    mkdir -p "$OUT"
    # EVERY expansion is ${VAR:-...}, even though all of them are set at the top of this
    # file. Under `set -u` one unset variable inside a heredoc makes bash abandon the
    # expansion AFTER it has already truncated the target, so the result is a silent
    # zero-byte hparams.json and the loop carries on -- the settings for that point are
    # then unrecoverable. With defaults the file is always valid JSON, and a `null` says
    # plainly which value went missing.
    cat > "$OUT/hparams.json" <<HP
{
  "ablation": "${ABL:-null}", "iter": ${ITER:-null}, "model": "${MODEL:-null}",
  "tag": "${TAGP:-null}",
  "lr": ${LR:-null}, "beta_kl": ${BETA:-null}, "grad_accum": ${GA:-null},
  "clip_eps": ${CLIP:-null},
  "max_steps": ${MAX_STEPS:-null}, "max_seq_len": ${MAX_SEQ_LEN:-null},
  "intra_steps": ${INTRA:-null},
  "kl_max": ${KL_MAX:-null}, "kl_clamp": ${KL_CLAMP:-null}, "seed": ${SEED:-null},
  "init_stage": "${INIT_STAGE:-null}", "jobid": "${JOBID:-null}",
  "sweep_point": "${I:-0}/${N_PTS:-0}"
}
HP
    python -c "import json,sys; json.load(open(sys.argv[1]))" "$OUT/hparams.json" \
        || echo "[warn] $OUT/hparams.json is not valid JSON"

    # `soft`, not `run`: a diverged point is a RESULT, not a reason to abandon the other
    # fifteen. The failure is recorded and the loop continues.
    soft python -u train_grpo.py --model "$MODEL" --ablation "$ABL" \
        --rollout-dir "$ROLL_DIR" $INIT_FLAG --out-dir "$OUT" \
        --lr "$LR" --beta-kl "$BETA" --clip-eps "$CLIP" \
        --kl-max "$KL_MAX" --kl-clamp "$KL_CLAMP" \
        --grad-accum "$GA" --max-seq-len "$MAX_SEQ_LEN" \
        --intra-steps "$INTRA" --max-steps "$MAX_STEPS" --seed "$SEED"

    if [[ -s "$OUT/grpo_summary.json" ]]; then
        N_OK=$((N_OK+1))
        python - "$OUT/grpo_summary.json" <<'PYK'
import json, sys
d = json.load(open(sys.argv[1]))
kl = d.get("mean_kl")
band = ("INERT" if kl < 0.005 else "USABLE" if kl <= 0.1 else
        "HOT" if kl <= 1.0 else "TOO FAR" if kl <= 10 else "DIVERGED")
print(f"[point] mean_kl {kl:.4f}  p50 {d.get('kl_p50', float('nan')):.4f}  "
      f"min_rho {d.get('min_ratio', float('nan')):.2e}  "
      f"collapsed {d.get('collapsed_frac', 0):.2%}  -> {band}")
PYK
    else
        echo "[point] FAILED — no grpo_summary.json written"
        N_FAIL=$((N_FAIL+1))
    fi
  done
 done
done

echo
echo "==================================================================="
echo "SWEEP DONE — $N_OK trained, $N_SKIP skipped, $N_FAIL failed, "
echo "             $(( ($(date +%s)-T_START)/60 )) minutes"
echo
soft python -u rl_sweep.py --report --pick --ablation "$ABL"
echo
echo "The winner is a SHORT run ($MAX_STEPS updates). Re-run it at full length with"
echo "the canonical adapter name before evaluating -- the --pick line above does that."
echo "==================================================================="

# A sweep that trained NOTHING is a failed job. Reporting exit 0 for 16/16 failures is
# how a job that did no work reads as a success in the log and in the -m abe mail.
if [[ "$N_OK" -eq 0 && "$N_SKIP" -eq 0 ]]; then
    echo
    echo "[FATAL] every point failed. Nothing was trained. The per-point [warn] lines"
    echo "        above carry the reason; error.txt has the tracebacks."
    exit 1
fi
'''


def fill(body: str, *, tag: str, short: str, stage: str, runname: str,
         gpu: bool = True) -> str:
    return (body.replace("@@LOG@@", log(stage, runname).strip("\n"))
                .replace("@@HELPERS@@", HELPERS.strip("\n"))
                .replace("@@LOCATOR@@", LOCATOR.strip("\n"))
                .replace("@@GPU_CHECK@@", GPU_CHECK.strip("\n") if gpu else "")
                .replace("@@TAG@@", tag)
                .replace("@@SHORT@@", short))


def main() -> int:
    os.makedirs(JOBS, exist_ok=True)
    import _mkjobs as MK
    n = 0
    for short, m in MODELS.items():
        blk = model_block(m)
        # A: CPU
        txt = (header(f"v7a_{short}", "06:00:00", 0, "48gb", ncpus=4)
               + PREAMBLE + blk
               + fill(A, tag=m["tag"], short=short, gpu=False, stage="a_score",
                      runname='a_score__${RUNTAG}__${ABL}__iter${ITER}__${JOBID}'))
        write(os.path.join(JOBS, f"job_v7_a_{short}_score.pbs"), txt); n += 1
        # B: unchanged from v6 (GRPO), only renamed
        txt = (header(f"v7b_{short}", "36:00:00", m["ngpus"], m["mem"])
               + PREAMBLE + blk
               + MK.fill(MK.B, tag=m["tag"], short=short)
                   .replace("jobs/job_rl_c_", "jobs/job_v7_c_")
                   # v6's stage A was job_rl_a_<short>_ROLLOUT.pbs because it generated
                   # rollouts. v7's is job_v7_a_<short>_SCORE.pbs because it scores V4's.
                   # Renaming only the prefix printed a path that does not exist, which
                   # sends you looking for a missing job file instead of running stage A.
                   .replace("jobs/job_rl_a_@@SHORT@@_rollout.pbs",
                            "jobs/job_v7_a_@@SHORT@@_score.pbs")
                   .replace("jobs/job_rl_b_", "jobs/job_v7_b_")
                   .replace("jobs/job_rl_a_", "jobs/job_v7_a_")
                   .replace("_a_{s}_rollout.pbs".format(s=short),
                            "_a_{s}_score.pbs".format(s=short)))
        write(os.path.join(JOBS, f"job_v7_b_{short}_train.pbs"), txt); n += 1
        # C: v5's runner_def
        txt = (header(f"v7c_{short}", "48:00:00", m["ngpus"], m["mem"])
               + PREAMBLE + blk + SERVER.replace("@@TAG@@", m["tag"])
               + fill(Cc, tag=m["tag"], short=short, stage="c_infer",
                      runname='c_infer__${RUNTAG}__${ABL}__${JOBID}'))
        write(os.path.join(JOBS, f"job_v7_c_{short}_infer.pbs"), txt); n += 1
        # E: the hyperparameter sweep, one job for the whole grid
        txt = (header(f"v7e_{short}", "24:00:00", m["ngpus"], m["mem"])
               + PREAMBLE + blk
               + fill(SWEEP, tag=m["tag"], short=short, stage="e_sweep",
                      runname='e_sweep__${RUNTAG}__${ABL}__${JOBID}'))
        write(os.path.join(JOBS, f"job_v7_e_{short}_sweep.pbs"), txt); n += 1

    txt = (header("v7d_mech", "04:00:00", 0, "48gb", ncpus=4)
           + PREAMBLE + fill(MECH, tag="cpu", short="cpu", gpu=False,
                             stage="d_mechint", runname='d_mechint__${JOBID}'))
    write(os.path.join(JOBS, "job_v7_d_mechint.pbs"), txt); n += 1

    print(f"\n{n} PBS files in {JOBS}")
    print("\nORDER (gemma shown)")
    print("  1. qsub jobs/job_v7_d_mechint.pbs                        # CPU, minutes")
    print("  2. qsub -v ABL=G_grpo_prov jobs/job_v7_a_gemma_score.pbs # CPU, minutes")
    print("  3. qsub -v ABL=G_grpo_prov jobs/job_v7_b_gemma_train.pbs # GPU")
    print("  4. qsub -v ABL=G_grpo_prov jobs/job_v7_c_gemma_infer.pbs # GPU, v5 runner")
    print("  repeat 2-4 per ablation; A/B/C/D rows are imported, never run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())