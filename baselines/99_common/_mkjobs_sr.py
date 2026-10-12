#!/usr/bin/env python3
"""
_mkjobs_sr.py — write the SafeReview PBS files.

    python _mkjobs_sr.py            -> jobs/job_sr_*.pbs (+ .txt copies)

The preamble, the PYTHONPATH locator with its import preflight, the GPU check, the
per-run log directory and run()/soft()/have() are IMPORTED from your v7ab package's
_mkjobs.py / _mkjobs_v7.py, so these jobs fail and log exactly the way your other jobs do.
Only three things change: the code directory (defense_baselines/safereview), the module
list the preflight checks, and $V7AB_DIR on PYTHONPATH (rl_config, rl_excise, rl_load,
v7ab_report are read from there, never copied).

    stage  job                               what                                         GPU
    I      job_sr_i_inputs.pbs               v7ab inputs -> inputs/ (copied, sha256)       -
    S      job_sr_s_<m>_static.pbs           Static DPO: corpus pairs -> DPO               yes
    F      job_sr_f_<m>_round.pbs            ONE co-evolution round (ROUND=t)              yes
    D      job_sr_d_<m>_adaptive.pbs         final attacker -> adaptive test set           yes
    C      job_sr_c_<m>_eval.pbs             rows on the test set (review_detail.csv)      yes
    P      job_sr_p_promptguard.pbs          PromptGuard 2 (stage1_detail.csv)             yes
    G      job_sr_g_gpt4omini.pbs            undefended + System Defense, gpt-4o-mini      -
    T      job_sr_t_table.pbs                sr_report.py, all models                      -
"""

from __future__ import annotations

import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
V7AB_SRC = os.environ.get("V7AB_DIR") or os.path.join(
    os.environ.get("AD_ROOT",
                   "."),
    "defense/lim_agents_v7_RL_with_argar_btr")
if not os.path.exists(os.path.join(V7AB_SRC, "_mkjobs.py")):
    raise SystemExit(f"[FATAL] {V7AB_SRC}/_mkjobs.py not found; set V7AB_DIR")
sys.path.insert(0, V7AB_SRC)
import _mkjobs as MK                                    # noqa: E402
import _mkjobs_v7 as M7                                 # noqa: E402

JOBS = os.path.join(HERE, "jobs")
SR_DIR = "$AD_ROOT/defense_baselines/safereview"
MODULES = sorted(os.path.basename(p) for p in glob.glob(os.path.join(HERE, "sr_*.py")))
V7AB_NEEDS = ("rl_config.py", "rl_rewards.py", "rl_excise.py", "rl_load.py",
              "v7ab_report.py", "v7ab_inputs.py")


def preamble() -> str:
    p = MK.PREAMBLE
    old = 'CODE_DIR="${RL_CODE_DIR:-$AD_ROOT/defense/lim_agents_v7_RL}"'
    assert old in p
    p = p.replace(old, (
        '# the v7ab package this baseline READS (rl_config, rl_excise, rl_load, v7ab_report,\n'
        '# v7ab_inputs and its inputs/): never copied, so there is one copy of each\n'
        'V7AB_DIR="${V7AB_DIR:-$AD_ROOT/defense/lim_agents_v7_RL_with_argar_btr}"\n'
        'export V7AB_DIR\n'
        f'CODE_DIR="${{SR_CODE_DIR:-{SR_DIR}}}"'))
    p = p.replace("qsub -v RL_CODE_DIR=/path/to/it", "qsub -v SR_CODE_DIR=/path/to/it")
    p = p.replace("So the directory is named. RL_CODE_DIR overrides it",
                  "So the directory is named. SR_CODE_DIR overrides it")
    start = p.index("for f in rl_config.py")
    end = p.index("done", start) + len("done")
    lines, cur = [], "for f in"
    for m in MODULES:
        if len(cur) + len(m) > 80:
            lines.append(cur + " \\")
            cur = "        "
        cur += " " + m
    lines.append(cur + "; do")
    loop = "\n".join(lines) + '\n    [[ -f "$CODE_DIR/$f" ]] || MISSING="$MISSING $f"\ndone\n'
    loop += ("for f in " + " ".join(V7AB_NEEDS) + "; do\n"
             '    [[ -f "$V7AB_DIR/$f" ]] || MISSING="$MISSING $V7AB_DIR/$f"\ndone')
    p = p[:start] + loop + p[end:]
    old = 'rm -rf "$CODE_DIR/__pycache__" "$V4_FT_DIR/__pycache__" "$V5_DIR/__pycache__" 2>/dev/null'
    assert old in p
    p = p.replace(old, old.replace('"$V5_DIR/__pycache__"',
                                   '"$V5_DIR/__pycache__" "$V7AB_DIR/__pycache__"'))
    old = 'export RL_WORK="$CODE_DIR"'
    assert old in p
    p = p.replace(old, old + '\nexport SR_WORK="$CODE_DIR"')
    p += r'''
# the v8 data files the inputs stage reads (built by llim_agents_v8_dvb's data stages)
export V8_DATA_DIR="${V8_DATA_DIR:-$AD_ROOT/data/final_data/v8_defense_data}"
'''
    return p


def locator() -> str:
    loc = MK.LOCATOR
    old = "export PYTHONPATH=$CODE_DIR:$V5_DIR:$V4_FT_DIR:$V4_DIR:${PYTHONPATH:-}"
    assert old in loc
    return loc.replace(old, "export PYTHONPATH=$CODE_DIR:$V7AB_DIR:$V5_DIR:$V4_FT_DIR:$V4_DIR:"
                            "${PYTHONPATH:-}")


CPU_PATH = 'export PYTHONPATH=$CODE_DIR:$V7AB_DIR:${PYTHONPATH:-}'

SERVER = r'''
# =====================================================================================
# vLLM lifecycle. TWO base models are served by this package -- the reviewer and the
# attacker -- never at the same time (one 40 GiB card), so the base model is an argument.
# Same serve_vllm_lora.sh, same `find -L` adapter discovery, same wait-for-free-memory.
# =====================================================================================
VPID=""
VIEW=""
VLLM_MAX_LEN=${VLLM_MAX_LEN:-16384}
VLLM_GPU_UTIL=${VLLM_GPU_UTIL:-0.90}
start_server () {   # $1 base model dir, $2 quant, $3 adapters dir or "", $4 log label
    local base="$1" quant="$2" adapters="$3" slot="$4"
    local vlog="$CODE_DIR/logs/vllm_@@TAG@@_${slot}.log"
    local SERVE="$V4_FT_DIR/serve_vllm_lora.sh"
    [[ -f "$SERVE" ]] || { echo "[FATAL] $SERVE not found"; exit 1; }
    run bash "$SERVE" "$base" auto "$quant" "$VLLM_MAX_LEN" "$VLLM_GPU_UTIL" "$vlog" \
        "$adapters" "" "$TP" ray
    local port; port="$(cat "${vlog}.port")"
    export VLLM_URL="http://127.0.0.1:${port}/v1"
    VPID="$(cat "$CODE_DIR/logs"/vllm_${port}_$(hostname -s).pid 2>/dev/null || true)"
    echo "[serve] $base  adapters=${adapters:-none}  $VLLM_URL  pid=$VPID"
}

stop_server () {
    if [[ -z "${VPID:-}" ]]; then return 0; fi
    echo "[serve] stopping vLLM (pid $VPID) and waiting for GPU memory"
    kill "$VPID" 2>/dev/null
    for i in $(seq 1 60); do kill -0 "$VPID" 2>/dev/null || break; sleep 5; done
    kill -9 "$VPID" 2>/dev/null
    ray stop >/dev/null 2>&1
    for i in $(seq 1 60); do
        local used
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
               | sort -rn | head -1)
        [[ -z "$used" ]] && break
        if [[ "$used" -lt 2000 ]]; then
            echo "[serve] GPU memory free after $((i*5))s (max ${used} MiB in use)"; break
        fi
        [[ "$i" -eq 60 ]] && echo "[serve] WARNING: ${used} MiB still in use after 5 min"
        sleep 5
    done
    VPID=""
}
trap 'stop_server' EXIT
'''

SR_HELPERS = r'''
# ---- SafeReview helpers ---------------------------------------------------------------
MK="$(echo "$MODEL" | tr '/' '_')"
# A symlinked view holding ONLY the named adapters (vLLM is given one directory; the
# other reviewers' adapters in the same tree would fail to load with a shape mismatch).
adapter_view () {   # $1 = label, rest = adapter dirs
    local v="$RL_WORK/adapters_view_${TAG}_$1"; shift
    rm -rf "$v"; mkdir -p "$v"
    local d
    for d in "$@"; do
        [[ -f "$d/adapter_config.json" ]] || { echo "[FATAL] adapter $d not found"; exit 1; }
        ln -sfn "$d" "$v/$(basename "$d")"
    done
    VIEW="$v"
}
# Every name a stage will call must be SERVED, or the row reports another model.
require_served () {
    run python - "$VLLM_URL" "$@" <<'PYS'
import json, sys, urllib.request
url, want = sys.argv[1], sys.argv[2:]
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with op.open(url + "/models", timeout=60) as r:
    served = [m["id"] for m in json.load(r).get("data", [])]
print("[serve] served:", served)
miss = [w for w in want if w not in served]
if miss:
    print(f"[FATAL] NOT served: {miss} -- the stage would score a different model")
    sys.exit(1)
PYS
}
# PEFT initialises lora_B to ZERO: an untrained adapter is the identity and reports the
# base model under a fine-tuned label. This reads the safetensors header directly.
check_adapter () {
    soft python - "$1" <<'PYCHK'
import json, os, struct, sys
import numpy as np
d = sys.argv[1]
st = os.path.join(d, "adapter_model.safetensors")
if not os.path.exists(st):
    print(f"[adapter] {d}: no adapter_model.safetensors"); sys.exit(0)
with open(st, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(n)); base = 8 + n
    worst = 0.0
    for k, v in hdr.items():
        if not isinstance(v, dict) or "lora_B" not in k: continue
        s, e = v["data_offsets"]; f.seek(base + s); raw = f.read(e - s)
        if v["dtype"] == "BF16":
            arr = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif v["dtype"] in ("F32", "F16"):
            arr = np.frombuffer(raw, dtype=np.float32 if v["dtype"] == "F32" else np.float16)
        else:
            continue
        worst = max(worst, float(np.abs(arr).max()))
print(f"[adapter] {os.path.basename(d)}  max|lora_B| {worst:.3e}  "
      + ("TRAINED" if worst > 1e-6 else "UNTRAINED (identity) -- do not evaluate"))
PYCHK
}
# The reviewer settings, IDENTICAL for pairs, attacker reward and evaluation. A row whose
# DPO pairs were reviewed with one prompt and whose test set with another is not SafeReview.
BASE_PROMPT=${BASE_PROMPT:-v7excise}
REVIEWER_PROMPT_FILE=${REVIEWER_PROMPT_FILE:-}
SYSDEF_FILE=${SYSDEF_FILE:-}
REVIEW_MAX_CHARS=${REVIEW_MAX_CHARS:-24000}
N_REVIEWS=${N_REVIEWS:-1}
REVIEW_TEMP=${REVIEW_TEMP:-0.0}
WORKERS=${WORKERS:-16}
REVFLAGS=(--base-prompt "$BASE_PROMPT" --review-max-chars "$REVIEW_MAX_CHARS"
          --n-reviews "$N_REVIEWS" --review-temperature "$REVIEW_TEMP" --workers "$WORKERS")
[[ -n "$REVIEWER_PROMPT_FILE" ]] && REVFLAGS+=(--prompt-file "$REVIEWER_PROMPT_FILE")
[[ -n "$SYSDEF_FILE" ]] && REVFLAGS+=(--sysdef-file "$SYSDEF_FILE")
echo "[reviewer] ${REVFLAGS[*]}"
'''

ATTACKER_BLOCK = r'''
# ---- the attacker ----------------------------------------------------------------------
# SafeReview's main attacker is Qwen3-4B-Instruct; qwen3-8b (its Table-2 scale variant) is
# on disk and is the default. ATTACKER=qwen3-4b-instruct after downloading it.
ATTACKER=${ATTACKER:-qwen3-8b}
ATK_DIR=$(python -c "import sr_config as S; print(S.ATTACKERS['$ATTACKER'].path)") || {
    echo "[FATAL] unknown ATTACKER=$ATTACKER"; exit 1; }
ATK_QUANT=${ATK_QUANT:-none}
[[ -d "$ATK_DIR" ]] || { echo "[FATAL] attacker weights not found: $ATK_DIR"; exit 1; }
atk_name () { python -c "import sr_config as S; print(S.attacker_name('$ATTACKER', '$MODEL', $1))"; }
echo "[attacker] $ATTACKER  $ATK_DIR"
'''

# ======================================================================================
I_JOB = r'''
# =====================================================================================
# STAGE I — INPUTS. CPU, minutes. Run once (and again whenever the v8 data changes).
#
#     qsub jobs/job_sr_i_inputs.pbs
#     qsub -v MAX_ROWS=5,RUN_TAG=pilot5 jobs/job_sr_i_inputs.pbs    # a pilot test set
#
# COPIES $V7AB_DIR/inputs/<model>/eval_records[__TAG].jsonl + eval_index[__TAG].csv and
# inputs/train_documents.csv here (sha256 checked), or BUILDS them with v7ab_inputs.py
# when v7ab has not built them yet. Same documents as every other row, byte for byte.
# =====================================================================================
MODELS_LIST="${MODELS_LIST:-gemma3-27b-it qwen3-8b mistral-small-3.1-24b gpt-4o-mini}"
MAX_ROWS=${MAX_ROWS:-0}
SOURCES="${SOURCES:-}"
RUN_TAG="${RUN_TAG:-}"
TRAIN_CAP=${TRAIN_CAP:-0}
MODE=${MODE:-auto}
TAG=cpu
MODEL=n/a
ABL=inputs

@@LOG@@
@@HELPERS@@
@@CPU_PATH@@

run python -u sr_selftest.py
FL=(--models $(echo "$MODELS_LIST" | tr ',+' '  ') --mode "$MODE" --max-rows "$MAX_ROWS" --train-cap "$TRAIN_CAP")
[[ -n "$SOURCES" ]] && FL+=(--sources $(echo "$SOURCES" | tr ',+' '  '))
[[ -n "$RUN_TAG" ]] && FL+=(--run-tag "$RUN_TAG")
run python -u sr_inputs.py "${FL[@]}"
'''

S_JOB = r'''
# =====================================================================================
# STAGE S — STATIC DPO: one DPO round on your EXISTING training attacks. No attacker.
#
#     qsub jobs/job_sr_s_@@SHORT@@_static.pbs
#     qsub -v REDO=dpo,DPO_LR=2e-5 jobs/job_sr_s_@@SHORT@@_static.pbs   # retrain only
#
#   [S1] serve the BASE reviewer; review the clean twin and the attacked document of up
#        to MAX_PAIRS training pairs (stratified over insertion / ARGAR / BTR)
#        -> static/<model>/pairs.jsonl   prompt = attacked paper, y+ = clean review,
#                                        y- = attacked review
#   [S2] DPO (beta 0.1, batch 8, 30 steps) -> adapters/<model>-sr_static_dpo
#
# CONTROLS
#   MAX_PAIRS=512  TRAIN_PAPERS=all|disjoint  PAIR_FILTER=all|changed|raised
#   DPO_LR=1e-5  DPO_STEPS=30  DPO_BATCH=8  DPO_BETA=0.1  MAX_SEQ_LEN=8192
#   FORCE=1        redo every step;  REDO=pairs|dpo  redo that step and the ones after it
# =====================================================================================
ABL=SR_static_dpo
MAX_PAIRS=${MAX_PAIRS:-512}
TRAIN_PAPERS=${TRAIN_PAPERS:-all}
PAIR_FILTER=${PAIR_FILTER:-all}
DPO_LR=${DPO_LR:-1e-5}
DPO_STEPS=${DPO_STEPS:-30}
DPO_BATCH=${DPO_BATCH:-8}
DPO_BETA=${DPO_BETA:-0.1}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-8192}
FORCE=${FORCE:-0}
REDO=${REDO:-}
SEED=${SEED:-0}
RUNTAG="$TAG"

@@LOG@@
@@HELPERS@@
@@GPU_CHECK@@
@@LOCATOR@@
@@SR_HELPERS@@
@@NEED@@

run python -u sr_selftest.py
have "$RL_WORK/inputs/train_documents.csv" || { echo "[FATAL] run qsub jobs/job_sr_i_inputs.pbs first"; exit 1; }
OUT="$RL_WORK/static/$MK"
PAIRS="$OUT/pairs.jsonl"
DEF_DIR="$RL_WORK/adapters/$MK-sr_static_dpo"
mkdir -p "$OUT"

stage "[S1] preference pairs from the existing attack corpus (base reviewer)"
if need pairs "$PAIRS"; then
    start_server "$MODEL_DIR" "$QUANT" "" "s_pairs"
    require_served "$MODEL_DIR"
    run python -u sr_build_pairs.py --model "$MODEL" --source corpus \
        --served "$MODEL_DIR" --vllm-url "$VLLM_URL" --out "$PAIRS" \
        --max-pairs "$MAX_PAIRS" --train-papers "$TRAIN_PAPERS" \
        --pair-filter "$PAIR_FILTER" --seed "$SEED" "${REVFLAGS[@]}"
    stop_server
fi

stage "[S2] DPO -> $DEF_DIR"
if need dpo "$DEF_DIR/dpo_summary.json"; then
    run python -u sr_train_dpo.py --model "$MODEL" --pairs "$PAIRS" --stage sr_static_dpo \
        --out-dir "$DEF_DIR" --lr "$DPO_LR" --steps "$DPO_STEPS" --batch "$DPO_BATCH" \
        --beta "$DPO_BETA" --max-seq-len "$MAX_SEQ_LEN" --seed "$SEED"
fi
check_adapter "$DEF_DIR"
echo
echo "==================================================================="
echo "STATIC DPO DONE. Evaluate it:"
echo "    qsub -v ROWS=SR_static_dpo jobs/job_sr_c_@@SHORT@@_eval.pbs"
echo "==================================================================="
'''

NEED = r'''
# need STEP FILE: run the step unless FILE exists. FORCE=1 redoes everything; REDO=a,b
# redoes those steps; once any step runs, every LATER step runs too (its input changed).
CASCADE=0
need () {
    local step="$1" f="$2"
    if [[ "$FORCE" == "1" || "$CASCADE" == "1" ]]; then CASCADE=1; return 0; fi
    case ",$REDO," in *",$step,"*) CASCADE=1; return 0 ;; esac
    if have "$f"; then echo "[skip] $step: $f exists (REDO=$step to redo it)"; return 1; fi
    CASCADE=1
    return 0
}
'''

F_JOB = r'''
# =====================================================================================
# STAGE F — ONE SafeReview CO-EVOLUTION ROUND t (Algorithm 1). One round per submission:
# a 3-round loop in one job is a multi-day allocation that loses everything if round 3
# crashes. Every step skips when its output exists, so a resubmission resumes.
#
#     qsub -v ROUND=1 jobs/job_sr_f_@@SHORT@@_round.pbs
#     qsub -v ROUND=2 jobs/job_sr_f_@@SHORT@@_round.pbs       # needs round 1's adapters
#
#   Phase 1 — attack evolution, ATK_ITERS offline cycles of:
#     [F1] rollout   serve attacker A_{t-1}: G=8 injections for each of ATK_PAPERS papers
#     [F2] score     serve reviewer R_{t-1}: r = s_adv - s_orig (Eq. 1), group advantages
#     [F3] update    GRPO (beta_KL 0.02, ref = A_{t-1}) -> adapters_attacker/<..>-r<t>
#   Phase 2 — defense strengthening
#     [F4] regen     serve A_t: a fresh injection for each of DPO_PAPERS training papers
#     [F5] pairs     serve R_{t-1}: y+ = clean review, y- = attacked review
#     [F6] dpo       DPO (beta 0.1, batch 8, 30 steps, ref = R_{t-1}) -> adapters/<m>-sr_r<t>
#
# R_0 is the base reviewer and A_0 the base attacker. Round 1's defender is the paper's
# "Static DPO" baseline; round T's is SafeReview.
#
# CONTROLS
#   ROUND=1   ATK_ITERS=1   ATK_PAPERS=64   G=8   ATK_TEMP=1.0   ATK_LR=5e-6
#   ATK_BETA=0.02   ATK_ACCUM=4   RANK_W=0 (Eq. 1; >0 adds batch ranking displacement)
#   DPO_PAPERS=320   DPO_LR=1e-5   DPO_STEPS=30   DPO_BATCH=8   DPO_BETA=0.1
#   PAIR_FILTER=all   KEEP_HISTORY=0 (Alg. 1 resets D_pref each round)
#   TRAIN_PAPERS=all|disjoint   ATTACKER=qwen3-8b|qwen3-4b-instruct
#   FORCE=1 | REDO=rollout,score,attacker,regen,pairs,dpo
# =====================================================================================
ROUND=${ROUND:-1}
ABL=SR_safereview_r$ROUND
ATK_ITERS=${ATK_ITERS:-1}
ATK_PAPERS=${ATK_PAPERS:-64}
G=${G:-8}
ATK_TEMP=${ATK_TEMP:-1.0}
ATK_LR=${ATK_LR:-5e-6}
ATK_BETA=${ATK_BETA:-0.02}
ATK_ACCUM=${ATK_ACCUM:-4}
ATK_MAX_NEW=${ATK_MAX_NEW:-384}
RANK_W=${RANK_W:-0.0}
DPO_PAPERS=${DPO_PAPERS:-320}
DPO_LR=${DPO_LR:-1e-5}
DPO_STEPS=${DPO_STEPS:-30}
DPO_BATCH=${DPO_BATCH:-8}
DPO_BETA=${DPO_BETA:-0.1}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-8192}
PAIR_FILTER=${PAIR_FILTER:-all}
KEEP_HISTORY=${KEEP_HISTORY:-0}
TRAIN_PAPERS=${TRAIN_PAPERS:-all}
FORCE=${FORCE:-0}
REDO=${REDO:-}
SEED=${SEED:-0}
RUNTAG="$TAG"

@@LOG@@
@@HELPERS@@
@@GPU_CHECK@@
@@LOCATOR@@
@@SR_HELPERS@@
@@ATTACKER@@
@@NEED@@

run python -u sr_selftest.py
have "$RL_WORK/inputs/train_documents.csv" || { echo "[FATAL] run qsub jobs/job_sr_i_inputs.pbs first"; exit 1; }
PREV=$((ROUND-1))
RDIR="$RL_WORK/coevo/$MK/round$ROUND"
mkdir -p "$RDIR"
DEF_T_DIR="$RL_WORK/adapters/$MK-sr_r$ROUND"
ATK_T_DIR="$RL_WORK/adapters_attacker/$(atk_name "$ROUND")"
if [[ "$PREV" -eq 0 ]]; then
    DEF_PREV_DIR=""; DEF_PREV_SERVED="$MODEL_DIR"; ATK_PREV_DIR=""
else
    DEF_PREV_DIR="$RL_WORK/adapters/$MK-sr_r$PREV"
    DEF_PREV_SERVED="$MK-sr_r$PREV"
    ATK_PREV_DIR="$RL_WORK/adapters_attacker/$(atk_name "$PREV")"
    for d in "$DEF_PREV_DIR" "$ATK_PREV_DIR"; do
        have "$d/adapter_config.json" || { echo "[FATAL] $d missing: run ROUND=$PREV first"; exit 1; }
    done
fi
echo "[round] t=$ROUND  defender R_{t-1}=${DEF_PREV_DIR:-base reviewer}  attacker A_{t-1}=${ATK_PREV_DIR:-base attacker}"

serve_reviewer () {   # R_{t-1}
    if [[ -z "$DEF_PREV_DIR" ]]; then start_server "$MODEL_DIR" "$QUANT" "" "f_rev"
    else adapter_view rev "$DEF_PREV_DIR"; start_server "$MODEL_DIR" "$QUANT" "$VIEW" "f_rev"; fi
    require_served "$DEF_PREV_SERVED"
}
serve_attacker () {   # $1 = adapter dir or "" (base attacker); sets POL_SERVED
    if [[ -z "$1" ]]; then start_server "$ATK_DIR" "$ATK_QUANT" "" "f_atk"; POL_SERVED="$ATK_DIR"
    else adapter_view atk "$1"; start_server "$ATK_DIR" "$ATK_QUANT" "$VIEW" "f_atk"
         POL_SERVED="$(basename "$1")"; fi
    require_served "$POL_SERVED"
}
POL_SERVED=""

for K in $(seq 1 "$ATK_ITERS"); do
    IDIR="$RDIR/atk_iter$K"; mkdir -p "$IDIR"
    if [[ "$K" -eq 1 ]]; then POL_DIR="$ATK_PREV_DIR"; else POL_DIR="$ATK_T_DIR"; fi

    stage "[F1.$K] attacker rollout (policy ${POL_DIR:-base attacker})"
    if need rollout "$IDIR/rollouts.jsonl"; then
        serve_attacker "$POL_DIR"
        run python -u sr_attacker.py rollout --model "$MODEL" --attacker "$ATTACKER" \
            --served "$POL_SERVED" --vllm-url "$VLLM_URL" --round "$ROUND" --iter "$K" \
            --n-papers "$ATK_PAPERS" --group-size "$G" --temperature "$ATK_TEMP" \
            --max-new-tokens "$ATK_MAX_NEW" --train-papers "$TRAIN_PAPERS" \
            --seed "$SEED" --workers "$WORKERS" --out "$IDIR/rollouts.jsonl"
        stop_server
    fi

    stage "[F2.$K] reward: score under R_{t-1}"
    if need score "$IDIR/scored.jsonl"; then
        serve_reviewer
        run python -u sr_attacker.py score --model "$MODEL" --attacker "$ATTACKER" \
            --served "$DEF_PREV_SERVED" --adapter-dir "$DEF_PREV_DIR" \
            --vllm-url "$VLLM_URL" --round "$ROUND" --iter "$K" \
            --rollouts "$IDIR/rollouts.jsonl" --rank-weight "$RANK_W" --seed "$SEED" \
            --out "$IDIR/scored.jsonl" "${REVFLAGS[@]}"
        stop_server
    fi

    stage "[F3.$K] attacker GRPO update -> $ATK_T_DIR"
    if need attacker "$IDIR/attacker_done.json"; then
        run python -u sr_train_attacker.py --model "$MODEL" --attacker "$ATTACKER" \
            --scored "$IDIR/scored.jsonl" --init-adapter "$POL_DIR" --out-dir "$ATK_T_DIR" \
            --lr "$ATK_LR" --beta-kl "$ATK_BETA" --grad-accum "$ATK_ACCUM" --seed "$SEED"
        run cp "$ATK_T_DIR/grpo_summary.json" "$IDIR/attacker_done.json"
        check_adapter "$ATK_T_DIR"
    fi
done

stage "[F4] Phase 2: regenerate injections with A_t over $DPO_PAPERS training papers"
if need regen "$RDIR/regen.jsonl"; then
    serve_attacker "$ATK_T_DIR"
    run python -u sr_attacker.py regen --model "$MODEL" --attacker "$ATTACKER" \
        --served "$POL_SERVED" --vllm-url "$VLLM_URL" --round "$ROUND" \
        --n-papers "$DPO_PAPERS" --temperature "$ATK_TEMP" --max-new-tokens "$ATK_MAX_NEW" \
        --review-max-chars "$REVIEW_MAX_CHARS" --train-papers "$TRAIN_PAPERS" \
        --seed "$SEED" --workers "$WORKERS" --out "$RDIR/regen.jsonl"
    stop_server
fi

stage "[F5] preference pairs under R_{t-1}"
if need pairs "$RDIR/pairs.jsonl"; then
    serve_reviewer
    run python -u sr_build_pairs.py --model "$MODEL" --source injections \
        --injections "$RDIR/regen.jsonl" --served "$DEF_PREV_SERVED" \
        --adapter-dir "$DEF_PREV_DIR" --vllm-url "$VLLM_URL" --out "$RDIR/pairs.jsonl" \
        --pair-filter "$PAIR_FILTER" --seed "$SEED" "${REVFLAGS[@]}"
    stop_server
fi

stage "[F6] defender DPO: R_t from R_{t-1} -> $DEF_T_DIR"
PAIRS_LIST=("$RDIR/pairs.jsonl")
if [[ "$KEEP_HISTORY" == "1" ]]; then
    for p in $(seq 1 "$PREV"); do
        f="$RL_WORK/coevo/$MK/round$p/pairs.jsonl"; have "$f" && PAIRS_LIST+=("$f")
    done
fi
if need dpo "$DEF_T_DIR/dpo_summary.json"; then
    run python -u sr_train_dpo.py --model "$MODEL" --pairs "${PAIRS_LIST[@]}" \
        --stage "sr_r$ROUND" --init-adapter "$DEF_PREV_DIR" --out-dir "$DEF_T_DIR" \
        --lr "$DPO_LR" --steps "$DPO_STEPS" --batch "$DPO_BATCH" --beta "$DPO_BETA" \
        --max-seq-len "$MAX_SEQ_LEN" --seed "$SEED"
fi
check_adapter "$DEF_T_DIR"
soft python - "$RDIR" <<'PYS'
import glob, json, os, sys
d = sys.argv[1]
for p in sorted(glob.glob(os.path.join(d, "atk_iter*", "stats.json"))):
    s = json.load(open(p))
    print(f"[round] {os.path.basename(os.path.dirname(p))}: valid {s['valid_frac']:.2f}  "
          f"mean dRating {s['mean_delta_valid']}  ASR {s['asr_valid']}  "
          f"flat groups {s['no_signal_group_frac']:.2f}")
p = os.path.join(d, "pairs_stats.json")
if os.path.exists(p):
    s = json.load(open(p))
    print(f"[round] D_pref: {s['n_pairs']} pairs, s_orig {s['mean_s_orig']}, s_adv "
          f"{s['mean_s_adv']}, raised {s['frac_raised']:.2f}")
PYS
echo
echo "==================================================================="
echo "ROUND $ROUND DONE.  defender $DEF_T_DIR"
echo "                    attacker $ATK_T_DIR"
echo "Next round:     qsub -v ROUND=$((ROUND+1)) jobs/job_sr_f_@@SHORT@@_round.pbs"
echo "Evaluate:       qsub -v ROWS=SR_safereview jobs/job_sr_c_@@SHORT@@_eval.pbs"
echo "==================================================================="
'''

D_JOB = r'''
# =====================================================================================
# STAGE D — THE ADAPTIVE TEST SET: the round-ROUND attacker A_t attacks every test paper's
# clean document once. SafeReview's own threat model (its Table 2: a GRPO attacker
# transferred to the test set), written in the same records/index format as the main set.
#
#     qsub -v ROUND=3 jobs/job_sr_d_@@SHORT@@_adaptive.pbs
#     then: qsub -v SET=adaptive_r3 jobs/job_sr_c_@@SHORT@@_eval.pbs
#           qsub -v SET=adaptive_r3 jobs/job_sr_p_promptguard.pbs
#
# CONTROLS  ROUND=3  TEST_SOURCES=insertion  RUN_TAG=  MAX_ROWS=0  ATK_TEMP=1.0
# =====================================================================================
ROUND=${ROUND:-3}
SET_NAME="adaptive_r$ROUND"
ABL=$SET_NAME
TEST_SOURCES="${TEST_SOURCES:-insertion}"
RUN_TAG="${RUN_TAG:-}"
MAX_ROWS=${MAX_ROWS:-0}
ATK_TEMP=${ATK_TEMP:-1.0}
ATK_MAX_NEW=${ATK_MAX_NEW:-384}
SEED=${SEED:-0}
RUNTAG="$TAG${RUN_TAG:+_$RUN_TAG}"

@@LOG@@
@@HELPERS@@
@@GPU_CHECK@@
@@LOCATOR@@
@@SR_HELPERS@@
@@ATTACKER@@

run python -u sr_selftest.py
ATK_T_DIR="$RL_WORK/adapters_attacker/$(atk_name "$ROUND")"
have "$ATK_T_DIR/adapter_config.json" || { echo "[FATAL] $ATK_T_DIR missing: run the co-evolution rounds first"; exit 1; }
adapter_view atk "$ATK_T_DIR"
start_server "$ATK_DIR" "$ATK_QUANT" "$VIEW" "d_adapt"
require_served "$(basename "$ATK_T_DIR")"
run python -u sr_attacker.py regen --test --model "$MODEL" --attacker "$ATTACKER" \
    --served "$(basename "$ATK_T_DIR")" --vllm-url "$VLLM_URL" --round "$ROUND" \
    --set-name "$SET_NAME" --test-sources $(echo "$TEST_SOURCES" | tr ',+' '  ') --temperature "$ATK_TEMP" \
    --max-new-tokens "$ATK_MAX_NEW" --review-max-chars "$REVIEW_MAX_CHARS" \
    --max-rows "$MAX_ROWS" --seed "$SEED" --workers "$WORKERS" ${RUN_TAG:+--run-tag "$RUN_TAG"}
stop_server
echo
echo "==================================================================="
echo "ADAPTIVE TEST SET: $RL_WORK/inputs/$MODEL/adaptive_records__${SET_NAME}*.jsonl"
echo "Score every row on it:"
echo "    qsub -v SET=$SET_NAME${RUN_TAG:+,RUN_TAG=$RUN_TAG} jobs/job_sr_c_@@SHORT@@_eval.pbs"
echo "==================================================================="
'''

C_JOB = r'''
# =====================================================================================
# STAGE C — EVALUATE rows on the test set: every document (clean AND attacked) reviewed
# by the row's reviewer -> runs/<model>/<ROW>[__set-S][__tag-T]/review_detail.csv
#
#     qsub jobs/job_sr_c_@@SHORT@@_eval.pbs                       # all rows available
#     qsub -v ROWS=SR_undefended jobs/job_sr_c_@@SHORT@@_eval.pbs  # R_0 / R_A only
#     qsub -v ROWS=SR_undefended+SR_sysdef jobs/job_sr_c_@@SHORT@@_eval.pbs
#     qsub -v ROWS=SR_safereview,EVAL_ROUNDS=last jobs/job_sr_c_@@SHORT@@_eval.pbs
#     qsub -v SET=adaptive_r3 jobs/job_sr_c_@@SHORT@@_eval.pbs      # adaptive test set
#     qsub -v RUN_TAG=pilot5 jobs/job_sr_c_@@SHORT@@_eval.pbs       # pilot test set
#
# One vLLM server: the base reviewer plus every adapter a requested row needs (vLLM's
# LoRA selection IS the request's model field). SR_undefended must be scored on a set
# before the table can say anything: its ratings are R_0 and R_A for every row.
#
# CONTROLS
#   ROWS=SR_undefended+SR_sysdef+SR_static_dpo+SR_safereview   ('+' or space separated:
#        qsub -v uses commas to separate VARIABLES)
#   EVAL_ROUNDS=all|last|1+3   which SafeReview rounds
#   SET=main|adaptive_r<t>   RUN_TAG=   MAX_LORAS=5
# =====================================================================================
ROWS="${ROWS:-SR_undefended SR_sysdef SR_static_dpo SR_safereview}"
ROWS="$(echo "$ROWS" | tr ',+' '  ')"
EVAL_ROUNDS="${EVAL_ROUNDS:-all}"
SET="${SET:-main}"
RUN_TAG="${RUN_TAG:-}"
MAX_LORAS=${MAX_LORAS:-5}
ABL=eval_${SET}
RUNTAG="$TAG${RUN_TAG:+_$RUN_TAG}"

@@LOG@@
@@HELPERS@@
@@GPU_CHECK@@
@@LOCATOR@@
@@SR_HELPERS@@

run python -u sr_selftest.py
E_ROW=(); E_ROUND=(); E_SERVED=(); E_DIR=(); ADAPTERS=()
for R in $ROWS; do
    case "$R" in
        SR_undefended|SR_sysdef)
            E_ROW+=("$R"); E_ROUND+=(""); E_SERVED+=("$MODEL_DIR"); E_DIR+=("") ;;
        SR_static_dpo)
            d="$RL_WORK/adapters/$MK-sr_static_dpo"
            if have "$d/adapter_config.json"; then
                E_ROW+=("$R"); E_ROUND+=(""); E_SERVED+=("$(basename "$d")"); E_DIR+=("$d"); ADAPTERS+=("$d")
            else echo "[skip] $R: $d not trained (jobs/job_sr_s_@@SHORT@@_static.pbs)"; fi ;;
        SR_safereview)
            ALL_T=$(ls -d "$RL_WORK/adapters/$MK-sr_r"* 2>/dev/null | sed 's/.*-sr_r//' | sort -n | tr '\n' ' ')
            case "$EVAL_ROUNDS" in
                all)  WANT="$ALL_T" ;;
                last) WANT="$(echo $ALL_T | awk '{print $NF}')" ;;
                *)    WANT="$(echo "$EVAL_ROUNDS" | tr ',+' '  ')" ;;
            esac
            [[ -z "${WANT// /}" ]] && echo "[skip] SR_safereview: no round adapters yet"
            for t in $WANT; do
                d="$RL_WORK/adapters/$MK-sr_r$t"
                if have "$d/adapter_config.json"; then
                    E_ROW+=("$R"); E_ROUND+=("$t"); E_SERVED+=("$(basename "$d")"); E_DIR+=("$d"); ADAPTERS+=("$d")
                else echo "[skip] SR_safereview round $t: $d missing"; fi
            done ;;
        SR_promptguard2) echo "[note] PromptGuard runs in jobs/job_sr_p_promptguard.pbs (no vLLM)" ;;
        *) echo "[FATAL] unknown row $R"; exit 1 ;;
    esac
done
[[ "${#E_ROW[@]}" -gt 0 ]] || { echo "[FATAL] nothing to evaluate"; exit 1; }
if [[ "${#ADAPTERS[@]}" -gt 0 ]]; then
    adapter_view eval "${ADAPTERS[@]}"
    export VLLM_MAX_LORAS="$MAX_LORAS"
    start_server "$MODEL_DIR" "$QUANT" "$VIEW" "c_eval"
else
    start_server "$MODEL_DIR" "$QUANT" "" "c_eval"
fi
require_served "${E_SERVED[@]}"
for i in "${!E_ROW[@]}"; do
    stage "[C] ${E_ROW[$i]}${E_ROUND[$i]:+ round ${E_ROUND[$i]}}  ($SET${RUN_TAG:+ / $RUN_TAG})"
    run python -u sr_eval.py --model "$MODEL" --row "${E_ROW[$i]}" \
        ${E_ROUND[$i]:+--round "${E_ROUND[$i]}"} --served "${E_SERVED[$i]}" \
        --adapter-dir "${E_DIR[$i]}" --vllm-url "$VLLM_URL" --set "$SET" \
        ${RUN_TAG:+--run-tag "$RUN_TAG"} "${REVFLAGS[@]}"
done
stop_server
soft python -u sr_report.py --model "$MODEL" --set "$SET" ${RUN_TAG:+--run-tag "$RUN_TAG"}
echo
echo "==================================================================="
echo "DONE. runs/$MK/*/review_detail.csv"
echo "Table: $RL_WORK/final/sr_${MK}$([[ "$SET" != main ]] && echo "__$SET")${RUN_TAG:+__$RUN_TAG}/final_table.tex"
echo "==================================================================="
'''

P_JOB = r'''
# =====================================================================================
# STAGE P — PromptGuard 2 (86M) on every model's test set (SafeReview App. A.1). No vLLM.
#
#     qsub jobs/job_sr_p_promptguard.pbs
#     qsub -v SET=adaptive_r3,MODELS_LIST=gemma3-27b-it jobs/job_sr_p_promptguard.pbs
#
# The model is gated and the node is offline: download it once (login node) to $PG_MODEL:
#     huggingface-cli download meta-llama/Llama-Prompt-Guard-2-86M --local-dir $PG_MODEL
# =====================================================================================
MODELS_LIST="${MODELS_LIST:-gemma3-27b-it qwen3-8b mistral-small-3.1-24b gpt-4o-mini}"
SET="${SET:-main}"
RUN_TAG="${RUN_TAG:-}"
PG_MODEL="${PG_MODEL:-path/to/models/llama_prompt_guard_2_86m}"
PG_THRESHOLD=${PG_THRESHOLD:-0.5}
TAG=pg
MODEL=n/a
ABL=promptguard2_$SET
TP=1

@@LOG@@
@@HELPERS@@
@@GPU_CHECK@@
@@CPU_PATH@@

run python -u sr_selftest.py
[[ -d "$PG_MODEL" ]] || { echo "[FATAL] $PG_MODEL not found (see the download line above)"; exit 1; }
for M in $(echo "$MODELS_LIST" | tr ',+' '  '); do
    if [[ "$SET" == "main" ]]; then R="$RL_WORK/inputs/$M/eval_records${RUN_TAG:+__$RUN_TAG}.jsonl"
    else R="$RL_WORK/inputs/$M/adaptive_records__$SET${RUN_TAG:+__$RUN_TAG}.jsonl"; fi
    if ! have "$R"; then echo "[skip] $M: $R not found"; continue; fi
    stage "[P] PromptGuard 2 — $M ($SET)"
    run python -u sr_promptguard.py --model "$M" --set "$SET" --pg-model "$PG_MODEL" \
        --threshold "$PG_THRESHOLD" ${RUN_TAG:+--run-tag "$RUN_TAG"}
done
'''

G_JOB = r'''
# =====================================================================================
# STAGE G — gpt-4o-mini: the rows that need NO weights (undefended, System Defense).
# SafeReview and Static DPO retrain the reviewer and cannot be applied to an API model:
# the table reports them as n/a for gpt-4o-mini, which is the point of the comparison.
#
#     qsub jobs/job_sr_g_gpt4omini.pbs
#     qsub -v SET=main,RUN_TAG=pilot5 jobs/job_sr_g_gpt4omini.pbs
#
# The key comes from path/to/.keys (OPENAI_API_KEY), never from this file. Outside 127.0.0.1
# the client honours http(s)_proxy, so a proxied node works.
# =====================================================================================
MODEL=gpt-4o-mini
TAG=gpt_4o_mini
ROWS="${ROWS:-SR_undefended SR_sysdef}"
SET="${SET:-main}"
RUN_TAG="${RUN_TAG:-}"
OPENAI_URL=${OPENAI_URL:-https://api.openai.com/v1}
ABL=gpt_$SET
RUNTAG="$TAG${RUN_TAG:+_$RUN_TAG}"

@@LOG@@
@@HELPERS@@
@@CPU_PATH@@
@@SR_HELPERS@@

run python -u sr_selftest.py
[[ -n "${OPENAI_API_KEY:-}" ]] || { echo "[FATAL] OPENAI_API_KEY not set: put it in path/to/.keys (chmod 600)"; exit 1; }
for R in $(echo "$ROWS" | tr ',+' '  '); do
    stage "[G] $R — gpt-4o-mini ($SET)"
    run python -u sr_eval.py --model gpt-4o-mini --row "$R" --served gpt-4o-mini \
        --vllm-url "$OPENAI_URL" --api-key-env OPENAI_API_KEY --set "$SET" \
        ${RUN_TAG:+--run-tag "$RUN_TAG"} "${REVFLAGS[@]}"
done
soft python -u sr_report.py --model gpt-4o-mini --set "$SET" ${RUN_TAG:+--run-tag "$RUN_TAG"}
'''

T_JOB = r'''
# =====================================================================================
# STAGE T — THE TABLE, every model, total AND per source. CPU, seconds.
#
#     qsub jobs/job_sr_t_table.pbs
#     qsub -v SETS=main+adaptive_r3,RUN_TAG=pilot5 jobs/job_sr_t_table.pbs
#     qsub -v EXTRA_ROWS="v8 (delete+re-review)|gemma3-27b-it|/path/v8_detail.csv" ...
#
# Human ratings / decisions for SafeReview's Spearman and FPR/FNR: HUMAN_CSV (default the
# primary-500 manuscript CSV), HUMAN_RATING_COL / HUMAN_DECISION_COL (auto-detected when
# empty), GOLD_ACCEPT_THRESHOLD (derive accept from the human rating if no decision).
# =====================================================================================
MODELS_LIST="${MODELS_LIST:-gemma3-27b-it qwen3-8b mistral-small-3.1-24b gpt-4o-mini}"
SETS="${SETS:-main}"
RUN_TAG="${RUN_TAG:-}"
EXTRA_ROWS="${EXTRA_ROWS:-}"
GOLD_ACCEPT_THRESHOLD="${GOLD_ACCEPT_THRESHOLD:-}"
TAG=cpu
MODEL=n/a
ABL=table

@@LOG@@
@@HELPERS@@
@@CPU_PATH@@

XR=()
if [[ -n "$EXTRA_ROWS" ]]; then
    IFS=';' read -r -a _rows <<< "$EXTRA_ROWS"
    for r in "${_rows[@]}"; do XR+=(--extra-row "$r"); done
fi
for S_ in $(echo "$SETS" | tr ',+' '  '); do
    run python -u sr_report.py --model $(echo "$MODELS_LIST" | tr ',+' '  ') --set "$S_" ${RUN_TAG:+--run-tag "$RUN_TAG"} \
        ${GOLD_ACCEPT_THRESHOLD:+--gold-accept-threshold "$GOLD_ACCEPT_THRESHOLD"} "${XR[@]+"${XR[@]}"}"
done
'''


def fill(body: str, *, tag: str, short: str, stage: str, runname: str, gpu: bool = True) -> str:
    return (body.replace("@@LOG@@", M7.log(stage, runname).strip("\n"))
                .replace("@@HELPERS@@", MK.HELPERS.strip("\n"))
                .replace("@@LOCATOR@@", locator().strip("\n"))
                .replace("@@GPU_CHECK@@", MK.GPU_CHECK.strip("\n") if gpu else "")
                .replace("@@CPU_PATH@@", CPU_PATH)
                .replace("@@SR_HELPERS@@", SR_HELPERS.strip("\n"))
                .replace("@@ATTACKER@@", ATTACKER_BLOCK.strip("\n"))
                .replace("@@NEED@@", NEED.strip("\n"))
                .replace("@@TAG@@", tag)
                .replace("@@SHORT@@", short))


def main() -> int:
    os.makedirs(JOBS, exist_ok=True)
    pre = preamble()
    out = []
    out.append(("job_sr_i_inputs",
                MK.header("sr_i", "02:00:00", 0, "48gb", ncpus=4) + pre +
                fill(I_JOB, tag="cpu", short="cpu", gpu=False, stage="i_inputs",
                     runname="i_inputs__${RUN_TAG:-full}__${JOBID}")))
    for short, m in MK.MODELS.items():
        blk = MK.model_block(m)
        srv = SERVER.replace("@@TAG@@", m["tag"])
        out.append((f"job_sr_s_{short}_static",
                    MK.header(f"sr_s_{short}", "24:00:00", m["ngpus"], m["mem"]) + pre + blk + srv +
                    fill(S_JOB, tag=m["tag"], short=short, stage="s_static",
                         runname="s_static__${RUNTAG}__${JOBID}")))
        out.append((f"job_sr_f_{short}_round",
                    MK.header(f"sr_f_{short}", "48:00:00", m["ngpus"], m["mem"]) + pre + blk + srv +
                    fill(F_JOB, tag=m["tag"], short=short, stage="f_round",
                         runname="f_round__${RUNTAG}__r${ROUND}__${JOBID}")))
        out.append((f"job_sr_d_{short}_adaptive",
                    MK.header(f"sr_d_{short}", "12:00:00", m["ngpus"], m["mem"]) + pre + blk + srv +
                    fill(D_JOB, tag=m["tag"], short=short, stage="d_adaptive",
                         runname="d_adaptive__${RUNTAG}__r${ROUND}__${JOBID}")))
        out.append((f"job_sr_c_{short}_eval",
                    MK.header(f"sr_c_{short}", "48:00:00", m["ngpus"], m["mem"]) + pre + blk + srv +
                    fill(C_JOB, tag=m["tag"], short=short, stage="c_eval",
                         runname="c_eval__${RUNTAG}__${SET}__${JOBID}")))
    out.append(("job_sr_p_promptguard",
                MK.header("sr_p_pg", "06:00:00", 1, "32gb", ncpus=4) + pre +
                fill(P_JOB, tag="pg", short="pg", stage="p_promptguard",
                     runname="p_promptguard__${SET}__${JOBID}")))
    out.append(("job_sr_g_gpt4omini",
                MK.header("sr_g_gpt", "72:00:00", 0, "24gb", ncpus=8) + pre +
                fill(G_JOB, tag="gpt_4o_mini", short="gpt", gpu=False, stage="g_gpt",
                     runname="g_gpt__${RUNTAG}__${SET}__${JOBID}")))
    out.append(("job_sr_t_table",
                MK.header("sr_t", "01:00:00", 0, "16gb", ncpus=2) + pre +
                fill(T_JOB, tag="cpu", short="cpu", gpu=False, stage="t_table",
                     runname="t_table__${RUN_TAG:-full}__${JOBID}")))
    for name, txt in out:
        assert "@@" not in txt, (name, [l for l in txt.splitlines() if "@@" in l][:3])
        assert "defense_baselines/safereview" in txt, name
        for ext in (".pbs", ".txt"):
            MK.write(os.path.join(JOBS, name + ext), txt)
    print(f"\n{len(out)} jobs in {JOBS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
