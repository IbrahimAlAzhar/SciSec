#!/usr/bin/env python3
"""
_mkjobs_v7ab.py — the v7 jobs, run on the v8 data (insertion + ARGAR + BTR).

    python _mkjobs_v7ab.py          -> jobs/job_v7ab_*.pbs  (+ .txt copies)

The v7 CODE is unchanged (it is copied into this folder as it is in lim_agents_v7_RL).
What differs is only what the jobs feed it and where they write:

    stage   job                              what                                         GPU
    I       job_v7ab_i_inputs.pbs            v8 data -> inputs/ (V4 + V5 formats)         -
    R       job_v7ab_r_<m>_rollout.pbs       V4's ft_trajectories.py on the NEW train set  yes
    A       job_v7ab_a_<m>_score.pbs         v7_score_rollouts.py (v7 stage A)            -
    B       job_v7ab_b_<m>_train.pbs         train_grpo.py (v7 stage B)                   yes
    E       job_v7ab_e_<m>_sweep.pbs         the lr/beta sweep (v7 stage E)               yes
    C       job_v7ab_c_<m>_infer.pbs         V5 runner_def.py on the NEW test set         yes
                                             rows: A_zeroshot_v5 (base), B_sft, C_sft_dpo
                                             (V4 adapters), and every GRPO row
    G       job_v7ab_g_gpt4omini_zeroshot.pbs  the same runner with gpt-4o-mini            -
    T       job_v7ab_t_table.pbs             v7ab_report.py: total AND per source         -

v7's own stage C imports rows A-D from the published runs. Those were measured on the
insertion corpus only, so on THIS corpus they are re-run (same runner, same flags, same
memory) -- otherwise the table would compare rows scored on different documents.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _mkjobs as MK                                   # noqa: E402  (v7's shared pieces)
import _mkjobs_v7 as M7                                # noqa: E402

JOBS = os.path.join(HERE, "jobs")
NEW_DIR = "$AD_ROOT/defense/lim_agents_v7_RL_with_argar_btr"


def preamble() -> str:
    p = MK.PREAMBLE
    old = 'CODE_DIR="${RL_CODE_DIR:-$AD_ROOT/defense/lim_agents_v7_RL}"'
    assert old in p
    p = p.replace(old, f'CODE_DIR="${{RL_CODE_DIR:-{NEW_DIR}}}"')
    old = "         v7_score_rollouts.py v7_mechint.py v7_table.py; do"
    assert old in p
    p = p.replace(old, "         v7_score_rollouts.py v7_mechint.py v7_table.py \\\n"
                       "         v7ab_inputs.py v7ab_report.py; do")
    p += r'''
# the v8 data files this package reads (built by llim_agents_v8_dvb's data stages)
export V8_DATA_DIR="${V8_DATA_DIR:-$AD_ROOT/data/final_data/v8_defense_data}"
'''
    return p


LINK = r'''
# V4's SFT / DPO adapters, LINKED (not copied) into this package's adapters/: stage B
# starts GRPO from <model>-sft, and stage C evaluates rows B_sft / C_sft_dpo with them.
mkdir -p "$RL_WORK/adapters"
for st in sft dpo; do
    src="$V4_FT_DIR/adapters/$(echo "$MODEL" | tr '/' '_')-$st"
    dst="$RL_WORK/adapters/$(basename "$src")"
    if [[ -d "$src" && ! -e "$dst" ]]; then ln -s "$src" "$dst"; echo "[adapters] linked $dst -> $src"; fi
done
'''

# ======================================================================================
I_JOB = r'''
# =====================================================================================
# STAGE I — INPUTS. CPU, minutes. Run once (and again whenever the v8 data changes).
#
#     qsub jobs/job_v7ab_i_inputs.pbs
#     qsub -v MAX_ROWS=5,RUN_TAG=pilot5 jobs/job_v7ab_i_inputs.pbs     # a small test set
#
# Reads $V8_DATA_DIR (train_pairs.csv.gz, test_pairs__<tag>.csv.gz,
# prepared_insertion__<tag>/eval_records.jsonl) and writes, in the formats v7 reads:
#     inputs/train_documents.csv                 (V4 format -> stage R rollouts)
#     inputs/<model>/eval_records[__TAG].jsonl   (V5 format -> stage C runner_def)
#     inputs/<model>/eval_index[__TAG].csv       (source of every document -> the table)
# =====================================================================================
MODELS_LIST="${MODELS_LIST:-gemma3-27b-it qwen3-8b mistral-small-3.1-24b gpt-4o-mini}"
MAX_ROWS=${MAX_ROWS:-0}
SOURCES="${SOURCES:-}"
RUN_TAG="${RUN_TAG:-}"
TRAIN_CAP=${TRAIN_CAP:-0}
TAG=cpu
MODEL=n/a
ABL=inputs
RUNTAG=cpu

@@LOG@@
@@HELPERS@@

for f in train_pairs.csv.gz; do
    have "$V8_DATA_DIR/$f" || echo "[warn] $V8_DATA_DIR/$f missing: no TRAIN documents (zero-shot still works)"
done
FL=(--models $MODELS_LIST --max-rows "$MAX_ROWS" --train-cap "$TRAIN_CAP")
[[ -n "$SOURCES" ]] && FL+=(--sources $SOURCES)
[[ -n "$RUN_TAG" ]] && FL+=(--run-tag "$RUN_TAG" --no-train)
run python -u v7ab_inputs.py "${FL[@]}"
'''

R_JOB = r'''
# =====================================================================================
# STAGE R — ROLLOUTS on the NEW training documents, with V4's OWN ft_trajectories.py.
#
#     qsub jobs/job_v7ab_r_@@SHORT@@_rollout.pbs
#
# v7 generates no rollouts: it scores V4's. V4's were sampled on traindata_shared/
# train_documents.csv (insertion only, ~13 papers). This runs the SAME script, with the
# same prompts (ft_roles) and the same sampling settings, on inputs/train_documents.csv
# (secondary_280 + tertiary_200 + the ARGAR/BTR train halves). Stage A then reads it.
#
# CONTROLS
#   N_SAMPLES=4    completions per (document, role) -- v7 needs >= 4 for GRPO groups
#   MAX_DOCS=0     cap the documents (0 = all). inputs/train_documents.csv is ordered so
#                  any cap still mixes sources.
#   POLICY=base    base | sft  (sample from the base model, or from V4's SFT adapter)
# =====================================================================================
N_SAMPLES=${N_SAMPLES:-4}
MAX_DOCS=${MAX_DOCS:-0}
POLICY=${POLICY:-base}
ABL=rollout
RUNTAG="$TAG"
DOCS=${DOCS:-$RL_WORK/inputs/train_documents.csv}
OUT="$RL_WORK/trajectories/$MODEL"

@@LOG@@
@@HELPERS@@
@@GPU_CHECK@@
@@LOCATOR@@
@@LINK@@

run python -u rl_selftest.py
have "$DOCS" || { echo "[FATAL] $DOCS not found: run  qsub jobs/job_v7ab_i_inputs.pbs  first"; exit 1; }
[[ -f "$V4_FT_DIR/ft_trajectories.py" ]] || { echo "[FATAL] $V4_FT_DIR/ft_trajectories.py not found"; exit 1; }
mkdir -p "$OUT"
if [[ "$POLICY" == "sft" ]]; then
    VIEW="$RL_WORK/adapters_view_${TAG}_r"; rm -rf "$VIEW"; mkdir -p "$VIEW"
    ln -sfn "$RL_WORK/adapters/$(echo "$MODEL" | tr '/' '_')-sft" "$VIEW/"
    start_server "$VIEW" "r_rollout"
    PMODEL="$(echo "$MODEL" | tr '/' '_')-sft"
else
    start_server "" "r_rollout"
    PMODEL="$MODEL_DIR"
fi
MAXFLAG=""
[[ "$MAX_DOCS" -gt 0 ]] && MAXFLAG="--max-docs $MAX_DOCS"
# V4's script, run from V4's folder so it imports V4's ft_config / ft_roles / ft_vllm
run python -u "$V4_FT_DIR/ft_trajectories.py" --model "$MODEL" --documents "$DOCS" \
    --out-dir "$OUT" --vllm-url "$VLLM_URL" --vllm-model "$PMODEL" \
    --n-samples "$N_SAMPLES" $MAXFLAG
stop_server
have "$OUT/samples.jsonl" || { echo "[FATAL] no samples.jsonl written"; exit 1; }
echo
echo "==================================================================="
echo "ROLLOUTS: $OUT/samples.jsonl   Next:"
echo "    qsub -v ABL=G_grpo_prov jobs/job_v7ab_a_@@SHORT@@_score.pbs"
echo "==================================================================="
'''

C_JOB = r'''
# =====================================================================================
# STAGE C — EVALUATE with V5's OWN runner_def.py on the NEW test set.
#
#     qsub -v ABL=A_zeroshot_v5 jobs/job_v7ab_c_@@SHORT@@_infer.pbs   # zero-shot agents
#     qsub -v ABL=C_sft_dpo     jobs/job_v7ab_c_@@SHORT@@_infer.pbs   # V4 SFT+DPO adapter
#     qsub -v ABL=G_grpo_prov   jobs/job_v7ab_c_@@SHORT@@_infer.pbs   # a GRPO row (stage B)
#     qsub -v ABL=A_zeroshot_v5,RUN_TAG=pilot5 jobs/job_v7ab_c_@@SHORT@@_infer.pbs
#
# job_01's command with two things changed: --eval-records points at the v8 test set
# (insertion + ARGAR + BTR) and there is no --reference-detail (that file describes the
# insertion-only corpus, so it would correctly refuse this one). Memory, calibration,
# cascade depth, workers, min_conf and seed are job_01's.
#
# CONTROLS
#   ABL=<row>       A_zeroshot_v5 | B_sft | C_sft_dpo | E_... M_ (GRPO rows)
#   ADAPTER=<name>  evaluate a different adapter under this row (out dir gets __<name>)
#   RUN_TAG=<tag>   use inputs/<model>/eval_records__<tag>.jsonl (stage I with RUN_TAG)
#                   and write runs/<model>/<row>__tag-<tag>/
#   MAX_LORAS=5
# =====================================================================================
ABL=${ABL:-A_zeroshot_v5}
ADAPTER=${ADAPTER:-}
RUN_TAG="${RUN_TAG:-}"
MAX_LORAS=${MAX_LORAS:-5}
SEED=${SEED:-0}
RUNTAG="$TAG${RUN_TAG:+_$RUN_TAG}"
V5_RUNNER="$V5_DIR/runner_def.py"
EVAL_RECORDS=${EVAL_RECORDS:-$RL_WORK/inputs/$MODEL/eval_records${RUN_TAG:+__$RUN_TAG}.jsonl}
BENCH=${BENCH:-$AD_ROOT/lim_bb_multimodel_500_data/output_primary500}
BENIGN=${BENIGN:-$AD_ROOT/data/final_data/dataset_primary_500.csv}
ATTACK_SEED=${ATTACK_SEED:-seed2}
ATTACKED_CSV=${ATTACKED_CSV:-$BENCH/$MODEL/$ATTACK_SEED/metrics_detail.csv}
MEMORY_DIR=${MEMORY_DIR:-$AD_ROOT/defense/llm_agents_v4/shared}
CALIBRATE=${CALIBRATE:-1}
STAGE2=${STAGE2:-0}
BTR_SIGNATURES=${BTR_SIGNATURES:-1}

@@LOG@@
@@HELPERS@@
@@GPU_CHECK@@
@@LOCATOR@@
@@LINK@@

run python -u rl_selftest.py
[[ -f "$V5_RUNNER" ]] || { echo "[FATAL] $V5_RUNNER not found"; exit 1; }
for f in "$EVAL_RECORDS" "$MEMORY_DIR/memory.json"; do
    have "$f" || { echo "[FATAL] missing input: $f"
                   echo "        eval records come from: qsub jobs/job_v7ab_i_inputs.pbs"; exit 1; }
done
# runner_def.py must accept every flag below; checked BEFORE the model is served, so a
# renamed flag costs one second, not a 27B server start
RD_HELP=$(python "$V5_RUNNER" --help 2>&1)
for fl in --input-mode --eval-records --memory-dir --id-column --threat --backend \
          --vllm-url --vllm-model --out-dir --max-rounds --workers --min-conf \
          --keep-transcripts --save-every --calibrate --fpr-budget --stage1-only; do
    case "$RD_HELP" in *"$fl"*) ;; *) echo "[FATAL] $V5_RUNNER does not accept $fl"; exit 1 ;; esac
done
echo "[preflight] runner_def.py accepts every flag this job passes"

MK="$(echo "$MODEL" | tr '/' '_')"
case "$ABL" in
    A_zeroshot_v5) ADAPTER_NAME="$MODEL_DIR" ;;          # the base model: no adapter
    B_sft)         ADAPTER_NAME="$MK-sft" ;;
    C_sft_dpo)     ADAPTER_NAME="$MK-dpo" ;;
    D_sft_dpo_agents)
        echo "[FATAL] D_sft_dpo_agents routes 5 adapters per role; runner_def serves ONE"
        echo "        model per run, so it cannot be scored here (it never was by v7 either)."
        exit 1 ;;
    *)  ADAPTER_NAME=$(python - "$MODEL" "$ABL" <<'PYN'
import sys, rl_config as C
ab = C.ABL.get(sys.argv[2])
print("" if (ab is None or ab.stage is None) else C.lora_name(sys.argv[1], ab.stage))
PYN
)
        [[ -n "$ADAPTER_NAME" ]] || { echo "[FATAL] unknown row $ABL"; exit 1; } ;;
esac
ROW_SUFFIX=""
if [[ -n "$ADAPTER" ]]; then ADAPTER_NAME="$ADAPTER"; ROW_SUFFIX="__$ADAPTER"; fi
echo "[abl] $ABL -> $ADAPTER_NAME"

if [[ "$ADAPTER_NAME" == "$MODEL_DIR" ]]; then
    start_server "" "c_infer"
else
    have "$RL_WORK/adapters/$ADAPTER_NAME/adapter_config.json" || {
        echo "[FATAL] adapter $RL_WORK/adapters/$ADAPTER_NAME not found."
        echo "        B_sft / C_sft_dpo are linked from \$V4_FT_DIR/adapters; GRPO rows come from stage B."
        exit 1; }
    VIEW="$RL_WORK/adapters_view_$TAG"
    rm -rf "$VIEW"; mkdir -p "$VIEW"
    ln -sfn "$RL_WORK/adapters/$ADAPTER_NAME" "$VIEW/$ADAPTER_NAME"
    export VLLM_MAX_LORAS="$MAX_LORAS"
    start_server "$VIEW" "c_infer"
fi
run python - "$VLLM_URL" "$ADAPTER_NAME" <<'PYS'
import json, sys, urllib.request
url, want = sys.argv[1], sys.argv[2]
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with op.open(url + "/models", timeout=60) as r:
    served = [m["id"] for m in json.load(r).get("data", [])]
print("[serve] served:", served)
if want not in served:
    print(f"[FATAL] {want!r} is NOT served -- the row would report the wrong model")
    sys.exit(1)
PYS

OUT_DIR="$RL_WORK/runs/$MK/${ABL}${ROW_SUFFIX}${RUN_TAG:+__tag-$RUN_TAG}"
mkdir -p "$OUT_DIR"
FLAGS=""
[[ "$CALIBRATE" == "1" ]]      && FLAGS="$FLAGS --calibrate --fpr-budget 0.02"
[[ "$BTR_SIGNATURES" == "0" ]] && FLAGS="$FLAGS --no-btr-signatures"
[[ "$STAGE2" == "0" ]]         && FLAGS="$FLAGS --stage1-only"
SRCFLAGS=""
have "$BENIGN" && SRCFLAGS="$SRCFLAGS --benign-csv $BENIGN"
have "$ATTACKED_CSV" && SRCFLAGS="$SRCFLAGS --attacked-csv $ATTACKED_CSV"
echo "[input] eval records  $EVAL_RECORDS"
echo "[input] memory        $MEMORY_DIR  (read-only; NO index rebuild)"
run python -u "$V5_RUNNER" --mode defend --input-mode sft-dpo \
    --eval-records "$EVAL_RECORDS" \
    --memory-dir "$MEMORY_DIR" $SRCFLAGS --id-column id --threat insertion \
    --model "$MODEL" --backend vllm \
    --vllm-url "$VLLM_URL" --vllm-model "$ADAPTER_NAME" \
    --out-dir "$OUT_DIR" --max-rounds 3 --workers INSTRUCTION PRESSURE VOICE \
    --min-conf 0.5 --keep-transcripts --seed "$SEED" --save-every 5 $FLAGS
stop_server
have "$OUT_DIR/stage1_detail.csv" || { echo "[FATAL] no stage1_detail.csv written"; exit 1; }
soft python -u v7ab_report.py --model "$MODEL" ${RUN_TAG:+--run-tag "$RUN_TAG"}
echo
echo "==================================================================="
echo "DONE.  $OUT_DIR/stage1_detail.csv"
echo "Table: $RL_WORK/final/v7ab_${MK}${RUN_TAG:+__$RUN_TAG}/final_table.tex"
echo "==================================================================="
'''

G_JOB = r'''
# =====================================================================================
# STAGE G — gpt-4o-mini, ZERO-SHOT, with V5's runner_def.py on the NEW test set.
#
#     qsub jobs/job_v7ab_g_gpt4omini_zeroshot.pbs
#     qsub -v RUN_TAG=pilot5 jobs/job_v7ab_g_gpt4omini_zeroshot.pbs
#
# runner_def's vLLM backend speaks the OpenAI protocol, so it is pointed at the OpenAI
# API: --vllm-url https://api.openai.com/v1 --vllm-model gpt-4o-mini, with the key in
# VLLM_API_KEY for THIS command only (sourced from path/to/.keys, never written here).
# NOTE: that client deliberately ignores http(s)_proxy (trust_env=False). If this node
# can reach the internet only through a proxy, the calls will hang: run it on a node with
# direct access, or tell me and the client can be pointed through the proxy.
# =====================================================================================
MODEL=gpt-4o-mini
TAG=gpt_4o_mini
ABL=A_zeroshot_v5
RUN_TAG="${RUN_TAG:-}"
SEED=${SEED:-0}
RUNTAG="$TAG${RUN_TAG:+_$RUN_TAG}"
V5_RUNNER="$V5_DIR/runner_def.py"
EVAL_RECORDS=${EVAL_RECORDS:-$RL_WORK/inputs/$MODEL/eval_records${RUN_TAG:+__$RUN_TAG}.jsonl}
MEMORY_DIR=${MEMORY_DIR:-$AD_ROOT/defense/llm_agents_v4/shared}
OPENAI_URL=${OPENAI_URL:-https://api.openai.com/v1}
CALIBRATE=${CALIBRATE:-1}
BTR_SIGNATURES=${BTR_SIGNATURES:-1}

@@LOG@@
@@HELPERS@@
@@LOCATOR@@

run python -u rl_selftest.py
[[ -n "${OPENAI_API_KEY:-}" ]] || { echo "[FATAL] OPENAI_API_KEY not set: put it in path/to/.keys (chmod 600)"; exit 1; }
for f in "$EVAL_RECORDS" "$MEMORY_DIR/memory.json"; do
    have "$f" || { echo "[FATAL] missing input: $f (run qsub jobs/job_v7ab_i_inputs.pbs)"; exit 1; }
done
# runner_def.py must accept every flag below; checked BEFORE the model is served, so a
# renamed flag costs one second, not a 27B server start
RD_HELP=$(python "$V5_RUNNER" --help 2>&1)
for fl in --input-mode --eval-records --memory-dir --id-column --threat --backend \
          --vllm-url --vllm-model --out-dir --max-rounds --workers --min-conf \
          --keep-transcripts --save-every --calibrate --fpr-budget --stage1-only; do
    case "$RD_HELP" in *"$fl"*) ;; *) echo "[FATAL] $V5_RUNNER does not accept $fl"; exit 1 ;; esac
done
echo "[preflight] runner_def.py accepts every flag this job passes"

OUT_DIR="$RL_WORK/runs/$MODEL/${ABL}${RUN_TAG:+__tag-$RUN_TAG}"
mkdir -p "$OUT_DIR"
FLAGS="--stage1-only"
[[ "$CALIBRATE" == "1" ]]      && FLAGS="$FLAGS --calibrate --fpr-budget 0.02"
[[ "$BTR_SIGNATURES" == "0" ]] && FLAGS="$FLAGS --no-btr-signatures"
run env VLLM_API_KEY="$OPENAI_API_KEY" python -u "$V5_RUNNER" --mode defend --input-mode sft-dpo \
    --eval-records "$EVAL_RECORDS" --memory-dir "$MEMORY_DIR" --id-column id \
    --threat insertion --model "$MODEL" --backend vllm \
    --vllm-url "$OPENAI_URL" --vllm-model gpt-4o-mini \
    --out-dir "$OUT_DIR" --max-rounds 3 --workers INSTRUCTION PRESSURE VOICE \
    --min-conf 0.5 --keep-transcripts --seed "$SEED" --save-every 5 $FLAGS
soft python -u v7ab_report.py --model "$MODEL" ${RUN_TAG:+--run-tag "$RUN_TAG"}
'''

T_JOB = r'''
# =====================================================================================
# STAGE T — THE TABLE, every model, total AND per source. CPU, seconds.
#
#     qsub jobs/job_v7ab_t_table.pbs
#     qsub -v RUN_TAG=pilot5 jobs/job_v7ab_t_table.pbs
# =====================================================================================
MODELS_LIST="${MODELS_LIST:-gemma3-27b-it qwen3-8b mistral-small-3.1-24b gpt-4o-mini}"
RUN_TAG="${RUN_TAG:-}"
TAG=cpu
MODEL=n/a
ABL=table
RUNTAG=cpu

@@LOG@@
@@HELPERS@@
run python -u v7ab_report.py --model $MODELS_LIST ${RUN_TAG:+--run-tag "$RUN_TAG"}
'''


def fill(body: str, *, tag: str, short: str, stage: str, runname: str, gpu: bool = True) -> str:
    return (M7.fill(body, tag=tag, short=short, stage=stage, runname=runname, gpu=gpu)
            .replace("@@LINK@@", LINK.strip("\n")))


def rename_jobs(txt: str) -> str:
    for a, b in (("jobs/job_v7_", "jobs/job_v7ab_"), ("jobs/job_rl_", "jobs/job_v7ab_"),
                 ("_rollout.pbs", "_rollout.pbs")):
        txt = txt.replace(a, b)
    return txt


def main() -> int:
    os.makedirs(JOBS, exist_ok=True)
    pre = preamble()
    out = []
    out.append(("job_v7ab_i_inputs",
                MK.header("v7ab_i", "02:00:00", 0, "48gb", ncpus=4) + pre +
                fill(I_JOB, tag="cpu", short="cpu", gpu=False, stage="i_inputs",
                     runname='i_inputs__${RUN_TAG:-full}__${JOBID}')))
    for short, m in MK.MODELS.items():
        blk = MK.model_block(m)
        srv = MK.SERVER.replace("@@TAG@@", m["tag"])
        out.append((f"job_v7ab_r_{short}_rollout",
                    MK.header(f"v7ab_r_{short}", "48:00:00", m["ngpus"], m["mem"]) + pre + blk + srv +
                    fill(R_JOB, tag=m["tag"], short=short, stage="r_rollout",
                         runname='r_rollout__${RUNTAG}__${JOBID}')))
        a = M7.A
        a = a.replace("SAMPLES=${SAMPLES:-$V4_FT_DIR/trajectories/$MODEL/samples.jsonl}",
                      "SAMPLES=${SAMPLES:-$RL_WORK/trajectories/$MODEL/samples.jsonl}")
        a = a.replace("DOCS=${DOCS:-$V4_FT_DIR/traindata_shared/train_documents.csv}",
                      "DOCS=${DOCS:-$RL_WORK/inputs/train_documents.csv}")
        a = a.replace('''    echo "        v7 does NOT generate rollouts. Run V4's rollout stage first:"
    echo "            cd $V4_FT_DIR && qsub -v MODEL=$MODEL job_ft_a_rollout.pbs"''',
                      '''    echo "        Run the rollout stage on the new training documents first:"
    echo "            qsub jobs/job_v7ab_r_@@SHORT@@_rollout.pbs"''')
        assert "$RL_WORK/trajectories" in a and "job_v7ab_r_" in a
        out.append((f"job_v7ab_a_{short}_score",
                    MK.header(f"v7ab_a_{short}", "06:00:00", 0, "48gb", ncpus=4) + pre + blk +
                    rename_jobs(fill(a, tag=m["tag"], short=short, gpu=False, stage="a_score",
                                     runname='a_score__${RUNTAG}__${ABL}__iter${ITER}__${JOBID}'))))
        b = (MK.fill(MK.B, tag=m["tag"], short=short)
             .replace("jobs/job_rl_a_@@SHORT@@_rollout.pbs", "jobs/job_v7ab_a_@@SHORT@@_score.pbs")
             .replace(f"jobs/job_rl_a_{short}_rollout.pbs", f"jobs/job_v7ab_a_{short}_score.pbs"))
        b = rename_jobs(b)
        out.append((f"job_v7ab_b_{short}_train",
                    MK.header(f"v7ab_b_{short}", "36:00:00", m["ngpus"], m["mem"]) + pre + blk +
                    LINK + b))
        out.append((f"job_v7ab_e_{short}_sweep",
                    MK.header(f"v7ab_e_{short}", "24:00:00", m["ngpus"], m["mem"]) + pre + blk +
                    rename_jobs(fill(M7.SWEEP, tag=m["tag"], short=short, stage="e_sweep",
                                     runname='e_sweep__${RUNTAG}__${ABL}__${JOBID}'))
                    .replace("@@LINK@@", "").replace("ROLL_DIR=", LINK + "\nROLL_DIR=", 1)))
        out.append((f"job_v7ab_c_{short}_infer",
                    MK.header(f"v7ab_c_{short}", "48:00:00", m["ngpus"], m["mem"]) + pre + blk + srv +
                    fill(C_JOB, tag=m["tag"], short=short, stage="c_infer",
                         runname='c_infer__${RUNTAG}__${ABL}__${JOBID}')))
    out.append(("job_v7ab_g_gpt4omini_zeroshot",
                MK.header("v7ab_g_gpt", "72:00:00", 0, "24gb", ncpus=8) + pre +
                fill(G_JOB, tag="gpt_4o_mini", short="gpt", gpu=False, stage="g_gpt",
                     runname='g_gpt__${RUNTAG}__${JOBID}')))
    out.append(("job_v7ab_t_table",
                MK.header("v7ab_t", "01:00:00", 0, "16gb", ncpus=2) + pre +
                fill(T_JOB, tag="cpu", short="cpu", gpu=False, stage="t_table",
                     runname='t_table__${RUN_TAG:-full}__${JOBID}')))
    for name, txt in out:
        assert "@@" not in txt, (name, [l for l in txt.splitlines() if "@@" in l][:3])
        assert "lim_agents_v7_RL_with_argar_btr" in txt, name
        for ext in (".pbs", ".txt"):
            MK.write(os.path.join(JOBS, name + ext), txt)
    print(f"\n{len(out)} jobs in {JOBS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
