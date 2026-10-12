"""
ft_config.py — every path, model, role and threshold this pipeline uses, in one place.

WHY A CONFIG MODULE AND NOT FLAGS EVERYWHERE
=======================================================================================
Six programs have to agree on things that are not obvious: which papers are TRAIN and
which are TEST, what a role's output schema is, which LoRA name a request must carry to
reach a particular adapter. When those live in six argparse defaults they drift, and the
drift is silent — a trajectory generated under one schema and trained under another simply
teaches the model the wrong format, and the first sign is a serving-time parse failure
that looks like a model problem.

THE ONE RULE THIS FILE EXISTS TO ENFORCE
    TRAIN = dataset_secondary_280.csv + dataset_tertiary_200.csv
    TEST  = dataset_primary_500.csv
and they must be DISJOINT BY PAPER ID. `assert_disjoint()` is called by every stage that
touches data. If a single paper appears on both sides, every number downstream is
contaminated and the correct action is to stop, not to warn — so it raises.

YOUR FILES OVERLAP, AND THE FIX IS ON THE TRAIN SIDE
    secondary_280 + tertiary_200 share 67 papers with primary_500. `clean_train_csvs()`
    writes copies of the TRAIN files with those papers removed, into traindata_clean/, and
    every stage reads those copies via `effective_train_csvs()`. The originals are never
    modified and NOTHING is ever removed from TEST — the 500-paper corpus has to stay
    identical to the zero-shot V4 run, or the comparison the whole exercise exists to make
    stops being a comparison. Training data is fungible; the measuring stick is not.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = os.environ.get(
    "AD_ROOT",
    ".")

# The directory you asked for, spelled exactly as you gave it ("llim", two l's).
WORK = os.environ.get("FT_WORK", os.path.join(ROOT, "defense/llim_agents_v4_SFT_DPO"))

V4_DIR = os.environ.get("V4_DIR", os.path.join(ROOT, "defense/llm_agents_v4"))

DATA = os.path.join(ROOT, "data/final_data")
TRAIN_CSVS = [os.path.join(DATA, "dataset_secondary_280.csv"),
              os.path.join(DATA, "dataset_tertiary_200.csv")]
TEST_CSV = os.path.join(DATA, "dataset_primary_500.csv")

# The five columns you named. `id` is the join key; the attack machinery calls it paper_id.
COLUMNS = dict(id="id", text="input_text_cleaned", sections="pdf_text",
               author_lim="author_men_lim", weaknesses="weaknesses")

# Sub-directories under WORK. Created by ensure_dirs().
D_TRAINDATA = "traindata"        # attacked/benign document pairs built from TRAIN
D_TRAINCLEAN = "traindata_clean"  # TRAIN csvs with every TEST paper removed
D_TRAJ = "trajectories"          # raw (role, prompt, completion) samples
D_JUDGED = "judged"              # judge verdicts + the chosen/rejected assignment
D_SFT = "datasets/sft"
D_DPO = "datasets/dpo"
D_ADAPTERS = "adapters"
D_RUNS = "runs"                  # inference ladder output, one dir per ablation
D_LOGS = "logs"


# ======================================================================================
# MODELS
# ======================================================================================
@dataclass
class ModelSpec:
    key: str                     # what the tables call it; matches your v4 --model
    path: str                    # local weights
    quant: str                   # serve_vllm.sh argument: "bnb" or "none"
    load_4bit: bool              # training: 4-bit base under the LoRA
    max_len: int
    # Extra chat_template_kwargs sent with every vLLM request. qwen3 emits a <think>...
    # </think> block by default; it consumes the whole token budget, the JSON never
    # arrives, and every sample scores as a parse failure — 54/54 on the first smoke run.
    chat_kwargs: Optional[Dict] = None
    # Gemma's chat template has NO system role. Merging the system prompt into the first
    # user turn is not cosmetic: apply_chat_template RAISES on a system message for gemma,
    # so a pipeline that assumes three roles trains two models and crashes on the third.
    system_role: bool = True


MODELS: Dict[str, ModelSpec] = {
    "gemma3-27b-it": ModelSpec(
        "gemma3-27b-it",
        "path/to/models/gemma3_27b_it",
        quant="bnb", load_4bit=True, max_len=8192, system_role=False),
    "qwen3-8b": ModelSpec(
        "qwen3-8b",
        "path/to/models/qwen3_8b",
        quant="none", load_4bit=False, max_len=8192, system_role=True,
        chat_kwargs={"enable_thinking": False}),
    "mistral-small-3.1-24b": ModelSpec(
        "mistral-small-3.1-24b",
        "path/to/models/mistral_small_3_1_24b_instruct",
        quant="bnb", load_4bit=True, max_len=8192, system_role=True),
}

JUDGE_MODEL = os.environ.get("FT_JUDGE_MODEL", "gpt-4o-mini")


# ======================================================================================
# ROLES — the agents that get their own adapter
# ======================================================================================
# Each role is a separate supervised task with its own input and its own output schema.
# They are trained separately because they want OPPOSITE behaviour: a worker is rewarded
# for finding evidence, the leader for refusing evidence that does not hold up. Mixed into
# one adapter those gradients fight, and the result is the V2 failure — a model that says
# "attacked" to everything, which scores F1 0.909 at FPR 1.000 and detects nothing.
ROLES: Tuple[str, ...] = ("CHUNK_VERIFIER", "INSTRUCTION", "PRESSURE", "VOICE", "LEADER")

# Roles whose decision is binary attacked/clean (the judge scores label correctness on
# these against gold). VOICE and PRESSURE emit evidence, not verdicts.
DECIDING_ROLES = ("CHUNK_VERIFIER", "LEADER")


def lora_name(model_key: str, stage: str, role: Optional[str] = None) -> str:
    """
    The name a request must send as `model` to reach one adapter on a multi-LoRA vLLM
    server. Derived, never typed: a served-adapter name that disagrees with the directory
    name by one character produces a 404 that reads like a model error.
    """
    base = f"{model_key.replace('/', '_')}-{stage}"
    return f"{base}-{role.lower()}" if role else base


def adapter_dir(model_key: str, stage: str, role: Optional[str] = None) -> str:
    return os.path.join(WORK, D_ADAPTERS, lora_name(model_key, stage, role))


# ======================================================================================
# ABLATIONS — the rows of the final table
# ======================================================================================
@dataclass
class Ablation:
    key: str
    label: str
    stage: Optional[str]         # None = no adapter (zero-shot v4)
    per_agent: bool = False
    note: str = ""


ABLATIONS: Tuple[Ablation, ...] = (
    Ablation("zeroshot", "V4 zero-shot (baseline)", None, note=(
        "Your published V4 numbers, RE-RUN here rather than copied, so the comparison "
        "is against the same 500 papers, the same memory and the same seed. A baseline "
        "quoted from an earlier run cannot distinguish a real gain from a changed "
        "corpus.")),
    Ablation("dpo_only", "DPO only, no SFT", "dpo_only", note=(
        "The control that prices SFT. If this matches sft_dpo, the supervised stage "
        "bought nothing and the pipeline can be halved.")),
    Ablation("sft", "SFT, single adapter", "sft", note=(
        "One adapter per base model, trained on every role's accepted trajectories. "
        "Measures what format-and-reasoning imitation alone is worth.")),
    Ablation("sft_dpo", "SFT then DPO, single adapter", "dpo", note=(
        "The same single adapter, then preference-optimised on same-prompt pairs. "
        "This is the row that tests whether preference data fixes the FPR 0.400.")),
    Ablation("sft_dpo_per_agent", "SFT+DPO per agent (5 adapters)", "dpo",
             per_agent=True, note=(
        "One adapter per role, served together from one vLLM process and selected per "
        "request. The deployed configuration if it wins.")),
)


# ======================================================================================
# generation / training defaults
# ======================================================================================
@dataclass
class GenConfig:
    # rejection sampling: k completions per (document, role) prompt
    n_samples: int = 4
    temperature: float = 0.9
    top_p: float = 0.95
    max_tokens: int = 700
    # cap the synthetic share of DPO rejected samples. A preference set made mostly of
    # hand-corrupted negatives teaches the model to beat a straw man: it learns to avoid
    # errors it never made, and keeps the ones it did.
    max_synthetic_rejected_frac: float = 0.35
    seed: int = 0


@dataclass
class TrainConfig:
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    # attention + MLP. Attention-only LoRA underfits the formatting constraint (strict
    # JSON) that this task lives or dies on.
    target_modules: Sequence[str] = field(default_factory=lambda: (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"))
    sft_epochs: float = 2.0
    sft_lr: float = 1e-4
    dpo_epochs: float = 1.0
    dpo_lr: float = 5e-6          # an order of magnitude below SFT; DPO diverges above ~1e-5
    dpo_beta: float = 0.1
    batch_size: int = 1
    grad_accum: int = 16
    max_seq_len: int = 4096
    warmup_ratio: float = 0.03
    bf16: bool = True
    grad_checkpointing: bool = True


GEN = GenConfig()
TRAIN = TrainConfig()


# ======================================================================================
# paths + the contamination guard
# ======================================================================================
def ensure_dirs() -> None:
    for d in (D_TRAINDATA, D_TRAJ, D_JUDGED, D_SFT, D_DPO, D_ADAPTERS, D_RUNS, D_LOGS):
        os.makedirs(os.path.join(WORK, d), exist_ok=True)


def wpath(*parts: str) -> str:
    return os.path.join(WORK, *parts)


def _ids(path: str, col: str = "") -> set:
    import pandas as pd
    df = pd.read_csv(path, low_memory=False)
    c = col or next((x for x in (COLUMNS["id"], "paper_id", "id") if x in df.columns), "")
    if not c:
        raise SystemExit(f"[FATAL] no id column in {path}. Columns: {sorted(df.columns)[:30]}")
    return {str(v).strip() for v in df[c].dropna()}


def clean_train_csvs(train_csvs: Sequence[str] = (), test_csv: str = "",
                     out_dir: str = "", force: bool = False) -> List[str]:
    """
    Write copies of the TRAIN files with every TEST paper REMOVED, and return their paths.

    WHY THE DROP GOES THIS WAY AND NOT THE OTHER
    -----------------------------------------------------------------------------------
    The overlap has to be resolved on the TRAIN side. Removing the shared papers from TEST
    instead would shrink the 500-paper corpus, and then the fine-tuned rows would be scored
    on a different set of documents from the zero-shot V4 rows — the comparison the whole
    exercise exists to make would no longer be a comparison. Training data is fungible;
    the test set is the measuring stick and must not move.

    Cost: the training corpus loses those papers. That is the correct price. A detector
    evaluated on papers it was trained on does not give a slightly optimistic number, it
    gives one that cannot be interpreted at all.
    """
    import pandas as pd
    train_csvs = list(train_csvs) or TRAIN_CSVS
    test_csv = test_csv or TEST_CSV
    out_dir = out_dir or os.path.join(WORK, D_TRAINCLEAN)
    os.makedirs(out_dir, exist_ok=True)

    test_ids = _ids(test_csv)
    out_paths: List[str] = []
    print(f"[clean] TEST {os.path.basename(test_csv)}: {len(test_ids)} papers — these are "
          f"removed from every TRAIN file")
    total_in = total_drop = 0
    for p in train_csvs:
        dst = os.path.join(out_dir, os.path.basename(p))
        if os.path.exists(dst) and not force:
            df = pd.read_csv(dst, low_memory=False)
            print(f"[clean] reusing {os.path.basename(dst)}: {len(df)} rows")
            out_paths.append(dst)
            continue
        df = pd.read_csv(p, low_memory=False)
        col = next((c for c in (COLUMNS["id"], "paper_id", "id") if c in df.columns), "")
        if not col:
            raise SystemExit(f"[FATAL] no id column in {p}. Columns: {sorted(df.columns)[:30]}")
        mask = df[col].astype(str).str.strip().isin(test_ids)
        n_drop = int(mask.sum())
        kept = df[~mask]
        kept.to_csv(dst, index=False)
        total_in += len(df)
        total_drop += n_drop
        print(f"[clean] {os.path.basename(p):<32} {len(df):>5} rows -> {len(kept):>5} "
              f"({n_drop} dropped for appearing in TEST)")
        if kept.empty:
            raise SystemExit(
                f"[FATAL] {os.path.basename(p)} is EMPTY after removing TEST papers. "
                f"That file is entirely contained in the test set and cannot contribute "
                f"training data — drop it from TRAIN_CSVS in ft_config.py.")
        out_paths.append(dst)
    if total_in:
        print(f"[clean] TOTAL {total_in} rows -> {total_in - total_drop} "
              f"({total_drop} removed, {100.0*total_drop/total_in:.1f}%)")
    print(f"[clean] cleaned copies in {out_dir}")
    print(f"[clean] the ORIGINAL files are untouched; nothing was removed from TEST.")
    return out_paths


def effective_train_csvs(auto_clean: bool = True, force: bool = False) -> List[str]:
    """
    The TRAIN files this pipeline should actually read.

    Cleaned copies when they exist (or can be made), the originals otherwise. Every stage
    calls this rather than TRAIN_CSVS directly, so one stage cannot quietly read the
    uncleaned files while another reads the cleaned ones.
    """
    if not auto_clean:
        return list(TRAIN_CSVS)
    out_dir = os.path.join(WORK, D_TRAINCLEAN)
    cleaned = [os.path.join(out_dir, os.path.basename(p)) for p in TRAIN_CSVS]
    if all(os.path.exists(c) for c in cleaned) and not force:
        return cleaned
    return clean_train_csvs(force=force)


def assert_disjoint(train_csvs: Sequence[str] = (), test_csv: str = "") -> Dict[str, int]:
    """
    TRAIN and TEST must not share a paper. Called by every stage.

    This RAISES rather than warning. A fine-tuned detector evaluated on papers it was
    trained on does not produce a slightly optimistic number, it produces a meaningless
    one — and the failure is invisible in the output table, which is exactly the kind of
    mistake that survives review and then does not replicate.
    """
    train_csvs = list(train_csvs) or TRAIN_CSVS
    test_csv = test_csv or TEST_CSV
    tr: set = set()
    per = {}
    for p in train_csvs:
        s = _ids(p)
        per[os.path.basename(p)] = len(s)
        tr |= s
    te = _ids(test_csv)
    per[os.path.basename(test_csv)] = len(te)
    overlap = tr & te
    print("[split] " + "  ".join(f"{k}={v}" for k, v in per.items()))
    print(f"[split] train {len(tr)} unique papers, test {len(te)}, overlap {len(overlap)}")
    if overlap:
        ex = sorted(overlap)[:10]
        raise SystemExit(
            f"[FATAL] {len(overlap)} paper(s) appear in BOTH train and test: {ex}\n"
            f"        Every metric from a model trained on these is contaminated and the "
            f"contamination does not show up in the table. Remove them from the TRAIN "
            f"files (never from test — test must stay identical to the zero-shot V4 run "
            f"or the comparison is not a comparison), then rerun.")
    return {"n_train": len(tr), "n_test": len(te)}


def describe() -> str:
    L = ["=" * 92, "SFT / DPO PIPELINE — CONFIGURATION", "=" * 92,
         f"  work dir     {WORK}",
         f"  v4 code      {V4_DIR}",
         f"  TRAIN        {', '.join(os.path.basename(p) for p in TRAIN_CSVS)}",
         f"  TEST         {os.path.basename(TEST_CSV)}  (identical to the zero-shot V4 run)",
         f"  models       {', '.join(MODELS)}",
         f"  roles        {', '.join(ROLES)}",
         f"  judge        {JUDGE_MODEL}  (quality ranking ONLY — never ground truth)",
         "",
         "  ABLATIONS (rows of FINAL_TABLE_FT.txt):"]
    for a in ABLATIONS:
        L.append(f"    {a.key:<20} {a.label}")
    L.append("=" * 92)
    return "\n".join(L)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="config + the train/test contamination guard")
    p.add_argument("--no-clean", dest="clean", action="store_false",
                   help="do NOT build cleaned TRAIN copies; just assert and fail on any "
                        "overlap")
    p.add_argument("--force", action="store_true",
                   help="rebuild the cleaned copies even if they already exist")
    a = p.parse_args()

    print(describe())
    ensure_dirs()
    if a.clean:
        paths = effective_train_csvs(auto_clean=True, force=a.force)
        print()
        # The cleaned files MUST now be disjoint. If they are not, the id columns disagree
        # between train and test in a way the filter could not see, and that is worth
        # failing on rather than papering over.
        assert_disjoint(paths, TEST_CSV)
        print("\n[clean] TRAIN is now disjoint from TEST. Every stage reads these copies.")
    else:
        assert_disjoint()
