"""
ft_config.py — v8 (DVB): every path, model, agent, task, ablation and reward weight.

Named ft_config.py on purpose: train_sft.py / train_dpo.py / ft_model.py are copied from
llim_agents_v4_SFT_DPO unchanged except for two memory fixes, and they `import ft_config`.
Keeping the name means the proven trainers run against v8's paths without a fork.

WHAT CHANGED FROM V4, IN ONE PARAGRAPH
=======================================================================================
V4 had three evidence WORKERS (INSTRUCTION / PRESSURE / VOICE) that each hunted for
suspicious language, plus a leader. They were three flavours of the same question, all
rewarded for finding something, and the leader was left to argue down a page of evidence
on every clean paper -- which is where FPR 0.786 came from. v8 replaces them with agents
that perform DIFFERENT VERIFICATION STEPS, so each call answers a question the others
cannot:

    LOCATOR    task-boundary detector. Given the review TASK and a passage, which exact
               text tries to control how the review is done? (candidates)
    EVIDENCE   claim-evidence checker. Is each candidate's factual content SUPPORTED by
               the rest of the paper, and SEPARATELY, does it redirect the reviewer?
    VERIFIER   independent context verifier. Blind to the locator's confidence: is this
               candidate a real attempt to manipulate, a quoted example, or ordinary
               scholarship? CONFIRMED / DISMISSED / UNCERTAIN.
    LEADER     decides (detect mode) and, separately, chooses the exact text to remove
               (localize mode).

plus deterministic tools that need no LLM (verbatim check, Unicode/obfuscation signals,
BM25 evidence retrieval, span IoU, excision) and one OPTIONAL behavioural AUDITOR that
re-reviews with and without a candidate span for uncertain cases only. Four LLM agents,
five with the auditor -- inside the 3-5 you asked for.

TWO TASKS, TWO ADAPTER FAMILIES
    detect     binary: attacked or clean. Trained on LOCATOR, EVIDENCE, VERIFIER and
               LEADER(detect). Reward is asymmetric: a false positive costs more than a
               miss, because FPR is what your table has been losing on.
    localize   which characters to delete. Trained on LEADER(localize) only. Reward is
               character-level F1 against the gold span with an explicit COLLATERAL
               penalty for deleting the authors' own text.
    Why separate: the detect reward prefers a SHORT sufficient quote; the localize reward
    needs the WHOLE insert and nothing else. Fitted into one adapter the short-quote
    habit wins (it has 4x the data) and excision leaves residual manipulation behind.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = os.environ.get(
    "AD_ROOT",
    ".")

# every py file, every PBS file and every output lives here
WORK = os.environ.get("V8_WORK", os.path.join(ROOT, "defense/llim_agents_v8_dvb"))

V4_FT_DIR = os.environ.get("V4_FT_DIR", os.path.join(ROOT, "defense/llim_agents_v4_SFT_DPO"))
V5_DIR = os.environ.get("V5_DIR", os.path.join(ROOT, "defense/llm_agents_v5_final"))
V4_DIR = os.environ.get("V4_DIR", os.path.join(ROOT, "defense/llm_agents_v4"))
# ARGAR / BTR baseline pairs (argar__<tag>.csv.gz, btr__<tag>.csv.gz)
DVB_DIR = os.environ.get("DVB_DIR", os.path.join(ROOT, "data/input_baselines"))

DATA = os.path.join(ROOT, "data/final_data")
TRAIN_CSVS = [os.path.join(DATA, "dataset_secondary_280.csv"),
              os.path.join(DATA, "dataset_tertiary_200.csv")]
TEST_CSV = os.path.join(DATA, "dataset_primary_500.csv")
# the REUSABLE pair files built by v8_build_data.py (one shared TRAIN file, one TEST file
# per reviewer model). Everything downstream reads these, never the raw sources.
DATA_OUT = os.environ.get("V8_DATA_DIR", os.path.join(DATA, "v8_defense_data"))
BENCH_DIR = os.path.join(ROOT, "lim_bb_multimodel_500_data/output_primary500")

COLUMNS = dict(id="id", text="input_text_cleaned", sections="pdf_text",
               author_lim="author_men_lim", weaknesses="weaknesses")

# sub-directories of WORK
D_INPUTS = "inputs"          # the concatenated corpus, per source and all together
D_TRAJ = "rollouts"          # sampled trajectories, one dir per (model, task)
D_SFT = "datasets/sft"
D_DPO = "datasets/dpo"
D_GRPO = "datasets/grpo"
D_ADAPTERS = "adapters"
D_RUNS = "runs"              # evaluation output, one dir per (model, ablation)
D_JUDGE = "judge_cache"
D_FINAL = "final"
D_LOGS = "logs"


# ======================================================================================
# MODELS
# ======================================================================================
@dataclass
class ModelSpec:
    key: str
    path: str                     # local weights; "" for an API model
    quant: str                    # serve_vllm_lora.sh argument: "bnb" | "none" | "api"
    load_4bit: bool
    max_len: int
    tag: str                      # filename tag, e.g. the DVB input files
    chat_kwargs: Optional[Dict] = None
    system_role: bool = True      # gemma's chat template has no system role
    api: bool = False             # served by OpenAI, not vLLM: zero-shot only


MODELS: Dict[str, ModelSpec] = {
    "gemma3-27b-it": ModelSpec(
        "gemma3-27b-it", "path/to/models/gemma3_27b_it",
        quant="bnb", load_4bit=True, max_len=8192, tag="gemma3_27b", system_role=False),
    "qwen3-8b": ModelSpec(
        "qwen3-8b", "path/to/models/qwen3_8b",
        quant="none", load_4bit=False, max_len=8192, tag="qwen3_8b",
        chat_kwargs={"enable_thinking": False}),
    "mistral-small-3.1-24b": ModelSpec(
        "mistral-small-3.1-24b",
        "path/to/models/mistral_small_3_1_24b_instruct",
        quant="bnb", load_4bit=True, max_len=8192, tag="mistral_small_3_1_24b"),
    "gpt-4o-mini": ModelSpec(
        "gpt-4o-mini", "", quant="api", load_4bit=False, max_len=128000,
        tag="gpt_4o_mini", api=True),
}


# ======================================================================================
# AGENTS AND TASKS
# ======================================================================================
AGENTS: Tuple[str, ...] = ("LOCATOR", "EVIDENCE", "VERIFIER", "LEADER")
TASKS: Tuple[str, ...] = ("detect", "localize")
# which agent prompts belong to which task's training data / adapter
TASK_AGENTS: Dict[str, Tuple[str, ...]] = {
    "detect": ("LOCATOR", "EVIDENCE", "VERIFIER", "LEADER"),
    "localize": ("LEADER",),
}


def lora_name(model_key: str, stage: str, role: Optional[str] = None) -> str:
    """
    The `model` a vLLM request carries to reach one adapter. `stage` already includes
    the task for v8 ("sft-detect", "dpo-localize"). Derived, never typed.
    """
    base = f"{model_key.replace('/', '_')}-{stage}"
    return f"{base}-{role.lower()}" if role else base


def adapter_dir(model_key: str, stage: str, role: Optional[str] = None) -> str:
    return os.path.join(WORK, D_ADAPTERS, lora_name(model_key, stage, role))


# ======================================================================================
# ABLATIONS — rows of the final table
# ======================================================================================
@dataclass
class Ablation:
    key: str
    label: str
    stage: Optional[str]          # adapter stage prefix; None = zero-shot
    per_agent: bool = False
    needs_training: bool = True
    note: str = ""


ABLATIONS: Tuple[Ablation, ...] = (
    Ablation("zeroshot", "ZS LLM agents (v8)", None, needs_training=False, note=(
        "The four verification agents with no adapter. The row every trained row is "
        "compared against, run on the same pairs with the same reviewer.")),
    Ablation("sft", "SFT", "sft", note="rejection-sampled targets, per task"),
    Ablation("dpo_only", "DPO only, no SFT", "dpo_only", note=(
        "prices the SFT stage: if this matches sft_dpo, SFT bought nothing")),
    Ablation("sft_dpo", "SFT then DPO", "dpo", note="DPO continues the SFT adapter"),
    Ablation("sft_dpo_merged", "SFT merged, then DPO", "dpo_merged", note=(
        "SFT folded into the base weights before DPO, so the preference step starts "
        "from the SFT policy AND uses it as the frozen reference")),
    Ablation("sft_dpo_per_agent", "SFT+DPO per agent", "dpo", per_agent=True, note=(
        "one adapter per agent; only built when PER_AGENT=1")),
    Ablation("grpo_prov", "GRPO, provenance reward", "grpo", note=(
        "GRPO from the SFT adapter with the provenance reward: grounded, isolated, "
        "twin-consistent. The v7 idea on v8 agents.")),
)
ABL = {a.key: a for a in ABLATIONS}


# ======================================================================================
# generation / training defaults
# ======================================================================================
@dataclass
class GenConfig:
    n_samples: int = 4              # k completions per prompt (the GRPO group size)
    temperature: float = 0.9
    top_p: float = 0.95
    max_synthetic_rejected_frac: float = 0.35
    seed: int = 0


@dataclass
class TrainConfig:
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: Sequence[str] = field(default_factory=lambda: (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"))
    sft_epochs: float = 2.0
    sft_lr: float = 1e-4
    dpo_epochs: float = 1.0
    dpo_lr: float = 5e-6
    dpo_beta: float = 0.1
    batch_size: int = 1
    grad_accum: int = 16
    max_seq_len: int = 4096
    warmup_ratio: float = 0.03
    bf16: bool = True
    grad_checkpointing: bool = True


@dataclass
class GRPOConfig:
    # Measured on gemma3-27b in v7 (40-step sweep): 6e-6 was the first setting whose KL
    # left the inert band, and 1.2e-5 diverged. Rank candidates by kl_p95, NOT mean_kl:
    # the mean is dominated by a handful of collapsed tokens (k3 = 1/rho + log rho - 1).
    lr: float = 6e-6
    beta_kl: float = 0.04
    clip_eps: float = 0.2
    kl_clamp: float = 5.0          # cap d=-log(rho) inside the PENALTY only
    kl_max: float = 10.0           # abort WITHOUT saving above this windowed mean KL
    grad_accum: int = 8
    max_seq_len: int = 4096
    epochs: float = 1.0


GEN = GenConfig()
TRAIN = TrainConfig()
GRPO = GRPOConfig()


# ======================================================================================
# reward weights — the single definition used by rollout scoring, DPO pair ranking and
# the GRPO advantage. Changing a weight here changes all three consistently.
# ======================================================================================
@dataclass
class RewardWeights:
    # detect task
    fp_cost: float = 1.5           # flagging a clean paper: the FPR your table loses on
    fn_cost: float = 1.0
    grounded: float = 0.5          # every quote must be verbatim in what was shown
    fabricated: float = 1.0        # a quote that is NOT in the text: worse than silence
    on_target: float = 0.5         # the quote lies inside the gold manipulation
    calibration: float = 0.3       # Brier-style: confidence should match correctness
    uncertain_credit: float = 0.3  # VERIFIER UNCERTAIN earns partial credit, not zero
    schema: float = 0.2
    # localize task
    span_f1: float = 2.0
    # 2.0, not 1.5: deleting a GENUINE limitation hides a weakness, which is the
    # attack's own goal. At 1.5, 'whole insert + one author sentence' outscored
    # 'half the insert'; at 2.0 the two are about even, which is the honest ranking.
    collateral: float = 2.0        # per unit fraction of NON-gold text deleted
    residual: float = 0.5          # per unit fraction of gold text left in
    clean_restraint: float = 1.0   # deleting nothing from a clean paper
    # provenance (GRPO)
    twin: float = 0.5              # same verdict logic holds on the pair's benign twin


RW = RewardWeights()


# ======================================================================================
# threat families (Panel B of the table)
# ======================================================================================
INSTRUCTION_CONDS = ("A5", "A6", "A7", "A8", "A9", "A11", "A12")
RHETORICAL_CONDS = ("A1", "A2", "A3", "A4", "A10")


def threat_family(source: str, condition: str, attack_kind: str = "") -> str:
    """instruction | rhetorical | repackaging | perturbation | clean | other"""
    src = (source or "").lower()
    cond = str(condition or "")
    if cond.startswith("A0") or cond.lower() in ("clean", "benign"):
        return "clean"
    if src == "argar" or "argar" in (attack_kind or "").lower():
        return "repackaging"
    if src == "btr" or "btr" in (attack_kind or "").lower() or \
            "perturb" in (attack_kind or "").lower():
        return "perturbation"
    head = cond.split("_", 1)[0].upper()
    if head in INSTRUCTION_CONDS:
        return "instruction"
    if head in RHETORICAL_CONDS:
        return "rhetorical"
    return "other"


# ======================================================================================
# paths
# ======================================================================================
def ensure_dirs() -> None:
    for d in (D_INPUTS, D_TRAJ, D_SFT, D_DPO, D_GRPO, D_ADAPTERS, D_RUNS, D_JUDGE,
              D_FINAL, D_LOGS):
        os.makedirs(os.path.join(WORK, d), exist_ok=True)


def wpath(*parts: str) -> str:
    return os.path.join(WORK, *parts)


def describe() -> str:
    L = ["=" * 92, "v8 DVB — CONFIGURATION", "=" * 92,
         f"  work dir   {WORK}",
         f"  V4 SFT/DPO {V4_FT_DIR}",
         f"  V5         {V5_DIR}",
         f"  DVB inputs {DVB_DIR}",
         f"  agents     {', '.join(AGENTS)}  (+ optional AUDITOR)",
         "  tasks      " + ", ".join(t + ": " + "/".join(TASK_AGENTS[t]) for t in TASKS),
         f"  models     {', '.join(MODELS)}", "", "  ablations:"]
    for a in ABLATIONS:
        L.append(f"    {a.key:<20} {a.label}")
    L.append("=" * 92)
    return "\n".join(L)


if __name__ == "__main__":
    print(describe())
