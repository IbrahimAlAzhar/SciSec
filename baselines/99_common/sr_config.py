#!/usr/bin/env python3
"""
sr_config.py — every path, model, row and hyperparameter of the SafeReview baseline.

WHAT THIS PACKAGE IS
=======================================================================================
"SafeReview (our reimplementation)": Xin et al., SafeReview: Defending LLM-based Review
Systems Against Adversarial Hidden Prompts, arXiv:2604.26506. No code or data were
released, so it is rebuilt from the paper (Algorithm 1, Sections 3-4, Appendices A, E-G)
on YOUR reviewers, YOUR training papers and YOUR test pairs.

    Defender  a LoRA on the reviewer (gemma3-27b-it / qwen3-8b / mistral-small-3.1-24b),
              trained with DPO. prompt = reviewer prompt + ATTACKED paper,
              chosen = the reviewer's review of the CLEAN paper,
              rejected = its review of the ATTACKED paper.
    Attacker  an instruction-tuned LM + LoRA trained with GRPO to write 8-12 sentence
              injections placed at random points of a paper section. Reward (Eq. 1):
              s_adv - s_orig under the current defender.
    Loop      T rounds; at round t the attacker trains against R_{t-1}, then regenerates
              injections, and R_t is DPO-trained from R_{t-1} with reference R_{t-1}.

SafeReview has NO detector: nothing is flagged and nothing is deleted. Its row in the
detection table is "n/a". It is compared where both systems produce the same thing: the
final rating of an attacked paper, and how far honest papers' ratings move.

THE SAME DATA AS EVERY OTHER ROW
    Test:  inputs/<model>/eval_records.jsonl + eval_index.csv, copied byte-for-byte from
           lim_agents_v7_RL_with_argar_btr/inputs (built from v8_defense_data by
           v7ab_inputs.py), so every row is scored on the documents your other baselines
           are scored on, and the table is reported in total AND per source.
    Train: inputs/train_documents.csv, the same training corpus (secondary_280 +
           tertiary_200 + ARGAR/BTR train halves).

Nothing here is a prompt for YOUR agents. The reviewer prompt is imported from
rl_excise.py (the delete-and-re-review arm), so R_0 / R_A / R_def here are produced by
the same reviewer call your excision arm uses.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

# ======================================================================================
# ROOTS
# ======================================================================================
AD_ROOT = os.environ.get(
    "AD_ROOT", ".")
HERE = os.path.dirname(os.path.abspath(__file__))
# Where outputs go. Defaults to this directory, like RL_WORK in v7: renaming the folder
# cannot split the pipeline in two.
WORK = os.environ.get("SR_WORK", HERE)

V7AB_DIR = os.environ.get(
    "V7AB_DIR", os.path.join(AD_ROOT, "defense/lim_agents_v7_RL_with_argar_btr"))
V4_FT_DIR = os.environ.get("V4_FT_DIR", os.path.join(AD_ROOT, "defense/llim_agents_v4_SFT_DPO"))
V5_DIR = os.environ.get("V5_DIR", os.path.join(AD_ROOT, "defense/llm_agents_v5_final"))
V8_DATA_DIR = os.environ.get("V8_DATA_DIR",
                             os.path.join(AD_ROOT, "data/final_data/v8_defense_data"))
MODELS_ROOT = os.environ.get("SR_MODELS_ROOT", "path/to/anonymous/models")


def bootstrap() -> None:
    """V7AB first (rl_config, rl_excise, rl_rewards, rl_load, v7ab_report), then V4's
    ft_* modules. APPENDED after this directory, never before it."""
    for d in (V7AB_DIR, V4_FT_DIR):
        if d and os.path.isdir(d) and d not in sys.path:
            sys.path.append(d)


bootstrap()

try:
    import rl_config as RC                     # v7ab: MODELS, bnb_config, compute_dtype
except Exception as e:                         # pragma: no cover
    raise SystemExit(
        f"[FATAL] cannot import rl_config from {V7AB_DIR}: {type(e).__name__}: {e}\n"
        f"        The reviewer specs (paths, quantisation, chat kwargs, system-role flag)\n"
        f"        are read from your v7ab package so the SafeReview rows load the SAME\n"
        f"        models the other rows do. Set V7AB_DIR and retry.")

MODELS = RC.MODELS                             # gemma3-27b-it, qwen3-8b, mistral-small-3.1-24b
bnb_config = RC.bnb_config
compute_dtype = RC.compute_dtype

# API-only reviewers. SafeReview cannot be applied to them (it needs the weights); only
# the undefended, System Defense and PromptGuard rows exist for these.
API_MODELS = ("gpt-4o-mini",)


# ======================================================================================
# THE ATTACKER
# ======================================================================================
@dataclass
class AttackerSpec:
    key: str
    path: str
    quant: str = "none"                        # serve_vllm_lora.sh argument
    load_4bit: bool = False
    chat_kwargs: Optional[Dict] = None
    system_role: bool = True
    note: str = ""


ATTACKERS: Dict[str, AttackerSpec] = {
    # The paper's main attacker is Qwen3-4B-Instruct; Qwen3-8B is its scale-variant
    # attacker (Table 2). qwen3-8b is already on disk, so it is the default.
    "qwen3-8b": AttackerSpec(
        "qwen3-8b", os.path.join(MODELS_ROOT, "qwen3_8b"),
        chat_kwargs={"enable_thinking": False},
        note="paper Table 2's Qwen3-8B GRPO attacker"),
    "qwen3-4b-instruct": AttackerSpec(
        "qwen3-4b-instruct",
        os.environ.get("SR_ATTACKER_4B_DIR",
                       os.path.join(MODELS_ROOT, "qwen3_4b_instruct_2507")),
        note="the paper's main attacker (Qwen3-4B-Instruct); download it first"),
}
DEFAULT_ATTACKER = os.environ.get("SR_ATTACKER", "qwen3-8b")


# ======================================================================================
# ROWS — what the final table reports
# ======================================================================================
@dataclass
class Row:
    key: str
    label: str
    kind: str                  # "review" (a reviewer produces ratings) | "detector"
    stage: Optional[str]       # adapter stage, or None for the base reviewer
    prompt: str = "plain"      # "plain" | "sysdef"
    needs_weights: bool = False
    note: str = ""


ROWS: Tuple[Row, ...] = (
    Row("SR_undefended", "Undefended reviewer", "review", None, note=(
        "The base reviewer, no defence. Its ratings ARE R_0 (clean) and R_A (attacked) for "
        "every other row, so it must be evaluated first.")),
    Row("SR_sysdef", "System Defense (prompt hardening)", "review", None, prompt="sysdef",
        note="SafeReview's System Defense baseline: an anti-injection notice prepended to "
             "the reviewer's system prompt."),
    Row("SR_promptguard2", "PromptGuard 2 (86M), reject-on-detect", "detector", None,
        note="SafeReview App. A.1: 6 chunks, 512/384 sliding windows, max injection prob, "
             "tau=0.5; a flagged paper's rating becomes 1."),
    Row("SR_static_dpo", "Static DPO (one round, existing attack corpus)", "review",
        "sr_static_dpo", needs_weights=True, note=(
            "One DPO round on your existing training attacks (no attacker). Cheap. The "
            "paper's own Static DPO (one round of GRPO attacks) is SR_safereview_r1.")),
    Row("SR_safereview", "SafeReview (our reimplementation)", "review", "sr_r{round}",
        needs_weights=True, note=(
            "Full co-evolution: GRPO attacker vs DPO defender, T rounds. Evaluated per round "
            "as SR_safereview_r<t>; r1 is the paper's Static DPO, r<T> is SafeReview.")),
)
ROW = {r.key: r for r in ROWS}
STATIC_STAGE = "sr_static_dpo"


def defender_stage(rnd: int) -> str:
    return f"sr_r{int(rnd)}"


def row_dir_name(row_key: str, rnd: Optional[int] = None) -> str:
    """The folder under runs/<model>/. SafeReview gets one folder per round."""
    if row_key == "SR_safereview":
        if rnd is None:
            raise ValueError("SR_safereview needs a round")
        return f"SR_safereview_r{int(rnd)}"
    return row_key


def mkey(model_key: str) -> str:
    return model_key.replace("/", "_")


def lora_name(model_key: str, stage: str) -> str:
    """The served name of a defender adapter == its directory name (as in v7)."""
    return f"{mkey(model_key)}-{stage}"


def adapter_dir(model_key: str, stage: str) -> str:
    return os.path.join(WORK, D_ADAPTERS, lora_name(model_key, stage))


def attacker_name(attacker_key: str, model_key: str, rnd: int) -> str:
    """Attackers are trained AGAINST one reviewer, so the reviewer is in the name."""
    return f"{mkey(attacker_key)}-sratk-{mkey(model_key)}-r{int(rnd)}"


def attacker_dir(attacker_key: str, model_key: str, rnd: int) -> str:
    return os.path.join(WORK, D_ATK_ADAPTERS, attacker_name(attacker_key, model_key, rnd))


# ---- sub-directories under WORK ------------------------------------------------------
D_INPUTS = "inputs"
D_CACHE = "cache"
D_STATIC = "static"
D_COEVO = "coevo"
D_ADAPTERS = "adapters"
D_ATK_ADAPTERS = "adapters_attacker"
D_RUNS = "runs"
D_FINAL = "final"
D_LOGS = "logs"


def wpath(*parts: str) -> str:
    return os.path.join(WORK, *parts)


def round_dir(model_key: str, rnd: int) -> str:
    return wpath(D_COEVO, mkey(model_key), f"round{int(rnd)}")


def inputs_paths(model_key: str, run_tag: str = "", set_name: str = "main"
                 ) -> Tuple[str, str]:
    """(records.jsonl, index.csv) for one reviewer's test set."""
    d = wpath(D_INPUTS, model_key)
    if set_name == "main":
        sfx = f"__{run_tag}" if run_tag else ""
        return (os.path.join(d, f"eval_records{sfx}.jsonl"),
                os.path.join(d, f"eval_index{sfx}.csv"))
    # adaptive test sets written by `sr_attacker.py regen --test`
    sfx = f"__{set_name}" + (f"__{run_tag}" if run_tag else "")
    return (os.path.join(d, f"adaptive_records{sfx}.jsonl"),
            os.path.join(d, f"adaptive_index{sfx}.csv"))


def run_dir(model_key: str, row_dirname: str, run_tag: str = "", set_name: str = "main"
            ) -> str:
    name = row_dirname
    if set_name != "main":
        name += f"__set-{set_name}"
    if run_tag:
        name += f"__tag-{run_tag}"
    return wpath(D_RUNS, mkey(model_key), name)


# ======================================================================================
# HYPERPARAMETERS — paper value, and ours where they differ (and why)
# ======================================================================================
@dataclass
class SRConfig:
    # ---- co-evolution (Algorithm 1) ---------------------------------------------------
    rounds: int = 3                 # paper App. A.2: T = 3
    # ---- attacker GRPO (Eq. 1-4, App. A.2) ----------------------------------------------
    group_size: int = 8             # paper: G = 8
    atk_papers: int = 64            # papers per GRPO batch B. Paper: not stated.
    atk_iters: int = 1              # rollout -> update cycles per round. Paper trains the
                                    # attacker ONLINE (vLLM reward server on 4 GPUs); one
                                    # 40 GiB card forces offline cycles, see README.
    atk_temperature: float = 1.0
    atk_top_p: float = 1.0
    atk_max_new_tokens: int = 384   # paper: 256. 8-12 sentences of "generic but strong
                                    # language" regularly exceed 256 tokens.
    atk_ctx_chars: int = 6000       # paper: the paper text, truncated to the context
    atk_lr: float = 5e-6            # paper: 5e-7 FULL fine-tuning. A LoRA at 5e-7 is inert
                                    # (v7 measured lr 1e-6 -> mean_kl 0.0009, unchanged).
    atk_beta_kl: float = 0.02       # paper: 0.02
    atk_clip_eps: float = 0.2       # paper: eps_clip (value not stated)
    atk_grad_accum: int = 4         # paper: per-device batch 1 x 4 GPUs = 4
    atk_epochs: int = 1
    atk_kl_max: float = 10.0        # v7's trust region: abort without saving past this
    atk_kl_clamp: float = 5.0
    atk_min_sentences: int = 3      # below this the completion is not an injection
    atk_max_chars: int = 2500
    invalid_reward: float = -1.0    # malformed / refused completion
    rank_weight: float = 0.0        # Eq. 1 is the score increase ONLY. >0 adds the batch
                                    # ranking displacement (see sr_attacker.rank_term).
    # ---- defender DPO (Eq. 5, App. A.2 / G) ----------------------------------------------
    dpo_beta: float = 0.1           # paper: 0.1
    dpo_lr: float = 1e-5            # paper: 8e-7 FULL fine-tuning; LoRA needs more
    dpo_batch: int = 8              # paper: 8
    dpo_steps: int = 30             # paper: 30 steps per iteration (Fig. 6)
    dpo_warmup_ratio: float = 0.03  # paper: linear warm-up over the first 3%
    dpo_papers: int = 320           # papers re-attacked per round for D_pref (>= 30 x 8)
    dpo_max_seq_len: int = 8192
    pair_filter: str = "all"        # Algorithm 1 adds EVERY (p+tau, y+, y-) pair
    keep_history: bool = False      # Algorithm 1 resets D_pref <- {} every round
    static_max_pairs: int = 512     # static DPO: pairs reviewed from the existing corpus
    # ---- LoRA (identical to v7's GRPO adapters, which vLLM already serves) -------------
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj",
                                       "gate_proj", "up_proj", "down_proj")
    # ---- the reviewer (rl_excise.review, unchanged) -----------------------------------
    review_max_chars: int = 24000   # rl_excise.review default: the LAST 24,000 chars
    review_max_tokens: int = 420    # rl_excise.review
    review_temperature: float = 0.0
    n_reviews: int = 1              # paper: 4 simulated reviewers, averaged
    # ---- evaluation ----------------------------------------------------------------------
    invariance_tol: float = 0.30    # rl_config.REWARD.invariance_tol
    default_accept_rate: float = 0.33   # App. A.3 (DeepReview-13k acceptance rate)
    seed: int = 0


SR = SRConfig()

# What the paper reports and what this reimplementation must reproduce in direction.
PAPER = {
    "undefended_jump_avg": 1.61,       # Table 1, average over 5 reviewers, 100 ICLR-2025
    "undefended_jump_dr14b": 0.31,     # Table 1, DeepReviewer-14B
    "spearman_clean_dr14b": 0.365,     # Table 2 / 4
    "spearman_safereview_qwen4b": 0.409,
    "fpr_fnr_safereview_qwen4b": (0.382, 0.337),
}

PAPER_VS_OURS = (
    ("defender model", "DeepReviewer-14B, full fine-tuning", "your reviewer + LoRA r=32"),
    ("attacker model", "Qwen3-4B-Instruct (main), Qwen3-8B, Llama-3.2-3B",
     f"{DEFAULT_ATTACKER} + LoRA (qwen3-4b-instruct selectable)"),
    ("training papers", "500 NeurIPS-2024 (1:1 accept/reject)",
     "your training corpus (inputs/train_documents.csv)"),
    ("test papers", "DeepReview-13k test (1,286 ICLR-2024)", "your test pairs (eval_index.csv)"),
    ("reviewer output", "DeepReviewer structured review, 4 simulated reviewers",
     "rl_excise reviewer JSON (rating + weaknesses), n_reviews=1, T=0"),
    ("rounds T", "3", "3 (ROUNDS)"),
    ("GRPO", "online, vLLM reward server, lr 5e-7 full FT, beta 0.02, G=8",
     "offline rollout->score->update cycles, LoRA lr 5e-6, beta 0.02, G=8"),
    ("GRPO ratio", "sequence-level in Eq. 2", "token-level (standard GRPO / OpenRLHF)"),
    ("DPO", "beta 0.1, lr 8e-7 full FT, batch 8, 30 steps",
     "beta 0.1, LoRA lr 1e-5, batch 8, 30 steps"),
    ("D_pref", "all regenerated pairs, reset each round", "same (PAIR_FILTER=all, no history)"),
    ("injection placement", "random position in Abstract/Intro/Method/Exp/Conclusion",
     "same sections when headed; else a random paragraph; inside the reviewer's window"),
)


def describe() -> str:
    L = ["=" * 92, "SAFEREVIEW (our reimplementation) — CONFIGURATION", "=" * 92,
         f"  work dir     {WORK}",
         f"  v7ab dir     {V7AB_DIR}",
         f"  v4 SFT/DPO   {V4_FT_DIR}",
         f"  v8 data      {V8_DATA_DIR}",
         f"  reviewers    {', '.join(MODELS)} (+ API: {', '.join(API_MODELS)})",
         f"  attacker     {DEFAULT_ATTACKER}  ({ATTACKERS[DEFAULT_ATTACKER].path})",
         "", "  ROWS"]
    for r in ROWS:
        L.append(f"    {r.key:<18} {r.label}")
    L += ["", "  PAPER vs OURS"]
    for k, p, o in PAPER_VS_OURS:
        L.append(f"    {k:<20} paper: {p}")
        L.append(f"    {'':<20} ours : {o}")
    L += ["", "  HYPERPARAMETERS"]
    for k, v in asdict(SR).items():
        L.append(f"    {k:<22} {v}")
    L.append("=" * 92)
    return "\n".join(L)


if __name__ == "__main__":
    print(describe())
