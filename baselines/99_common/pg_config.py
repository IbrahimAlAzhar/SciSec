"""
pg_config.py — paths and defaults for the PaperGuard baseline, in one place.

WHERE EVERYTHING LIVES
    code + every output   defense_baselines/paperguard_defense
                          (runs/, emb_cache/, logs/, .pycache/ are all created under it)
    read only             the SFT/DPO code dir (infer_ladder.load_eval_set, its runs/ for
                          the comparison table, traindata_shared/ for the `train` registry)
                          and the V4 dir (def_dataset.build_eval_set)

The baseline is only comparable to the SFT/DPO rows if it reads THE SAME TEST CORPUS and
writes THE SAME ROW SCHEMA, so the inputs are derived from the same environment variables
the SFT/DPO jobs use (AD_ROOT, V4_DIR):

    TEST   dataset_primary_500.csv + the gemma3-27b-it/seed2 metrics_detail.csv, built into
           clean/attacked documents by def_dataset.build_eval_set (imported through
           infer_ladder.load_eval_set, never re-implemented). "gemma3-27b-it" here names
           the ATTACKED CORPUS your SFT/DPO rows were scored on; it is not the detector.
    LLM    gpt-4o-mini, for PaperGuard's verifier and for the re-rating reviewer.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

AD_ROOT = os.environ.get(
    "AD_ROOT", ".")
FT_CODE_DIR = os.environ.get(
    "FT_CODE_DIR", os.path.join(AD_ROOT, "defense/llim_agents_v4_SFT_DPO"))
V4_DIR = os.environ.get("V4_DIR", os.path.join(AD_ROOT, "defense/llm_agents_v4"))

# The deployment location you asked for. The code runs from wherever it sits (PG_DIR); the
# PBS runs it from here and points PG_WORK here, so every output lands under this folder.
DEPLOY_DIR = os.path.join(AD_ROOT, "defense_baselines/paperguard_defense")
PG_DIR = os.path.dirname(os.path.abspath(__file__))
PG_WORK = os.environ.get("PG_WORK", PG_DIR)

TEST_CSV = os.path.join(AD_ROOT, "data/final_data/dataset_primary_500.csv")

# the attacked corpus the SFT/DPO Panel-A rows use (job_c_gemma_infer.pbs)
CORPUS_MODEL = "gemma3-27b-it"
# the chat model PaperGuard's verifier (and the re-rating reviewer) calls
LLM = os.environ.get("PG_LLM", "gpt-4o-mini")


def attacked_csv(corpus_model: str = CORPUS_MODEL, seed: str = "seed2") -> str:
    """The V4 metrics_detail.csv the SFT/DPO inference job reads (job_c_gemma_infer.pbs)."""
    return os.path.join(AD_ROOT, "lim_bb_multimodel_500_data/output_primary500",
                        corpus_model, seed, "metrics_detail.csv")


def train_documents_candidates() -> List[str]:
    """Where ft_attack.py wrote the attacked TRAIN corpus (job_ft_a_gemma_rollout.pbs)."""
    return [os.path.join(FT_CODE_DIR, "traindata_shared", "train_documents.csv"),
            os.path.join(FT_CODE_DIR, "traindata_gemma3_27b", "train_documents.csv"),
            os.path.join(FT_CODE_DIR, "traindata", "train_documents.csv")]


def find_train_documents() -> Optional[str]:
    for p in train_documents_candidates():
        if os.path.exists(p):
            return p
    return None


# ======================================================================================
# PaperGuard's own defaults (main_defense.py / emb_search.py / README / paper App. E.1)
# ======================================================================================
@dataclass
class PGDefaults:
    top_k: int = 5                  # README + main_defense.py --top-k default
    max_len: int = 800              # _split_text(max_len=800) in detect()
    a1_size: int = 10               # main_defense.py --a1-size
    seed: int = 42                  # main_defense.py --seed
    embed_model: str = "text-embedding-3-large"   # main_defense.py --model-name
    embed_batch: int = 256          # _embed_openai(batch_size=256)
    verifier_max_tokens: int = 200  # DEFENSE_EMB_MAX_TOKENS default
    verifier_temperature: float = 0.0   # DEFENSE_EMB_TEMPERATURE default
    review_max_tokens: int = 2048   # paper App. E.1 (max new tokens)
    # the paper samples reviews at 0.3; pg_rerate.py defaults to 0.0 so that a rating
    # change between B / A / S is caused by the text, not by sampling.
    review_temperature: float = 0.3


PG = PGDefaults()


# ======================================================================================
# the rows this baseline adds to the table
# ======================================================================================
@dataclass
class Registry:
    key: str            # --registry value
    ablation: str       # directory prefix under runs/
    label: str          # row label (the LLM is appended: "... + gpt-4o-mini")
    role: str           # "baseline" | "ablation" — decides where pg_table.py reports it
    note: str


REGISTRIES: Dict[str, Registry] = {
    # THE BASELINE. Strictly PaperGuard: its own known-attack list, nothing of ours.
    "paperguard": Registry(
        "paperguard", "paperguard_orig", "PaperGuard (released registry)", "baseline",
        "PaperGuard's own known-attack set: A1 = split_prompts_into_groups(seed=42, "
        "a1_size=10), exactly as main_defense.py builds it. Every attack in our corpus is "
        "therefore an UNKNOWN attack for it -- the paper's zero-shot transfer setting. "
        "This is the only row reported as the PaperGuard baseline."),
    # AN ABLATION, NOT A BASELINE. PaperGuard's A1 with our TRAIN payloads appended.
    "combined": Registry(
        "combined", "paperguard_plus_train", "PaperGuard registry + our TRAIN attacks",
        "ablation",
        "ABLATION, not a baseline: PaperGuard's A1 (the same 10 prompts, first, in the "
        "same order) followed by one medoid payload per attack family from the attacked "
        "TRAIN documents our SFT/DPO adapters were trained on. Asks whether telling "
        "PaperGuard about our attack families helps. TEST papers are excluded by an "
        "assertion. Verifier answers for the 10 shared patterns are reused from the "
        "baseline run (same prompt -> same cached answer), so the two rows differ ONLY by "
        "the added patterns."),
}


def llm_slug(llm: str) -> str:
    return re.sub(r"[^A-Za-z0-9.\-]+", "-", os.path.basename(llm.rstrip("/"))).strip("-")


def ablation_name(registry_key: str, llm: str = LLM) -> str:
    """runs/<this>/ — e.g. paperguard_orig__gpt-4o-mini. The LLM is in the name so a run
    with a different verifier can never overwrite or resume this one."""
    return f"{REGISTRIES[registry_key].ablation}__{llm_slug(llm)}"


def row_label(registry_key: str, llm: str = LLM) -> str:
    r = REGISTRIES[registry_key]
    lab = f"{r.label} + {llm_slug(llm)}"
    return lab + " [ablation]" if r.role == "ablation" else lab


def runs_root(work: str = "") -> str:
    return os.path.join(work or PG_WORK, "runs")


def run_dir(ablation: str, work: str = "") -> str:
    return os.path.join(runs_root(work), ablation)
