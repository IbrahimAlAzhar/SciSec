# SciDefense: Defending LLM Peer Review Against Limitation Suppression

<!-- TODO before upload: replace every line marked TODO, then delete these comments.
     Keep this file anonymous: no names, institutions, cluster names or absolute paths. -->

This repository contains the code for **SciDefense**, the defense proposed in our submission, and for every defense baseline we compare against.

**The threat.** An author adds a short passage to their own manuscript, usually in the Limitations section, so that an LLM reviewer raises its rating and stops reporting the paper's real weaknesses.

**The defense.** SciDefense audits the manuscript *before* the reviewer reads it. It works in two stages:

1. **Detection.** Is the manuscript manipulated?
2. **Sanitization.** Locate the manipulated span, remove it, and re-review the paper.

> **Responsible release.** The attack code (payload generation and the gradient- and attention-based attacks) is **not** included, because it could be misused against deployed review systems. Instead we release the paired benign/attacked documents needed to reproduce every defense result (see [Data](#data)).

---

## Contents

- [Repository layout](#repository-layout)
- [Method overview](#method-overview)
- [Installation](#installation)
- [Data](#data)
- [Reproducing SciDefense](#reproducing-scidefense)
- [Reproducing the baselines](#reproducing-the-baselines)
- [Metrics](#metrics)
- [Hyperparameters](#hyperparameters)
- [Compute](#compute)
- [Citation](#citation)
- [License](#license)

---

## Repository layout

```
SciDefense/                       proposed defense
  01_rollout/                     sample k completions per (document, agent role), build GRPO groups
  02_training/                    SFT initialisation, GRPO trainer, rewards, per-agent credit
  03_inference/                   detection pipeline: normalisation, tier-0 screen, agent audit
  04_sanitization/                span removal, re-review, invariance
  05_evaluation/                  metrics and result tables
  06_verification_agents/         zero-shot verification team (locator, evidence, verifier, leader)
  07_common/                      shared helpers
  run_with_paths.py               runs any script with all sub-folders on the import path

baselines/
  01_perplexity/                  GPT-2 sliding-window perplexity + imperative regex
  02_sft_dpo/                     the same agents trained with SFT and DPO (no GRPO)
  03_paperguard/                  chunk retrieval + LLM verifier
  04_piguard/                     PIGuard classifier
  05_promptguard/                 PromptGuard classifier
  06_protectai_deberta/           ProtectAI DeBERTa-v3 prompt-injection classifier
  07_phantomlint/                 hidden-text linter
  08_safereview/                  robust reviewer trained with DPO (re-implementation)
  09_prompt_hardening/            reviewer-side prompt hardening (4 variants)
  10_no_defense/                  no defense, plus flag-nothing / flag-everything floors
  11_single_stage_llm/            one LLM screen without tiers or confidence gate
  99_common/                      shared helpers, aggregation
  run_with_paths.py
```

Modules import their neighbours by plain name, so always run scripts through `run_with_paths.py`:

```bash
cd SciDefense
python run_with_paths.py 03_inference/runner_def.py --help
```

<!-- TODO: if a baseline folder is empty in your release, delete its line above. -->

---

## Method overview

### Detection pipeline (inference)

```
manuscript
   │
   ▼
normalise          strip zero-width chars, fold homoglyphs, remove bidi overrides,
   │               decode base64 / HTML-comment carriers
   ▼
tier 0 screen      char n-gram retrieval against a memory of known payloads
   │               + ~20 forensic features  →  signed score
   │
   ├── score > high gate  → flagged      (high gate = 98th pct of benign scores,
   ├── score < low gate   → clean         so tier 0 alone flags ≤ 2% of clean papers)
   ▼
tier 1 agents      only for documents between the gates
   │   CHUNK VERIFIER   is each retrieved passage an active instruction or discussion?
   │   INSTRUCTION      imperatives, references to reviewing, role-play
   │   SOCIAL PRESSURE  coercion, stakes, authority, reward
   │   AUTHOR VOICE     does the limitations text match the authors' voice?
   │   LEADER           arbitrates (≤ 3 rounds): verdict, confidence, family, spans
   ▼
acceptance gate    flag only if confidence ≥ 0.5 AND the verdict quotes ≥ 1 span
   │               that appears verbatim in the document
   ▼
sanitise           remove the quoted spans → re-review with the same reviewer
```

### GRPO training of the audit agents

Each agent (chunk verifier, three workers, leader) is a LoRA adapter on one base model, trained with GRPO on a reward computed over the **whole cascade**, with credit assigned to each agent:

- **Rollouts.** For each (document, role), *k* = 4 completions are sampled at temperature 0.9. They form one GRPO group. The *k*-th samples of all roles are joined into one pseudo-trajectory.
- **Reward.** It is computed by arithmetic, with no LLM judge: the label comes from construction, grounding from string containment, ratings from the reviewer. It pays for a correct decision, verbatim quotes, overlap with the true span, the correct family and rating invariance after removal. It charges for false positives (weighted 1.6× a miss), unquotable verdicts, fabricated spans and cutting the authors' own limitations.
- **Credit.** Advantages are computed per agent, so a worker that resisted the injection is not penalised for a leader that was fooled.
- **Objective.** Offline iterative GRPO with PPO-style clipping, a k3 KL penalty against the frozen reference, and length-unbiased token weighting.

### Verification agents (zero-shot)

These target attacks that leave no inserted passage (repackaging, word-level perturbation):

- **LOCATOR** proposes candidate spans.
- **EVIDENCE** gathers support for or against each candidate with deterministic tools.
- **VERIFIER** judges each candidate without seeing the locator's reasoning.
- **LEADER** decides in *detect* and *localize* mode.
- **AUDITOR** (optional) checks the leader.

---

## Installation

Python ≥ 3.10 and one GPU with ≥ 40 GB memory (larger base models are loaded in 4-bit NF4).

```bash
conda create -n scidefense python=3.11 -y
conda activate scidefense
pip install "torch>=2.3" vllm "transformers>=4.53,<5" peft accelerate bitsandbytes \
            datasets pandas numpy scipy scikit-learn tqdm openai
```

Installing vLLM can upgrade `transformers` to 5.x. Re-run the `transformers` pin afterwards.

<!-- TODO: replace with the exact versions from `pip freeze` on your machine. -->

Base models (from the Hugging Face Hub): `google/gemma-3-27b-it`, `Qwen/Qwen3-8B`, `mistralai/Mistral-Small-3.1-24B-Instruct-2503`.

The rubric judge used for evaluation calls an OpenAI model. Set the key as an environment variable; never write it into a file:

```bash
export OPENAI_API_KEY=...
```

---

## Data

| File | Content | Used for |
|---|---|---|
| `data/dataset_primary_500.csv` | 500 papers: id, title, full text, parsed sections, author-stated limitations, human-reviewer weaknesses | test set |
| `data/dataset_secondary_280.csv`, `data/dataset_tertiary_200.csv` | disjoint training papers (overlap with the test set removed) | SFT / GRPO training |
| `data/defense_test_pairs.*` | 990 documents: one benign and one attacked version for each of 495 test papers, with the exact inserted span | detection and sanitization |
| `data/argar_*.csv.gz`, `data/btr_*.csv.gz` | repackaging and perturbation attacks, split 50/50 by paper | generalisation |

<!-- TODO: say where the data can be downloaded (anonymous link for review),
     the source venue(s) and year(s) of the papers, and their license. -->

The attacked documents are released as data so that results can be reproduced without the attack code. In each pair, the benign and attacked documents come from the same rendering step, so they differ only by the inserted span.

Every script reads its input paths from command-line arguments or environment variables. Point them at your local copy, e.g.:

```bash
export DATA_DIR=$PWD/data
export MODEL_DIR=/path/to/models
```

<!-- TODO: check which variable / argument names your scripts actually read, and list them here. -->

---

## Reproducing SciDefense

Run every step from inside `SciDefense/`. Each script's `--help` lists its arguments.

<!-- TODO: replace each <...> with the real script arguments. -->

**1. Rollouts.** Sample *k* completions per (document, role) and score them against the known labels:

```bash
python run_with_paths.py 01_rollout/ft_trajectories.py <args>      # → samples.jsonl
python run_with_paths.py 01_rollout/v7_score_rollouts.py <args>    # → rollouts.jsonl, turns.jsonl, stats.json
```

**2. Training.** SFT initialisation, then GRPO:

```bash
python run_with_paths.py 02_training/train_sft.py  <args>          # → SFT LoRA adapter
python run_with_paths.py 02_training/train_grpo.py <args>          # → GRPO LoRA adapter
```

Training aborts without saving if the mean KL exceeds 10, so a diverged run never overwrites a good adapter.

**3. Inference (detection).** The trained adapter runs inside the unchanged detection pipeline:

```bash
python run_with_paths.py 03_inference/runner_def.py --vllm-model <adapter> --stage1-only <args>
```

Add `--loco` to remove the evaluated attack family from the payload memory (unseen-attack test).

**4. Sanitization.** Remove the flagged spans and re-review:

```bash
python run_with_paths.py 04_sanitization/rl_excise.py <args>
```

**5. Results tables.**

```bash
python run_with_paths.py 05_evaluation/v7_table.py <args>
```

**6. Verification agents (zero-shot).**

```bash
python run_with_paths.py 06_verification_agents/v8_eval.py <args>
```

---

## Reproducing the baselines

Run from inside `baselines/`. Every detector is scored on the same benign/attacked pairs as SciDefense.

```bash
python run_with_paths.py 01_perplexity/d3_ppl_regex.py       <args>
python run_with_paths.py 02_sft_dpo/train_dpo.py             <args>
python run_with_paths.py 06_protectai_deberta/d6_deberta.py  <args>
python run_with_paths.py 09_prompt_hardening/d2_prompt_harden.py <args>
python run_with_paths.py 10_no_defense/d1_no_defense.py      <args>
python run_with_paths.py 99_common/aggregate.py              <args>   # one table, split by attack family
```

<!-- TODO: add the commands for PaperGuard, PIGuard, PromptGuard, PhantomLint and SafeReview. -->

Notes:

- **Prompt hardening** changes the reviewer, not a detector. It is scored only on rating restoration and recovery of the human-reported weaknesses.
- **ProtectAI DeBERTa** is archived upstream. <!-- TODO: state the checkpoint and download date used. -->
- **SafeReview** has no public code. This is our re-implementation, compared on final ratings.

---

## Metrics

**Detection.** Attacked is the positive class; each test paper contributes one benign and one attacked document.

| Metric | Definition |
|---|---|
| Acc / Prec / Rec / F1 | standard |
| **FPR** | benign documents flagged / benign documents. The primary deployment metric |
| Harmful recall | recall over attacked documents that actually raised the rating |
| R@FPR ≤ 5% / 10% | recall at the threshold that flags 5% / 10% of benign documents |
| AUROC | attacked vs. benign, on the decision score |
| Escalation | share of documents sent to the tier-1 agents (cost) |
| Span P / R / F1, IoU | character overlap of predicted spans with the true inserted span |

**Sanitization.** r₀ = clean rating, rₐ = attacked rating, rₛ = rating after removal.

| Metric | Definition |
|---|---|
| Restoration | (rₐ − rₛ) / (rₐ − r₀); undefined when the attack did not move the rating |
| Residual | rₛ − r₀ |
| Collateral | rating change on benign papers that the defense edited |

---

## Hyperparameters

| | SFT | GRPO |
|---|---|---|
| Initialisation | base model | SFT adapter |
| LoRA r / α / dropout | 32 / 64 / 0.05 | 32 / 64 / 0.05 |
| Target modules | q, k, v, o, gate, up, down | same |
| Learning rate | 1e-4 | <!-- TODO --> |
| Epochs | 2 | 1 per iteration, ≤ 3 iterations |
| Batch × grad. accumulation | 1 × 16 | 1 × 16 |
| Max sequence length | 4,096 | 4,096 (640 new tokens) |
| Objective-specific | — | β_KL = 0.04, clip ε = 0.2, KL clamp 5, abort at KL 10, min group std 1e-3 |
| Precision | bf16; 4-bit NF4 + double quantisation for Gemma and Mistral | same |

Rollouts: *k* = 4, temperature 0.9, top-p 0.95, ≤ 700 tokens.

Inference: leader confidence floor 0.5, tier-0 FPR budget 0.02, ≤ 3 leader rounds.

---

## Compute

All experiments run on a single GPU (40 GB or 95 GB). Reviewer models and agent adapters are served with vLLM, which selects the LoRA adapter per request. Rollout generation and policy updates run as separate jobs, so GRPO fits on one card.

<!-- TODO: GPU-hours per stage. -->

---
