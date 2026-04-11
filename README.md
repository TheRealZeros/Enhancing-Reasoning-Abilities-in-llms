# Enhancing Reasoning Abilities of Large Language Models

Experimental code for the honours thesis:

**Enhancing Reasoning Abilities of Large Language Models**

Bachelor of Engineering (Software Engineering)  
Macquarie University  

---

# Project Overview

This project investigates whether structured prompting changes the internal mechanisms used by transformer models when performing multi-hop reasoning. 

The experiment evaluates the Pythia-2.8B base model on a custom synthetic two-hop reasoning dataset (A→B→C fact-composition chains) engineered for token-aligned activation analysis. The study employs a 2 × 2 factorial design comparing baseline and structured prompting conditions under clean and distractor-augmented contexts.

**Primary analysis method:**
- Activation patching (causal mediation analysis) using TransformerLens — layer-level, component-level (attention vs MLP), and individual attention head patching.

**Secondary diagnostics:**
- Logit lens analysis (tracking emergence of the correct answer representation across layers).
- Attention head visualisation (pending Phase 4b).

---

# Thesis Context and Scope

This repository supports the empirical component of the thesis exploring reasoning in large language models via a three-pillar conceptual framework. 

The thesis explicitly **does not** test all three pillars equally. The experiment is scoped as a mechanistic case study where:
- **Pillar 1 (Memory & Contextual Reasoning)** is not tested as architectural memory, but is instead *proxied* through in-context evidence routing by varying the presence of contextual distractors.
- **Pillar 2 (Structured Reasoning Frameworks)** is the *manipulated independent variable*, operationalised as a structured few-shot prompting intervention.
- **Pillar 3 (Representation & Cognitive Generalisation)** acts as the *observational measurement lens*, examining how internal activation geometry shifts across conditions using interpretability techniques.

---

# Experiment Design

The dataset consists of 200 synthetic two-hop reasoning examples across four domains (geography, science, biology, culture). Each example contains an A→B→C entity chain where the question asks for the composition of relations, and the gold answer is entity C.

Each example is evaluated under five prompt conditions (cells) padded to identical token lengths to enable exact cross-condition activation patching:

| Cell | Prompt Type | Context |
|------|-------------|---------|
| A | Direct (few-shot, direct answers) | Clean (2 supporting facts only) |
| B | Direct | Noisy (+ 3 distractor facts) |
| C | Structured (few-shot, explicit reasoning steps) | Clean |
| D | Structured | Noisy (+ 3 distractor facts) |
| E | Filler control (length-matched neutral padding) | Clean |

**Contrast examples** are defined as cases where the direct prompt fails and the structured prompt succeeds. These examples are isolated for all downstream mechanistic analysis:
- **Clean contrasts:** Cell A incorrect, Cell C correct.
- **Noisy contrasts:** Cell B incorrect, Cell D correct.

---

# Quick Start

From the repository root:

```powershell
# Setup and verify environment
.\setup-env\setup.ps1

# Run full experiment pipeline sequentially
python scripts/phase_0_sanity/prompt_inference_check.py
python scripts/phase_1_dataset/build_dataset.py
python scripts/phase_2_behaviour/run_evaluation.py
python scripts/phase_2_behaviour/run_noisy_contrasts.py
python scripts/phase_3a_layer_patching/activation_patching.py
python scripts/phase_3b_component_patching/component_patching.py --layers 24 25 29 30 31
python scripts/phase_3b_component_patching/head_patching.py --layers 30 31
python scripts/phase_3c_cross_condition/cross_condition_patching.py
python scripts/phase_4_logit_lens/logit_lens_analysis.py --include-noisy
```

---

# Experiment Pipeline & Status

1. **Phase 0 — Model sanity check (Completed):** Verified Pythia-2.8B model loading, deterministic generation, and activation caching.
2. **Phase 1 — Synthetic dataset construction (Completed):** Built 200 token-aligned examples across all five cells with entity chains and distractors.
3. **Phase 2 — Behavioural evaluation (Completed):** Ran all cells, computed exact-match accuracy, and identified 38 clean contrast and 54 noisy contrast examples.
4. **Phase 3 — Activation patching (Completed):**
   - **3a (Completed):** Layer-level residual stream patching identified late layers (24–31) as the primary mediators.
   - **3b (Completed):** Component-level decomposition identified the final-layer MLP as the dominant causal transformation.
   - **3c (Completed):** Cross-condition comparison confirmed the late-layer circuit remains active and strengthens under distractor contexts.
5. **Phase 4 — Diagnostic analysis:**
   - **4a (Completed):** Logit lens analysis demonstrated early emergence of the correct answer token (layer ~22) under structured prompting.
   - **4b (Pending):** Attention head visualisation and qualitative routing illustration.

---

# Repository Structure

```text
dataset/
  raw/
    entity_chains.json
    distractors.json
  processed/
    dataset.json
    contrast_examples.json
    noisy_contrast_examples.json

scripts/
  phase_0_sanity/
    prompt_inference_check.py
  phase_1_dataset/
    build_dataset.py
  phase_2_behaviour/
    run_evaluation.py
    run_noisy_contrasts.py
  phase_3a_layer_patching/
    activation_patching.py
  phase_3b_component_patching/
    component_patching.py
    head_patching.py
  phase_3c_cross_condition/
    cross_condition_patching.py
  phase_4_logit_lens/
    logit_lens_analysis.py
  utils/
    verify_env.py

results/
  phase_1_dataset/
    dataset_alignment_report.csv
  phase_2_behaviour/
    evaluation_results.csv
    accuracy_summary.csv
  phase_3a_layer_patching/
    layer_patch_results.csv
    layer_patch_summary.csv
  phase_3b_component_patching/
    component_patch_results.csv
    component_patch_summary.csv
    head_patch_results.csv
    head_patch_summary.csv
  phase_3c_cross_condition/
    noisy_layer_patch_results.csv
    noisy_layer_patch_summary.csv
    cross_condition_layer_comparison.csv
  phase_4_logit_lens/
    logit_lens_per_example_clean.csv
    logit_lens_summary_clean.csv
    logit_lens_per_example_noisy.csv
    logit_lens_summary_noisy.csv

figures/
  phase_3a_layer_patching/
    layer_patch_curve.png
  phase_3b_component_patching/
    component_patch_heatmap.png
    head_patch_heatmap.png
  phase_3c_cross_condition/
    clean_vs_noisy_layer_patch_overlay.png
  phase_4_logit_lens/
    logit_lens_logit_clean.png
    logit_lens_top1_clean.png
    logit_lens_logit_noisy.png
    logit_lens_top1_noisy.png

setup-env/
  environment.yml
  setup.ps1

README.md
.gitignore
```

---

# Environment Setup

This project uses a Conda environment and a PowerShell setup script.

### One-step setup

Run:

```powershell
.\setup-env\setup.ps1
```

This will create or update the Conda environment (`enhancing-reasoning-mi`), activate it, and run the verification script to confirm model loading and CUDA availability.

### Verification

The setup script automatically runs:

```powershell
python scripts/utils/verify_env.py
```

This checks package imports (PyTorch, TransformerLens), Pythia-2.8B model loading, CUDA availability, and project directories. A successful run will output `Environment setup complete and verified.`

### Manual activation

If returning to the project later, activate the environment manually:

```powershell
conda activate enhancing-reasoning-mi
```

---

# Running the Experiment

All scripts assume the current working directory is the repository root.

## Phase 0 — Model Sanity Check
Verifies model loading, generation, and activation caching functionality.
```powershell
python scripts/phase_0_sanity/prompt_inference_check.py
```

## Phase 1 — Dataset Construction
Builds the synthetic dataset and strictly verifies token alignment across prompt variants.
```powershell
python scripts/phase_1_dataset/build_dataset.py
```

## Phase 2 — Behavioural Evaluation
Runs all prompt cells, computes exact-match accuracy, and identifies contrast examples.
```powershell
python scripts/phase_2_behaviour/run_evaluation.py
python scripts/phase_2_behaviour/run_noisy_contrasts.py
```

## Phase 3a — Layer-Level Activation Patching
Computes causal mediation curves across transformer layers using clean contrast examples.
```powershell
python scripts/phase_3a_layer_patching/activation_patching.py
```

## Phase 3b — Component and Head Patching
Decomposes layer mediation into attention and MLP components, then evaluates individual attention heads at peak layers.
```powershell
python scripts/phase_3b_component_patching/component_patching.py --layers 24 25 29 30 31
python scripts/phase_3b_component_patching/head_patching.py --layers 30 31
```

## Phase 3c — Cross-Condition Comparison
Compares clean and noisy reasoning circuits using noisy contrast examples.
```powershell
python scripts/phase_3c_cross_condition/cross_condition_patching.py
```

## Phase 4a — Logit Lens Diagnostic
Examines the emergence layer of the correct answer representation across conditions.
```powershell
python scripts/phase_4_logit_lens/logit_lens_analysis.py --include-noisy
```

---

# Reproducibility Notes

- All experiments use the base model `EleutherAI/pythia-2.8b`. The base (non-instruction-tuned) model is strictly required to ensure clean interpretability without RLHF/DPO interference.
- Generation is entirely deterministic (`temperature=0`, `do_sample=False`).
- Prompt variants are strictly token-aligned (padded with neutral fillers) using the Pythia tokenizer prior to evaluation. Activation patching mathematically requires dimensional alignment of token positions.
- Activation patching runs use the structured prompt as the cached activation source and the direct prompt as the baseline forward pass.
- Reported mediation values (Δℓ) correspond to the change in the gold answer logit when patched activations are injected at the final token position.

---

# License

For academic research use.