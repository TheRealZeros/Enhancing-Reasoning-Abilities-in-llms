# Enhancing Reasoning Abilities of Large Language Models

Experimental code for the honours thesis:

**Enhancing Reasoning Abilities of Large Language Models**

Bachelor of Engineering (Software Engineering)  
Macquarie University  

---

# Project Overview

This project investigates whether structured prompting changes the internal mechanisms used by transformer models when performing multi-hop reasoning. 

The experiment evaluates the Pythia-2.8B model on a synthetic two-hop
reasoning dataset (A→B→C fact-composition chains) under five prompting
conditions and analyses internal model activations using mechanistic
interpretability techniques.

Primary analysis method:
- Activation patching (causal) using TransformerLens:
  - layer-level mediation (Δℓ curves)
  - component-level decomposition (attention vs MLP)
  - attention head-level attribution

Secondary diagnostics:
- Logit lens analysis (emergence layer comparison)
- Attention pattern visualisation (final-token routing behaviour)

---

# Quick Start

From the repository root:

```bash
# Setup environment
.\setup\setup.ps1

# Run full experiment pipeline
python scripts/phase_0_sanity/prompt_inference_check.py
python scripts/phase_1_dataset/build_dataset.py
python scripts/phase_2_behaviour/run_evaluation.py
python scripts/phase_2_behaviour/run_noisy_contrasts.py
python scripts/phase_3a_layer_patching/activation_patching.py
python scripts/phase_3b_component_patching/component_patching.py --layers 24 25 29 30 31
python scripts/phase_3b_component_patching/head_patching.py --layers 30 31
python scripts/phase_3c_cross_condition/cross_condition_patching.py
python scripts/phase_4_logit_lens/logit_lens_analysis.py --include-noisy

---

# Current Status

The full experimental pipeline has been implemented and executed.

Completed:

- Phase 0: Model sanity check
- Phase 1: Dataset construction (200 aligned examples)
- Phase 2: Behavioural evaluation
- Phase 3a: Layer-level activation patching
- Phase 3b: Component and head-level patching
- Phase 3c: Cross-condition comparison (clean vs noisy)
- Phase 4a: Logit lens analysis
- Phase 4b: Attention visualisation (initial run complete, refinement ongoing)

---

# Key Results (High-Level)

Behavioural:

- Structured prompting improves multi-hop reasoning accuracy:
  - Cell C > Cell A
  - Cell D > Cell B
- Noisy structured condition (Cell D) performs strongest overall
- Filler control (Cell E) ≈ 0%, confirming length alone does not explain improvements

Mechanistic:

- Late layers (24–31) carry the majority of causal mediation
- Final-layer MLP contributes strongly to correct answer formation
- Attention components contribute but are more distributed

Cross-condition:

- Similar causal layers appear in clean and noisy conditions
- Noisy condition shows stronger late-layer mediation

Logit lens:

- Correct answer becomes linearly decodable earlier under structured prompting
- Structured prompts shift representation formation forward across layers

Attention (Phase 4b):

- Direct prompting shows narrow, collapsed attention near answer tokens
- Structured prompting shows broader attention across relevant context
- Late layers (30–31) show clearer routing differences between conditions

---

# Experiment Design

The dataset consists of 200 synthetic two-hop reasoning examples across four
domains (geography, science, biology, culture).

Each example follows:

A → B → C

The model must compute:

r2(r1(A)) = C

---

## Prompt Conditions

| Cell | Prompt Type | Context |
|------|-------------|---------|
| A | Direct (few-shot, direct answers) | Clean |
| B | Direct | Noisy (+ 3 distractors) |
| C | Structured (Step 1 / Step 2) | Clean |
| D | Structured | Noisy (+ 3 distractors) |
| E | Filler control | Clean |

All prompt variants are strictly token-aligned.

---

## Contrast Examples

Defined as:

Cell A incorrect AND Cell C correct

Used for all activation-level analysis.

---

# Experiment Pipeline

1. Phase 0 — Sanity check  
2. Phase 1 — Dataset construction  
3. Phase 2 — Behavioural evaluation  
4. Phase 3 — Activation patching  
   - 3a: Layer-level  
   - 3b: Component + head-level  
   - 3c: Cross-condition  
5. Phase 4 — Diagnostics  
   - 4a: Logit lens  
   - 4b: Attention visualisation  

---

# Phase 4b – Attention Visualisation

This phase inspects how attention is distributed at the final token.

Method:

- Extract attention from `blocks.{layer}.attn.hook_pattern`
- Slice final token → all source tokens
- Average across heads for visualisation
- Compare Cell A vs Cell C

Layers analysed:

20, 30, 31

Initial findings:

- Structured prompts distribute attention across relevant context
- Direct prompts collapse attention locally
- Differences are strongest in late layers

---

# Repository Structure

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

  phase_4a_logit_lens/
    logit_lens_analysis.py

  phase_4b_attention_visualisation/
    attention_heatmaps.py

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

  phase_4a_logit_lens/
    logit_lens_per_example_clean.csv
    logit_lens_summary_clean.csv
    logit_lens_per_example_noisy.csv
    logit_lens_summary_noisy.csv

  phase_4b_attention_visualisation/
    attention_manifest.json
    geo_006_layer_20_comparison.json
    geo_006_layer_30_comparison.json
    geo_006_layer_31_comparison.json
    geo_029_layer_20_comparison.json
    geo_029_layer_30_comparison.json
    geo_029_layer_31_comparison.json

figures/
  phase_3a_layer_patching/
    layer_patch_curve.png
  phase_3b_component_patching/
    component_patch_heatmap.png
    head_patch_heatmap.png

  phase_4a_logit_lens/
    logit_lens_logit_clean.png
    logit_lens_top1_clean.png
    logit_lens_logit_noisy.png
    logit_lens_top1_noisy.png

  phase_4b_attention_visualisation/
    geo_006_layer_20_comparison.png
    geo_006_layer_30_comparison.png
    geo_006_layer_31_comparison.png
    geo_029_layer_20_comparison.png
    geo_029_layer_30_comparison.png
    geo_029_layer_31_comparison.png

setup/
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

Outputs:

dataset/processed/dataset.json  
results/phase_1_dataset/dataset_alignment_report.csv  

---

## Phase 2 — Behavioural Evaluation
Runs all prompt cells, computes exact-match accuracy, and identifies contrast examples.
```powershell
python scripts/phase_2_behaviour/run_evaluation.py

Outputs:

results/phase_2_behaviour/evaluation_results.csv  
results/phase_2_behaviour/accuracy_summary.csv  
dataset/processed/contrast_examples.json  

Identify noisy contrast examples:

python scripts/phase_2_behaviour/run_noisy_contrasts.py

Outputs:

dataset/processed/noisy_contrast_examples.json  

---

## Phase 3a — Layer-Level Activation Patching
Computes causal mediation curves across transformer layers using clean contrast examples.
```powershell
python scripts/phase_3a_layer_patching/activation_patching.py

Outputs:

results/phase_3a_layer_patching/layer_patch_results.csv  
results/phase_3a_layer_patching/layer_patch_summary.csv  
figures/phase_3a_layer_patching/layer_patch_curve.png  

---

## Phase 3b — Component and Head Patching

Decomposes layer mediation into attention and MLP components,  
then evaluates individual attention heads.

python scripts/phase_3b_component_patching/component_patching.py --layers 24 25 29 30 31  

python scripts/phase_3b_component_patching/head_patching.py --layers 30 31  

Outputs:

results/phase_3b_component_patching/component_patch_results.csv  
results/phase_3b_component_patching/component_patch_summary.csv  
results/phase_3b_component_patching/head_patch_results.csv  
results/phase_3b_component_patching/head_patch_summary.csv  

figures/phase_3b_component_patching/component_patch_heatmap.png  
figures/phase_3b_component_patching/head_patch_heatmap.png  

---

## Phase 3c — Cross-Condition Comparison

Compares clean and noisy reasoning circuits.

python scripts/phase_3c_cross_condition/cross_condition_patching.py  

Outputs:

results/phase_3a_layer_patching/noisy_layer_patch_results.csv  
results/phase_3a_layer_patching/noisy_layer_patch_summary.csv  
results/phase_3c_cross_condition/cross_condition_layer_comparison.csv  

figures/phase_3a_layer_patching/clean_vs_noisy_layer_patch_overlay.png  

---

## Phase 4a — Logit Lens Diagnostic

Examines the emergence layer of the correct answer representation.

python scripts/phase_4a_logit_lens/logit_lens_analysis.py --include-noisy  

Outputs:

results/phase_4a_logit_lens/logit_lens_per_example_clean.csv  
results/phase_4a_logit_lens/logit_lens_per_example_noisy.csv  
results/phase_4a_logit_lens/logit_lens_summary_clean.csv  
results/phase_4a_logit_lens/logit_lens_summary_noisy.csv  

figures/phase_4a_logit_lens/logit_lens_logit_clean.png  
figures/phase_4a_logit_lens/logit_lens_logit_noisy.png  
figures/phase_4a_logit_lens/logit_lens_top1_clean.png  
figures/phase_4a_logit_lens/logit_lens_top1_noisy.png  

---

## Phase 4b — Attention Visualisation

Visualises final-token attention patterns to analyse information routing behaviour.

python scripts/phase_4b_attention_visualisation/attention_heatmaps.py --num-examples 2 --layers 20 30 31  

Outputs:

figures/phase_4b_attention_visualisation/{example_id}_layer_{layer}_comparison.png  

results/phase_4b_attention_visualisation/{example_id}_layer_{layer}_comparison.json  
results/phase_4b_attention_visualisation/attention_manifest.json  

Notes:

- compares Cell A vs Cell C attention at the final token  
- uses head-averaged attention for visualisation  
- focuses on layers 20, 30, 31  
- designed for qualitative inspection, not primary causal evidence  

---


# Model

Experiments use:

```
EleutherAI/pythia-2.8b
```

Loaded using TransformerLens `HookedTransformer.from_pretrained`. All generation
is deterministic (`do_sample=False`, `temperature=0`). The model is a base
(non-instruction-tuned) language model and operates via few-shot pattern
completion.

---

# Thesis Context

This repository supports the empirical component of the thesis examining
whether reasoning in large language models emerges from the interaction of:

- memory and contextual reasoning (operationalised as in-context evidence routing; tested via distractor conditions)
- structured reasoning processes (the manipulated variable; implemented as structured few-shot prompting)
- internal representations supporting generalisation (the measurement lens; observed via activation patching, logit lens, and attention analysis)

The experimental design tests whether structured prompting alters the causal
pathways used by transformer models when solving multi-hop reasoning tasks,
and whether contextual distractors shift those pathways.

Methodological precedents: Wang et al. 2022 (IOI circuit), Meng et al. 2022
(ROME/causal tracing), Elhage et al. 2021 (transformer circuits framework).

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