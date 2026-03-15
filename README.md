# Enhancing Reasoning Abilities of Large Language Models

Experimental code for the honours thesis:

**Enhancing Reasoning Abilities of Large Language Models**

Bachelor of Engineering (Software Engineering)  
Macquarie University  

---

# Project Overview

This project investigates whether structured prompting changes the internal
mechanisms used by transformer models when performing multi-hop reasoning.

The experiment evaluates the Pythia-2.8B model on a synthetic two-hop
reasoning dataset (A→B→C fact-composition chains) under five prompting
conditions and analyses internal model activations using mechanistic
interpretability techniques.

Primary analysis method:
- Activation patching (causal) using TransformerLens — layer-level,
  component-level (attention vs MLP), and individual attention head patching

Secondary diagnostics:
- Logit lens analysis (emergence layer comparison across conditions)
- Attention head visualisation (qualitative routing illustration)

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

# Experiment Design

The dataset consists of 200 synthetic two-hop reasoning examples across four
domains (geography, science, biology, culture). Each example contains an
A→B→C entity chain where the question asks for the composition r2(r1(A)),
and the gold answer is entity C.

Each example is evaluated under five prompt conditions (cells):

| Cell | Prompt Type | Context |
|------|-------------|---------|
| A | Direct (few-shot, direct answers) | Clean (2 supporting facts only) |
| B | Direct | Noisy (+ 3 distractor facts) |
| C | Structured (few-shot, Step 1 / Step 2 reasoning) | Clean |
| D | Structured | Noisy (+ 3 distractor facts) |
| E | Filler control (length-matched neutral padding) | Clean |

All five prompt variants for every example have identical token counts,
verified using the Pythia tokeniser. This is a strict prerequisite for
valid activation patching.

**Contrast examples** are defined as cases where Cell A (direct) produces the
wrong answer and Cell C (structured) produces the correct answer. These
examples are used for all downstream activation analysis. Noisy contrasts
(Cell B wrong, Cell D correct) are identified separately for cross-condition
comparison.

---

# Experiment Pipeline

The workflow follows five main phases:

1. **Phase 0 — Model sanity check:** Verify model loads, few-shot prompting
   works, and activation caching functions correctly.
2. **Phase 1 — Synthetic dataset construction:** Build 200 token-aligned
   examples across all five cells with entity chains and distractors.
3. **Phase 2 — Behavioural evaluation:** Run all cells, compute exact-match
   accuracy, and identify contrast examples.
4. **Phase 3 — Activation patching:**
   - 3a: Layer-level residual stream patching (Δℓ curves)
   - 3b: Component-level decomposition (attention vs MLP) at high-effect layers,
     followed by individual attention head patching
   - 3c: Cross-condition comparison (clean vs noisy Δℓ overlay)
5. **Phase 4 — Diagnostic analysis:** Logit lens and attention visualisation
   to support interpretation of activation patching results.

The goal is to identify which transformer components causally contribute to
structured reasoning behaviour and whether contextual distractors shift
that causal pattern.

## Full Pipeline Execution

From a clean repository state, the entire experiment can be run with:

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

# Repository Structure

```
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
    noisy_layer_patch_results.csv
    noisy_layer_patch_summary.csv

  phase_3b_component_patching/
    component_patch_results.csv
    component_patch_summary.csv
    head_patch_results.csv
    head_patch_summary.csv

  phase_3c_cross_condition/
    cross_condition_layer_comparison.csv

  phase_4_logit_lens/
    logit_lens_per_example_clean.csv
    logit_lens_per_example_noisy.csv
    logit_lens_summary_clean.csv
    logit_lens_summary_noisy.csv

figures/
  phase_3a_layer_patching/
    layer_patch_curve.png
    clean_vs_noisy_layer_patch_overlay.png

  phase_3b_component_patching/
    component_patch_heatmap.png
    head_patch_heatmap.png

  phase_4_logit_lens/
    logit_lens_logit_clean.png
    logit_lens_logit_noisy.png
    logit_lens_top1_clean.png
    logit_lens_top1_noisy.png

setup/
  environment.yml
  setup.ps1

README.md
.gitignore
```

---

## Environment Setup

This project uses a Conda environment and a PowerShell setup script.

### One-step setup

Run:

```powershell
.\setup-env\setup.ps1
```

This will:

- create or update the Conda environment
- activate the environment
- run environment verification
- confirm model loading and CUDA availability

### Verification

The setup script runs:

```powershell
python scripts/utils/verify_env.py
```

This checks:

- Python installation
- package imports (PyTorch, TransformerLens, transformers, pandas, matplotlib)
- PyTorch and CUDA availability
- TransformerLens import
- Pythia-2.8B model loading
- required project directories

A successful run ends with:

```
Environment verification passed.
Environment setup complete and verified.
```

### Manual activation

If needed:

```powershell
conda activate enhancing-reasoning-mi
```

---

# Running the Experiment
# Running the Experiment

All scripts assume the current working directory is the repository root.

The entire pipeline can be executed sequentially with default parameters.

---

## Phase 0 — Model Sanity Check

Verifies model loading, generation, and activation caching.

python scripts/phase_0_sanity/prompt_inference_check.py

---

## Phase 1 — Dataset Construction

Builds the synthetic dataset and verifies token alignment.

python scripts/phase_1_dataset/build_dataset.py

Outputs:

dataset/processed/dataset.json
results/phase_1_dataset/dataset_alignment_report.csv

---

## Phase 2 — Behavioural Evaluation

Runs all prompt cells and computes exact-match accuracy.

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

Computes causal mediation curves across transformer layers.

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

## Phase 4 — Logit Lens Diagnostic

Examines the emergence layer of the correct answer representation.

python scripts/phase_4_logit_lens/logit_lens_analysis.py --include-noisy

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

- All experiments use the base model `EleutherAI/pythia-2.8b`.
- Generation is deterministic (`temperature=0`, `do_sample=False`).
- All prompt variants are token-aligned using the Pythia tokenizer.
- Contrast examples are defined as cases where:
  - Direct prompting fails (Cell A incorrect)
  - Structured prompting succeeds (Cell C correct)

Activation patching experiments use the structured run (Cell C) as the
activation source and the direct run (Cell A) as the baseline forward pass.

Reported mediation values (Δℓ) correspond to the change in the gold answer
logit when patched activations are injected at the final token position.

All analysis scripts assume execution from the repository root.

---

# License

For academic research use.
