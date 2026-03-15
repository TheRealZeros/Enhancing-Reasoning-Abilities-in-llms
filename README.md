# Enhancing Reasoning Abilities of LLMs

Experimental code for the honours thesis:

**Enhancing Reasoning Abilities of Large Language Models**

Bachelor of Engineering (Software Engineering)  
Macquarie University  
Supervisor: Usman Naseem

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

---

# Repository Structure

```
dataset/
    entity_chains.json          — raw A→B→C entity chains
    distractors.json            — domain-specific distractor facts
    dataset.json                — final dataset with all 5 cells per example

scripts/
    load_model_test.py          — Phase 0: model sanity check
    build_dataset.py            — Phase 1: dataset construction
    run_evaluation.py           — Phase 2: behavioural evaluation
    run_noisy_contrasts.py      — identify noisy contrast examples (B wrong ∧ D correct)
    activation_patching.py      — Phase 3a: layer-level patching
    component_patching.py       — Phase 3b step 1: attention vs MLP decomposition
    head_patching.py            — Phase 3b step 2: individual head patching
    cross_condition_patching.py — Phase 3c: clean vs noisy comparison
    logit_lens_analysis.py      — Phase 4a: logit lens diagnostic

results/
    evaluation_results.csv      — one row per (example, cell)
    accuracy_summary.csv        — per-cell accuracy table
    contrast_examples.json      — clean contrasts (A wrong ∧ C correct)
    noisy_contrast_examples.json — noisy contrasts (B wrong ∧ D correct)
    layer_patch_results.csv     — layer-level patching per (example, layer)
    layer_patch_summary.csv     — per-layer aggregated Δℓ
    component_patch_results.csv — component-level per (example, layer, component)
    component_patch_summary.csv — per-(layer, component) aggregated Δℓ,c
    head_patch_results.csv      — head-level per (example, layer, head)
    head_patch_summary.csv      — per-(layer, head) aggregated Δℓ,h
    noisy_layer_patch_results.csv
    noisy_layer_patch_summary.csv
    cross_condition_layer_comparison.csv

figures/
    layer_patch_curve.png
    component_patch_heatmap.png
    head_patch_heatmap.png
    clean_vs_noisy_layer_patch_overlay.png
    logit_lens_top1_clean.png
    logit_lens_top1_noisy.png
    logit_lens_logit_clean.png
    logit_lens_logit_noisy.png

setup-env/
    setup-scripts/
        verify_env.py
    environment.yml
    setup.ps1
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
python setup-env/setup-scripts/verify_env.py
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

### Phase 0 — Model Sanity Check

```
python scripts/load_model_test.py
```

Verifies model loading, few-shot generation, and activation caching.

---

### Phase 1 — Generate Dataset

```
python scripts/build_dataset.py
```

Outputs:

```
dataset/entity_chains.json
dataset/distractors.json
dataset/dataset.json
```

---

### Phase 2 — Behavioural Evaluation

```
python scripts/run_evaluation.py \
    --dataset dataset/dataset.json \
    --output-dir results \
    --model EleutherAI/pythia-2.8b
```

Outputs:

```
results/evaluation_results.csv
results/accuracy_summary.csv
results/contrast_examples.json
```

---

### Phase 3a — Layer-Level Activation Patching

```
python scripts/activation_patching.py \
    --contrast-file results/contrast_examples.json \
    --output-dir results \
    --model EleutherAI/pythia-2.8b
```

Outputs:

```
results/layer_patch_results.csv
results/layer_patch_summary.csv
figures/layer_patch_curve.png
```

---

### Phase 3b — Component-Level and Head-Level Patching

```
python scripts/component_patching.py \
    --contrast-file results/contrast_examples.json \
    --output-dir results \
    --layers 24 25 29 30 31

python scripts/head_patching.py \
    --contrast-file results/contrast_examples.json \
    --output-dir results \
    --layers 30 31
```

Outputs:

```
results/component_patch_results.csv
results/component_patch_summary.csv
results/head_patch_results.csv
results/head_patch_summary.csv
figures/component_patch_heatmap.png
figures/head_patch_heatmap.png
```

---

### Phase 3c — Cross-Condition Comparison (Clean vs Noisy)

```
python scripts/cross_condition_patching.py \
    --dataset dataset/dataset.json \
    --eval-results results/evaluation_results.csv \
    --clean-summary results/layer_patch_summary.csv \
    --output-dir results
```

Outputs:

```
results/noisy_contrast_examples.json
results/noisy_layer_patch_results.csv
results/noisy_layer_patch_summary.csv
results/cross_condition_layer_comparison.csv
figures/clean_vs_noisy_layer_patch_overlay.png
```

---

### Phase 4 — Logit Lens Diagnostic

```
python scripts/logit_lens_analysis.py --include-noisy
```

Outputs:

```
figures/logit_lens_top1_clean.png
figures/logit_lens_top1_noisy.png
figures/logit_lens_logit_clean.png
figures/logit_lens_logit_noisy.png
```

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

# License

For academic research use.
