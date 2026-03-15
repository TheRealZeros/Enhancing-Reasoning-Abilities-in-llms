# Enhancing Reasoning Abilities of LLMs

Experimental code for the honours thesis:

**Enhancing Reasoning Abilities of Large Language Models**

Bachelor of Engineering (Software Engineering)  
Macquarie University

---

# Project Overview

This project investigates whether structured prompting changes the internal
mechanisms used by transformer models when performing multi-hop reasoning.

The experiment evaluates the Pythia-2.8B model on synthetic two-hop reasoning
tasks under multiple prompting conditions and analyses internal model
activations using mechanistic interpretability techniques.

Primary analysis method:
- Activation patching using TransformerLens

Secondary diagnostics:
- Logit lens analysis
- Attention head visualisation

---

# Experiment Pipeline

The workflow follows five main stages:

1. Model sanity check
2. Synthetic dataset generation
3. Behavioural evaluation across prompting conditions
4. Activation patching analysis
5. Diagnostic visualisations

The goal is to identify which transformer components causally contribute to
structured reasoning behaviour.

---

# Repository Structure

```
dataset/
    entity_chains.json
    distractors.json
    dataset.json

scripts/
    load_model_test.py
    build_dataset.py
    run_evaluation.py
    activation_patching.py

results/
    evaluation_results.csv
    accuracy_summary.csv
    contrast_examples.json
    layer_patch_results.csv

figures/
    layer_patch_curve.png

setup-env/
    setup-scripts/
        torch-verify
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
- package imports
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

### 1. Test Model Loading

```
python scripts/load_model_test.py
```

---

### 2. Generate Dataset

```
python scripts/build_dataset.py
```

Output:

```
dataset/dataset.json
```

---

### 3. Run Behavioural Evaluation

```
python scripts/run_evaluation.py
```

Outputs:

```
results/evaluation_results.csv
results/accuracy_summary.csv
```

---

### 4. Run Activation Patching

```
python scripts/activation_patching.py
```

Outputs:

```
results/layer_patch_results.csv
figures/layer_patch_curve.png
```

---

# Model

Experiments use:

```
EleutherAI/pythia-2.8b
```

Loaded using TransformerLens.

---

# Thesis Context

This repository supports the empirical component of the thesis examining
whether reasoning in large language models emerges from the interaction of:

- memory and contextual reasoning
- structured reasoning processes
- internal representations supporting generalisation

The experimental design tests whether structured prompting alters the causal
pathways used by transformer models when solving multi-hop reasoning tasks.

---

# License

For academic research use.