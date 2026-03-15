#!/usr/bin/env python3
"""
build_dataset.py — Construct token-aligned prompt variants for activation patching.

Generates 5 prompt cells (A-E) per example with identical token counts.
Token alignment is MANDATORY for activation patching: the default mode
requires the Pythia-2.8B tokeniser and will fail hard if it is unavailable.

Use --draft-only to save unaligned prompts for inspection (NOT for patching).

Cells:
  A — Direct Clean   (few-shot, direct answers, supporting facts only)
  B — Direct Noisy   (few-shot, direct answers, + 3 distractors)
  C — Structured Clean (few-shot, Step 1/Step 2 reasoning, facts only)
  D — Structured Noisy (few-shot, Step 1/Step 2 reasoning, + 3 distractors)
  E — Filler Control  (length-matched neutral EOS padding, no reasoning cues)

Design decisions:
  - Cells B and D use the SAME 3 distractors per example (shared random draw).
  - Noisy cells use monotonic fact numbering (Fact 1..Fact 5).
  - Cell E uses EOS-token padding instead of semantic filler text, ensuring
    it adds length but zero reasoning signal.
  - All padding (alignment + filler) uses the model's EOS token, which
    carries minimal semantic content for a base language model (low-signal
    padding, not a claim of perfect neutrality).

Storage format:
  Each cell is stored as a dict with:
    - "prompt": clean human-readable prompt text (no EOS padding visible)
    - "prefix_eos_pad": int — number of EOS tokens to prepend at runtime
    - "inline_eos_filler": int — (Cell E only) number of EOS tokens to insert
      before the final "Answer:" suffix at runtime
  Use materialise_prompt(cell_dict, tokenizer) to reconstruct the exact
  runnable model input from this schema.

Usage:
  python scripts/phase_1_dataset/build_dataset.py               # Build + align (REQUIRED)
  python scripts/phase_1_dataset/build_dataset.py --draft-only   # Build unaligned draft
  python scripts/phase_1_dataset/build_dataset.py --align-only   # Align existing dataset.json
"""

import json
import os
import random
import csv
import sys
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
RAW_DIR = os.path.join(PROJECT_DIR, "dataset", "raw")
PROCESSED_DIR = os.path.join(PROJECT_DIR, "dataset", "processed")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "phase_1_dataset")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

CHAINS_PATH = os.path.join(RAW_DIR, "entity_chains.json")
DISTRACTORS_PATH = os.path.join(RAW_DIR, "distractors.json")
OUTPUT_PATH = os.path.join(PROCESSED_DIR, "dataset.json")
DRAFT_PATH = os.path.join(PROCESSED_DIR, "dataset_draft.json")
REPORT_PATH = os.path.join(RESULTS_DIR, "dataset_alignment_report.csv")

# Maximum EOS-padding tokens. Cell D (structured + noisy) is naturally
# ~150 tokens longer than Cell A (direct + clean). EOS tokens prepended
# at the start carry minimal semantic content for a base model.
MAX_PAD_TOKENS = 250

# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

def load_tokenizer():
    """Load the Pythia-2.8B tokeniser. Raises if unavailable."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-2.8b")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


# ---------------------------------------------------------------------------
# Prompt materialisation (shared utility)
# ---------------------------------------------------------------------------

def materialise_prompt(cell_dict, tokenizer) -> str:
    """
    Reconstruct the exact runnable model input from the stored cell schema.

    Supports both the new schema (dict with 'prompt', 'prefix_eos_pad',
    optional 'inline_eos_filler') and legacy format (plain string).

    For Cell E, inline filler is inserted before the final '\\nAnswer:'
    suffix in the clean prompt text. This exactly reproduces the original
    aligned prompt that was previously stored with literal EOS tokens.

    Parameters
    ----------
    cell_dict : dict or str
        If dict: must have 'prompt' (str) and 'prefix_eos_pad' (int).
                 May have 'inline_eos_filler' (int, Cell E only).
        If str: returned as-is (legacy compatibility).
    tokenizer : transformers.PreTrainedTokenizer
        Used to obtain the EOS token string.

    Returns
    -------
    str : The exact prompt string to feed to the model.
    """
    # Legacy support: if cell is already a plain string, return as-is
    if isinstance(cell_dict, str):
        return cell_dict

    eos = tokenizer.eos_token
    prompt = cell_dict["prompt"]
    prefix_pad = cell_dict.get("prefix_eos_pad", 0)
    inline_filler = cell_dict.get("inline_eos_filler", 0)

    # Insert inline EOS filler for Cell E (before the final "\nAnswer:" suffix)
    if inline_filler > 0:
        marker = "\nAnswer:"
        idx = prompt.rfind(marker)
        if idx != -1:
            prompt = prompt[:idx] + (eos * inline_filler) + prompt[idx:]
        else:
            # Fallback: append filler at the end
            prompt = prompt + (eos * inline_filler)

    # Prepend EOS alignment padding
    if prefix_pad > 0:
        prompt = (eos * prefix_pad) + prompt

    return prompt


# ---------------------------------------------------------------------------
# Few-shot demonstrations
# ---------------------------------------------------------------------------

# DIRECT style (Cells A, B)
DIRECT_DEMO_1 = (
    "Fact 1: The Danube River flows through Vienna.\n"
    "Fact 2: Vienna is the capital of Austria.\n\n"
    "Q: The Danube River flows through the capital of what country?\n"
    "A: Austria"
)

DIRECT_DEMO_2 = (
    "Fact 1: Insulin is produced by the pancreas.\n"
    "Fact 2: The pancreas is located in the abdomen.\n\n"
    "Q: Insulin is produced by an organ located in what part of the body?\n"
    "A: the abdomen"
)

# STRUCTURED style (Cells C, D, E)
STRUCTURED_DEMO_1 = (
    "Fact 1: The Danube River flows through Vienna.\n"
    "Fact 2: Vienna is the capital of Austria.\n\n"
    "Q: The Danube River flows through the capital of what country?\n"
    "Step 1: The Danube River flows through Vienna.\n"
    "Step 2: Vienna is the capital of Austria.\n"
    "Answer: Austria"
)

STRUCTURED_DEMO_2 = (
    "Fact 1: Insulin is produced by the pancreas.\n"
    "Fact 2: The pancreas is located in the abdomen.\n\n"
    "Q: Insulin is produced by an organ located in what part of the body?\n"
    "Step 1: Insulin is produced by the pancreas.\n"
    "Step 2: The pancreas is located in the abdomen.\n"
    "Answer: the abdomen"
)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_cell_A(example: dict) -> str:
    """Cell A — Direct Clean."""
    test = (
        f"Fact 1: {example['fact_1']}\n"
        f"Fact 2: {example['fact_2']}\n\n"
        f"Q: {example['question']}\n"
        f"A:"
    )
    return f"{DIRECT_DEMO_1}\n\n{DIRECT_DEMO_2}\n\n{test}"


def build_cell_B(example: dict, distractors: List[str]) -> str:
    """Cell B — Direct Noisy.

    Uses the SAME 3 distractors as Cell D (passed in, not re-sampled).
    Fact numbering is monotonic: 1-5, with supporting facts at positions
    1 and 4, distractors at 2, 3, 5.
    """
    test = (
        f"Fact 1: {example['fact_1']}\n"
        f"Fact 2: {distractors[0]}\n"
        f"Fact 3: {distractors[1]}\n"
        f"Fact 4: {example['fact_2']}\n"
        f"Fact 5: {distractors[2]}\n\n"
        f"Q: {example['question']}\n"
        f"A:"
    )
    return f"{DIRECT_DEMO_1}\n\n{DIRECT_DEMO_2}\n\n{test}"


def build_cell_C(example: dict) -> str:
    """Cell C — Structured Clean."""
    test = (
        f"Fact 1: {example['fact_1']}\n"
        f"Fact 2: {example['fact_2']}\n\n"
        f"Q: {example['question']}\n"
        f"Step 1: {example['fact_1']}\n"
        f"Step 2: {example['fact_2']}\n"
        f"Answer:"
    )
    return f"{STRUCTURED_DEMO_1}\n\n{STRUCTURED_DEMO_2}\n\n{test}"


def build_cell_D(example: dict, distractors: List[str]) -> str:
    """Cell D — Structured Noisy.

    Uses the SAME 3 distractors as Cell B. Monotonic fact numbering.
    """
    test = (
        f"Fact 1: {example['fact_1']}\n"
        f"Fact 2: {distractors[0]}\n"
        f"Fact 3: {distractors[1]}\n"
        f"Fact 4: {example['fact_2']}\n"
        f"Fact 5: {distractors[2]}\n\n"
        f"Q: {example['question']}\n"
        f"Step 1: {example['fact_1']}\n"
        f"Step 2: {example['fact_2']}\n"
        f"Answer:"
    )
    return f"{STRUCTURED_DEMO_1}\n\n{STRUCTURED_DEMO_2}\n\n{test}"


def build_cell_E_clean(example: dict) -> str:
    """Cell E — Filler Control (clean prompt text WITHOUT inline EOS filler).

    Uses the STRUCTURED few-shot demos (to match Cell C length context)
    but does NOT include Step 1/Step 2 reasoning lines. The gap between
    this prompt and Cell C is filled at runtime using inline_eos_filler
    metadata.

    The clean prompt text ends with "\\nAnswer:" — the inline filler
    will be inserted before this suffix during materialisation.
    """
    test_base = (
        f"Fact 1: {example['fact_1']}\n"
        f"Fact 2: {example['fact_2']}\n\n"
        f"Q: {example['question']}\n"
    )
    test_suffix = "Answer:"
    return f"{STRUCTURED_DEMO_1}\n\n{STRUCTURED_DEMO_2}\n\n{test_base}{test_suffix}"


def compute_cell_E_filler(example: dict, tokenizer) -> int:
    """
    Compute how many EOS tokens Cell E needs as inline filler to match
    Cell C's token count (before cross-cell alignment padding).

    Returns the number of inline EOS tokens needed (may be 0).
    """
    eos = tokenizer.eos_token
    cell_c_text = build_cell_C(example)
    cell_c_tokens = count_tokens(tokenizer, cell_c_text)

    cell_e_clean = build_cell_E_clean(example)
    cell_e_tokens = count_tokens(tokenizer, cell_e_clean)

    filler_needed = cell_c_tokens - cell_e_tokens
    if filler_needed <= 0:
        return 0

    # Iteratively find exact filler count.
    # We reconstruct the full string with filler to check the actual token
    # count, since tokeniser boundary effects may cause slight deviations.
    test_base = (
        f"Fact 1: {example['fact_1']}\n"
        f"Fact 2: {example['fact_2']}\n\n"
        f"Q: {example['question']}\n"
    )
    test_suffix = "Answer:"

    current_n = filler_needed
    for _ in range(30):
        candidate = (
            f"{STRUCTURED_DEMO_1}\n\n{STRUCTURED_DEMO_2}\n\n"
            f"{test_base}{eos * current_n}\n{test_suffix}"
        )
        actual = count_tokens(tokenizer, candidate)
        if actual == cell_c_tokens:
            return current_n
        diff = cell_c_tokens - actual
        current_n += diff
        if current_n < 0:
            current_n = 0

    return max(0, current_n)


# ---------------------------------------------------------------------------
# Token alignment
# ---------------------------------------------------------------------------

def align_cells(
    cells: Dict[str, dict],
    tokenizer,
    max_pad: int = MAX_PAD_TOKENS,
) -> Optional[Dict[str, dict]]:
    """Compute prefix EOS padding needed to align all cells to identical token count.

    GPT-NeoX tokeniser encodes '<|endoftext|>' as token ID 0 (single token).
    Prepending N copies adds exactly N tokens.

    Updates the 'prefix_eos_pad' field in each cell dict.
    Returns aligned cells dict, or None if alignment fails.
    """
    eos = tokenizer.eos_token

    # Materialise each cell to get its current token count (with inline filler
    # for Cell E, but before prefix padding). Reset prefix_eos_pad to 0 first
    # to get the base count.
    base_cells = {}
    counts = {}
    for key, cell in cells.items():
        base = dict(cell)
        base["prefix_eos_pad"] = 0  # reset so we measure the base
        base_cells[key] = base
        materialised = materialise_prompt(base, tokenizer)
        counts[key] = count_tokens(tokenizer, materialised)

    target = max(counts.values())

    if target - min(counts.values()) > max_pad:
        return None

    aligned = {}
    for key, cell in base_cells.items():
        gap = target - counts[key]
        cell_copy = dict(cell)

        if gap == 0:
            cell_copy["prefix_eos_pad"] = 0
            aligned[key] = cell_copy
            continue

        # Compute prefix pad needed
        candidate_pad = gap
        base_prompt = materialise_prompt(cell, tokenizer)  # with inline filler, no prefix

        for _ in range(50):
            candidate_text = (eos * candidate_pad) + base_prompt
            actual = count_tokens(tokenizer, candidate_text)
            if actual == target:
                break
            diff = target - actual
            candidate_pad += diff
            if candidate_pad < 0:
                candidate_pad = 0
        else:
            if count_tokens(tokenizer, (eos * candidate_pad) + base_prompt) != target:
                return None

        final_text = (eos * candidate_pad) + base_prompt
        if count_tokens(tokenizer, final_text) != target:
            return None

        cell_copy["prefix_eos_pad"] = candidate_pad
        aligned[key] = cell_copy

    # Final verification: all cells must produce the same token count
    final_counts = {}
    for key, cell in aligned.items():
        materialised = materialise_prompt(cell, tokenizer)
        final_counts[key] = count_tokens(tokenizer, materialised)

    if len(set(final_counts.values())) != 1:
        return None

    return aligned


# ---------------------------------------------------------------------------
# Cross-domain distractor pool
# ---------------------------------------------------------------------------

def get_cross_domain_pool(domain: str, all_distractors: dict) -> List[str]:
    """Pool distractors from OTHER domains to avoid answer leakage."""
    pool = []
    for d, facts in all_distractors.items():
        if d != domain:
            pool.extend(facts)
    return pool


def sample_safe_distractors(
    pool: List[str],
    answer: str,
    bridge_entity: str,
    n: int = 3,
    max_attempts: int = 100,
) -> List[str]:
    """Sample n distractors that do NOT contain the answer or bridge entity.

    Uses case-insensitive substring matching. If a safe set cannot be found
    within max_attempts random draws, raises ValueError.
    """
    answer_lower = answer.lower().strip()
    bridge_lower = bridge_entity.lower().strip()

    safe_pool = []
    for fact in pool:
        fl = fact.lower()
        if len(answer_lower) > 2 and answer_lower in fl:
            continue
        if len(bridge_lower) > 2 and bridge_lower in fl:
            continue
        safe_pool.append(fact)

    if len(safe_pool) < n:
        raise ValueError(
            f"Only {len(safe_pool)} safe distractors available "
            f"(need {n}) after filtering for answer='{answer}', "
            f"bridge='{bridge_entity}'."
        )

    return random.sample(safe_pool, n)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_dataset(tokenizer=None) -> List[dict]:
    """Build all prompt cells for every example.

    If tokenizer is provided, Cell E uses it for precise EOS filler computation.
    Distractors are sampled ONCE per example and shared between B and D.
    """
    print("Loading entity chains...")
    with open(CHAINS_PATH) as f:
        chains = json.load(f)
    print(f"  Loaded {len(chains)} chains.")

    print("Loading distractors...")
    with open(DISTRACTORS_PATH) as f:
        all_distractors = json.load(f)
    for k, v in all_distractors.items():
        print(f"  {k}: {len(v)} distractors")

    dataset = []
    for ex in chains:
        pool = get_cross_domain_pool(ex["domain"], all_distractors)

        shared_distractors = sample_safe_distractors(
            pool, ex["answer"], ex["bridge_entity"], n=3,
        )

        if tokenizer is not None:
            inline_filler = compute_cell_E_filler(ex, tokenizer)
        else:
            inline_filler = 30

        cells = {
            "A": {"prompt": build_cell_A(ex), "prefix_eos_pad": 0},
            "B": {"prompt": build_cell_B(ex, shared_distractors), "prefix_eos_pad": 0},
            "C": {"prompt": build_cell_C(ex), "prefix_eos_pad": 0},
            "D": {"prompt": build_cell_D(ex, shared_distractors), "prefix_eos_pad": 0},
            "E": {"prompt": build_cell_E_clean(ex), "prefix_eos_pad": 0,
                  "inline_eos_filler": inline_filler},
        }
        dataset.append({
            "id": ex["id"],
            "domain": ex["domain"],
            "answer": ex["answer"],
            "bridge_entity": ex["bridge_entity"],
            "question": ex["question"],
            "fact_1": ex["fact_1"],
            "fact_2": ex["fact_2"],
            "distractors": shared_distractors,
            "aligned": False,
            "token_count": None,
            "cells": cells,
        })
    return dataset


def perform_alignment(dataset: List[dict], tokenizer) -> List[dict]:
    """Token-align all examples. Drop failures. Save report."""
    aligned_out = []
    dropped = 0
    report_rows = []

    for entry in dataset:
        cells = entry["cells"]

        counts_raw = {}
        for k, cell in cells.items():
            materialised = materialise_prompt(cell, tokenizer)
            counts_raw[k] = count_tokens(tokenizer, materialised)

        result = align_cells(cells, tokenizer)
        if result is None:
            dropped += 1
            report_rows.append({
                "example_id": entry["id"],
                **{f"token_count_{k}": counts_raw[k] for k in "ABCDE"},
                "aligned": False,
            })
            continue

        counts_final = {}
        for k, cell in result.items():
            materialised = materialise_prompt(cell, tokenizer)
            counts_final[k] = count_tokens(tokenizer, materialised)
        tok_count = list(counts_final.values())[0]

        entry["cells"] = result
        entry["aligned"] = True
        entry["token_count"] = tok_count
        aligned_out.append(entry)

        report_rows.append({
            "example_id": entry["id"],
            **{f"token_count_{k}": counts_final[k] for k in "ABCDE"},
            "aligned": True,
        })

    fieldnames = ["example_id"] + [f"token_count_{k}" for k in "ABCDE"] + ["aligned"]
    with open(REPORT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    print(f"  Report: {REPORT_PATH}")
    print(f"  Aligned: {len(aligned_out)}/{len(dataset)}, Dropped: {dropped}")
    return aligned_out


def main():
    random.seed(SEED)

    draft_only = "--draft-only" in sys.argv
    align_only = "--align-only" in sys.argv

    if draft_only and align_only:
        print("ERROR: Cannot use --draft-only and --align-only together.")
        sys.exit(1)

    if draft_only:
        print("=== DRAFT MODE (no alignment, not for patching) ===\n")
        dataset = build_dataset(tokenizer=None)

        with open(DRAFT_PATH, "w") as f:
            json.dump(dataset, f, indent=2)
        print(f"\nSaved {len(dataset)} UNALIGNED examples to {DRAFT_PATH}")
        print("WARNING: This is a draft. Run without --draft-only to produce")
        print("         the aligned dataset required for activation patching.")
        return

    print("Loading Pythia-2.8B tokeniser...")
    try:
        tokenizer = load_tokenizer()
        print("  => Tokeniser loaded successfully.")
    except Exception as e:
        print(f"\nFATAL: Cannot load tokeniser ({type(e).__name__}: {e}).")
        print("Token alignment is mandatory. Install `transformers`:")
        print("  pip install transformers")
        print("\nOr use --draft-only for unaligned inspection drafts.")
        sys.exit(1)

    if align_only:
        print(f"\nLoading existing dataset from {OUTPUT_PATH}...")
        with open(OUTPUT_PATH) as f:
            dataset = json.load(f)
        print(f"  Loaded {len(dataset)} examples.")
    else:
        print()
        dataset = build_dataset(tokenizer=tokenizer)

    total_before = len(dataset)
    print("\nPerforming token alignment...")
    dataset = perform_alignment(dataset, tokenizer)
    total_after = len(dataset)
    total_dropped = total_before - total_after

    if total_after == 0:
        print("\nFATAL: Zero examples survived alignment. Check prompts.")
        sys.exit(1)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(dataset, f, indent=2)

    toks = [d["token_count"] for d in dataset]
    success_rate = total_after / total_before * 100 if total_before > 0 else 0

    from collections import Counter
    by_domain = Counter(d["domain"] for d in dataset)

    print(f"\n{'='*60}")
    print(f"  DATASET BUILD COMPLETE")
    print(f"{'='*60}")
    print(f"  Chains loaded:           {total_before}")
    print(f"  Examples kept (aligned):  {total_after}")
    print(f"  Examples dropped:         {total_dropped}")
    print(f"  Alignment success rate:   {success_rate:.1f}%")
    print(f"  Token count range:        {min(toks)} - {max(toks)}")
    print(f"  Domains:                  {dict(by_domain)}")
    print(f"  Output:                   {OUTPUT_PATH}")
    print(f"  Report:                   {REPORT_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()