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


def build_cell_E(example: dict, tokenizer, target_tokens: int = None) -> str:
    """Cell E — Filler Control.

    Uses the STRUCTURED few-shot demos (to match Cell C length context)
    but replaces the Step 1/Step 2 reasoning lines with EOS-token padding.
    This ensures Cell E has similar length to C but carries zero reasoning
    signal — the padding is semantically null.

    If tokenizer is available and target_tokens is set, pads to exact count.
    Otherwise, uses a fixed block of EOS tokens as approximate filler.
    """
    # The base of Cell E: same facts, same question, same ending as C,
    # but NO step-by-step reasoning. We'll fill the gap with EOS padding.
    test_base = (
        f"Fact 1: {example['fact_1']}\n"
        f"Fact 2: {example['fact_2']}\n\n"
        f"Q: {example['question']}\n"
    )
    test_suffix = "Answer:"

    if tokenizer is not None:
        # Compute how many EOS tokens we need to fill the gap between
        # this base+suffix and the Cell C equivalent.
        eos = tokenizer.eos_token
        cell_c_text = build_cell_C(example)
        cell_c_tokens = count_tokens(tokenizer, cell_c_text)
        base_text = f"{STRUCTURED_DEMO_1}\n\n{STRUCTURED_DEMO_2}\n\n{test_base}{test_suffix}"
        base_tokens = count_tokens(tokenizer, base_text)
        filler_needed = cell_c_tokens - base_tokens

        if filler_needed > 0:
            filler = eos * filler_needed
            candidate = f"{STRUCTURED_DEMO_1}\n\n{STRUCTURED_DEMO_2}\n\n{test_base}{filler}\n{test_suffix}"
            # Verify and correct
            actual = count_tokens(tokenizer, candidate)
            for _ in range(30):
                if actual == cell_c_tokens:
                    break
                diff = cell_c_tokens - actual
                if diff > 0:
                    filler = filler + eos * diff
                elif diff < 0:
                    filler = filler[:len(eos) * (filler_needed + diff)]
                candidate = f"{STRUCTURED_DEMO_1}\n\n{STRUCTURED_DEMO_2}\n\n{test_base}{filler}\n{test_suffix}"
                actual = count_tokens(tokenizer, candidate)
            return candidate
        else:
            return f"{STRUCTURED_DEMO_1}\n\n{STRUCTURED_DEMO_2}\n\n{test_base}{test_suffix}"
    else:
        # Offline fallback: use a fixed EOS block as approximate filler.
        # This will be re-aligned when the real tokeniser is available.
        eos_approx = "<|endoftext|>" * 30  # rough estimate
        return f"{STRUCTURED_DEMO_1}\n\n{STRUCTURED_DEMO_2}\n\n{test_base}{eos_approx}\n{test_suffix}"


# ---------------------------------------------------------------------------
# Token alignment
# ---------------------------------------------------------------------------

def align_cells(
    cells: Dict[str, str],
    tokenizer,
    max_pad: int = MAX_PAD_TOKENS,
) -> Optional[Dict[str, str]]:
    """Pad all cells to identical token count using EOS-token prefix.

    GPT-NeoX tokeniser encodes '<|endoftext|>' as token ID 0 (single token).
    Prepending N copies adds exactly N tokens.

    Returns aligned cells dict, or None if alignment fails.
    """
    eos = tokenizer.eos_token
    counts = {k: count_tokens(tokenizer, v) for k, v in cells.items()}
    target = max(counts.values())

    if target - min(counts.values()) > max_pad:
        return None

    aligned = {}
    for key, text in cells.items():
        gap = target - counts[key]
        if gap == 0:
            aligned[key] = text
            continue

        # Prepend EOS tokens
        candidate = (eos * gap) + text
        actual = count_tokens(tokenizer, candidate)

        # Iterative correction
        for _ in range(50):
            if actual == target:
                break
            diff = target - actual
            if diff > 0:
                candidate = (eos * diff) + candidate
            elif diff < 0:
                trim = min(-diff, gap)
                candidate = candidate[len(eos) * trim:]
            actual = count_tokens(tokenizer, candidate)
        else:
            if actual != target:
                return None

        if actual != target:
            return None
        aligned[key] = candidate

    # Final verification
    final = {k: count_tokens(tokenizer, v) for k, v in aligned.items()}
    if len(set(final.values())) != 1:
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

    # Pre-filter the pool to only safe candidates
    safe_pool = []
    for fact in pool:
        fl = fact.lower()
        # Check answer substring (skip very short answers to avoid
        # false positives on common words like "one", "green")
        if len(answer_lower) > 2 and answer_lower in fl:
            continue
        # Check bridge entity substring
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

    If tokenizer is provided, Cell E uses it for precise EOS padding.
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

        # Sample distractors ONCE — shared between Cell B and Cell D.
        # Integrity check: distractors must not contain the answer or bridge entity.
        shared_distractors = sample_safe_distractors(
            pool, ex["answer"], ex["bridge_entity"], n=3,
        )

        cells = {
            "A": build_cell_A(ex),
            "B": build_cell_B(ex, shared_distractors),
            "C": build_cell_C(ex),
            "D": build_cell_D(ex, shared_distractors),
            "E": build_cell_E(ex, tokenizer),
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
        counts_raw = {k: count_tokens(tokenizer, v) for k, v in cells.items()}

        result = align_cells(cells, tokenizer)
        if result is None:
            dropped += 1
            report_rows.append({
                "example_id": entry["id"],
                **{f"token_count_{k}": counts_raw[k] for k in "ABCDE"},
                "aligned": False,
            })
            continue

        counts_final = {k: count_tokens(tokenizer, v) for k, v in result.items()}
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

    # Save report
    fieldnames = ["example_id"] + [f"token_count_{k}" for k in "ABCDE"] + ["aligned"]
    with open(REPORT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    print(f"  Report: {REPORT_PATH}")
    print(f"  Aligned: {len(aligned_out)}/{len(dataset)}, Dropped: {dropped}")
    return aligned_out


def main():
    random.seed(SEED)  # Deterministic generation for thesis reproducibility

    draft_only = "--draft-only" in sys.argv
    align_only = "--align-only" in sys.argv

    if draft_only and align_only:
        print("ERROR: Cannot use --draft-only and --align-only together.")
        sys.exit(1)

    # ----------------------------------------------------------------
    # MODE 1: --draft-only  (no tokeniser needed, saves unaligned draft)
    # ----------------------------------------------------------------
    if draft_only:
        print("=== DRAFT MODE (no alignment, not for patching) ===\n")
        dataset = build_dataset(tokenizer=None)

        with open(DRAFT_PATH, "w") as f:
            json.dump(dataset, f, indent=2)
        print(f"\nSaved {len(dataset)} UNALIGNED examples to {DRAFT_PATH}")
        print("WARNING: This is a draft. Run without --draft-only to produce")
        print("         the aligned dataset required for activation patching.")
        return

    # ----------------------------------------------------------------
    # Load tokeniser (REQUIRED for default and --align-only modes)
    # ----------------------------------------------------------------
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

    # ----------------------------------------------------------------
    # MODE 2: --align-only  (load existing dataset, align it)
    # ----------------------------------------------------------------
    if align_only:
        print(f"\nLoading existing dataset from {OUTPUT_PATH}...")
        with open(OUTPUT_PATH) as f:
            dataset = json.load(f)
        print(f"  Loaded {len(dataset)} examples.")
    else:
        # ----------------------------------------------------------------
        # MODE 3: Default  (build + align)
        # ----------------------------------------------------------------
        print()
        dataset = build_dataset(tokenizer=tokenizer)

    # ----------------------------------------------------------------
    # Perform alignment
    # ----------------------------------------------------------------
    total_before = len(dataset)
    print("\nPerforming token alignment...")
    dataset = perform_alignment(dataset, tokenizer)
    total_after = len(dataset)
    total_dropped = total_before - total_after

    if total_after == 0:
        print("\nFATAL: Zero examples survived alignment. Check prompts.")
        sys.exit(1)

    # ----------------------------------------------------------------
    # Save final aligned dataset
    # ----------------------------------------------------------------
    with open(OUTPUT_PATH, "w") as f:
        json.dump(dataset, f, indent=2)

    # ----------------------------------------------------------------
    # Sanity print (Constraint 6)
    # ----------------------------------------------------------------
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