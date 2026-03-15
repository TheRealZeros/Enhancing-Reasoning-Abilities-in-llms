import json
import os

import numpy as np
import pandas as pd

EVAL_FILE = "results/phase_2_behaviour/evaluation_results.csv"
DATASET_FILE = "dataset/processed/dataset.json"
OUTPUT_FILE = "dataset/processed/noisy_contrast_examples.json"


# ---------------------------------------------------------------------------
# Robust boolean parsing (mirrors cross_condition_patching.py)
# ---------------------------------------------------------------------------

def robust_bool(value) -> bool:
    """
    Normalise a value to bool.  Handles:
      - Python bool / numpy bool
      - int/float 0 or 1
      - strings: 'true', 'True', 'TRUE', 'false', 'False', 'FALSE',
        '1', '0', 'yes', 'no'
    Raises ValueError on anything else.
    """
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        if value == 1:
            return True
        if value == 0:
            return False
        raise ValueError(f"Cannot interpret numeric value {value!r} as bool")
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("true", "1", "yes"):
            return True
        if low in ("false", "0", "no"):
            return False
        raise ValueError(f"Cannot interpret string {value!r} as bool")
    raise ValueError(f"Cannot interpret {type(value).__name__} value {value!r} as bool")


def main():
    print("Loading evaluation results...")
    df = pd.read_csv(EVAL_FILE)

    print("Loading dataset...")
    with open(DATASET_FILE, "r") as f:
        dataset = {ex["id"]: ex for ex in json.load(f)}

    print("Finding noisy contrast examples (B incorrect AND D correct)...")

    noisy_contrast = []

    example_ids = df["example_id"].unique()

    for ex_id in example_ids:

        rows = df[df["example_id"] == ex_id]

        try:
            row_B = rows[rows["cell"] == "B"].iloc[0]
            row_D = rows[rows["cell"] == "D"].iloc[0]
        except IndexError:
            continue

        B_correct = robust_bool(row_B["correct"])
        D_correct = robust_bool(row_D["correct"])

        if (not B_correct) and D_correct:

            data = dataset[ex_id]

            # Store the clean cell schema (dict with prompt + metadata),
            # NOT the materialised prompt with EOS padding.
            cell_b_data = data["cells"]["B"]
            cell_d_data = data["cells"]["D"]

            # Ensure cell data is a dict (handle legacy string format)
            if isinstance(cell_b_data, str):
                cell_b_data = {"prompt": cell_b_data, "prefix_eos_pad": 0}
            if isinstance(cell_d_data, str):
                cell_d_data = {"prompt": cell_d_data, "prefix_eos_pad": 0}

            noisy_contrast.append({
                "example_id": ex_id,
                "gold_answer": data["answer"],
                "cell_B": cell_b_data,
                "cell_D": cell_d_data,
            })

    print(f"Found {len(noisy_contrast)} noisy contrast examples")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(noisy_contrast, f, indent=2)

    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()