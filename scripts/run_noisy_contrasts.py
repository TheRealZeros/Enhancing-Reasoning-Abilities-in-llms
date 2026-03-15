import json
import pandas as pd

EVAL_FILE = "results/evaluation_results.csv"
DATASET_FILE = "dataset/dataset.json"
OUTPUT_FILE = "results/noisy_contrast_examples.json"


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
        except:
            continue

        B_correct = bool(row_B["correct"])
        D_correct = bool(row_D["correct"])

        if (not B_correct) and D_correct:

            data = dataset[ex_id]

            noisy_contrast.append({
                "example_id": ex_id,
                "gold_answer": data["answer"],
                "cell_B": {
                    "prompt": data["cells"]["B"]
                },
                "cell_D": {
                    "prompt": data["cells"]["D"]
                }
            })

    print(f"Found {len(noisy_contrast)} noisy contrast examples")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(noisy_contrast, f, indent=2)

    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()