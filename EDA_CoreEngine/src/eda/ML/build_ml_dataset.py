"""
build_ml_dataset.py

ML dataset builder for the airport ranking model.

This module takes the full generated dataset and produces a clean, ML-ready CSV by:
- Filtering out infeasible airport candidates
- Keeping one positive (top choice) and the top-K ranked negatives per scenario
- Dropping leakage-prone and non-ML columns (scores, names, feasibility metadata)

Input:  data_generated/training_1000/dataset_full.csv
Output: data_generated/ml_dataset/ml_dataset.csv
"""

from __future__ import annotations

import pandas as pd


INPUT_CSV = "data_generated/training_1000/dataset_full.csv"
OUTPUT_CSV = "data_generated/ml_dataset/ml_dataset.csv"
TOP_K_NEGATIVES = 15


def select_top_k_candidates(group: pd.DataFrame, k: int = 15) -> pd.DataFrame:
    positive = group[group["is_top_choice"] == 1]
    negatives = group[group["is_top_choice"] == 0].sort_values("target_rank").head(k)
    return pd.concat([positive, negatives], axis=0)


def main() -> None:
    # 1) Load full dataset
    df = pd.read_csv(INPUT_CSV)
    print("Loaded dataset shape:", df.shape)

    # 2) Keep only feasible airports
    df = df[df["feasible"] == 1].copy()
    print("After feasible filter:", df.shape)

    # 3) Keep one positive + top-K negative competitors per scenario
    df_ml = (
        df.groupby("scenario_id", group_keys=False)
          .apply(lambda g: select_top_k_candidates(g, k=TOP_K_NEGATIVES))
          .reset_index(drop=True)
    )
    print("After top-K selection:", df_ml.shape)

    # 4) Sanity checks
    print("Unique scenarios:", df_ml["scenario_id"].nunique())

    positive_counts = df_ml.groupby("scenario_id")["is_top_choice"].sum()
    print("Min positives per scenario:", positive_counts.min())
    print("Max positives per scenario:", positive_counts.max())

    candidate_counts = df_ml.groupby("scenario_id").size()
    print("Min candidates per scenario:", candidate_counts.min())
    print("Max candidates per scenario:", candidate_counts.max())
    print("Average candidates per scenario:", candidate_counts.mean())

    # 5) Drop leakage / non-ML columns
    columns_to_drop = [
        "baseline_score",
        "feasibility_reason",
        "feasible",
        "split",
        "airport_name",
        "airport_city",
        "airport_country",
        "airport_iata",
    ]
    df_ml = df_ml.drop(columns=columns_to_drop, errors="ignore")

    # 6) Save ML-ready dataset
    df_ml.to_csv(OUTPUT_CSV, index=False)
    print(f"ML-ready dataset saved to: {OUTPUT_CSV}")
    print("Final ML dataset shape:", df_ml.shape)


if __name__ == "__main__":
    main()