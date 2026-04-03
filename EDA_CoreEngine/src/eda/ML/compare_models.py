"""
compare_models.py

Final model comparison for the EDA ML stage.

Purpose:
- Compare Logistic Regression, Random Forest, and LightGBM
- Focus on scenario-level performance (decision quality)
- Generate charts and tables for the final report

Key Metrics:
- Top-1 Accuracy (did we pick the correct airport)
- MRR (how well we rank the correct airport)

Output:
- Comparison charts (grouped bar plots)
- Comparison table (CSV)
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt



# Paths


BASE_REPORT_DIR = Path("reports/ml_experiments")
OUTPUT_DIR = BASE_REPORT_DIR / "comparison"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)



# Load metrics from saved JSON


def load_metrics(model_name: str, split: str):
    """
    Loads scenario-level metrics for a given model and split.
    """
    path = BASE_REPORT_DIR / model_name / "metrics" / f"{split}_metrics.json"
    return pd.read_json(path, typ="series")



# Build comparison table


def build_comparison_table():
    """
    Creates a table comparing all models on VALIDATION and TEST sets.
    """

    lr_val = load_metrics("logistic_regression", "validation")
    lr_test = load_metrics("logistic_regression", "test")

    rf_val = load_metrics("random_forest", "validation")
    rf_test = load_metrics("random_forest", "test")

    lgb_val = load_metrics("lightgbm", "validation")
    lgb_test = load_metrics("lightgbm", "test")

    df = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "LightGBM"],

        "Val Top-1": [
            lr_val["top1_accuracy"],
            rf_val["top1_accuracy"],
            lgb_val["top1_accuracy"],
        ],

        "Test Top-1": [
            lr_test["top1_accuracy"],
            rf_test["top1_accuracy"],
            lgb_test["top1_accuracy"],
        ],

        "Val MRR": [
            lr_val["mrr"],
            rf_val["mrr"],
            lgb_val["mrr"],
        ],

        "Test MRR": [
            lr_test["mrr"],
            rf_test["mrr"],
            lgb_test["mrr"],
        ],
    })

    return df



# Plot comparison charts


def plot_metric(df, val_col, test_col, title, filename):
    """
    Creates grouped bar chart for Validation vs Test comparison.
    """

    fig, ax = plt.subplots(figsize=(7, 5))

    x = range(len(df["Model"]))

    ax.bar(x, df[val_col], width=0.4, label="Validation")
    ax.bar([i + 0.4 for i in x], df[test_col], width=0.4, label="Test")

    ax.set_xticks([i + 0.2 for i in x])
    ax.set_xticklabels(df["Model"])

    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=300)
    plt.close(fig)



# Main


def main():

    print("Building final model comparison...")

    # Create comparison table
    df = build_comparison_table()

    # Save comparison table
    df.to_csv(TABLE_DIR / "comparison_metrics.csv", index=False)

    # Create comparison charts
    plot_metric(
        df,
        "Val Top-1",
        "Test Top-1",
        "Top-1 Accuracy (Validation vs Test)",
        "top1_comparison.png",
    )

    plot_metric(
        df,
        "Val MRR",
        "Test MRR",
        "MRR (Validation vs Test)",
        "mrr_comparison.png",
    )

    print("Comparison complete.")
    print("Saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()