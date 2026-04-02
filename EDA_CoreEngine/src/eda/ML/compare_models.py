"""
compare_models.py

Model comparison for the EDA ML stage.

Purpose:
- Compare Logistic Regression vs Random Forest
- Focus on scenario-level performance (decision quality)
- Generate charts and tables for report

Key Metrics:
- Top-1 Accuracy (did we pick the correct airport)
- MRR (how well we rank the correct airport)

Output:
- Comparison charts (bar plots)
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
    Creates a table comparing models on TEST set (final evaluation).
    """

    lr_test = load_metrics("logistic_regression", "test")
    rf_test = load_metrics("random_forest", "test")

    df = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest"],
        "Top-1 Accuracy": [lr_test["top1_accuracy"], rf_test["top1_accuracy"]],
        "MRR": [lr_test["mrr"], rf_test["mrr"]],
    })

    return df



# Plot comparison charts

def plot_metric(df, metric_name, filename):
    """
    Creates a bar chart comparing models for a given metric.
    """

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.bar(df["Model"], df[metric_name])
    ax.set_title(f"Model Comparison - {metric_name}")
    ax.set_ylabel(metric_name)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=300)
    plt.close(fig)



# Main

def main():

    print("Building model comparison...")

    # Create table
    df = build_comparison_table()

    # Save table
    df.to_csv(TABLE_DIR / "comparison_metrics.csv", index=False)

    # Create charts
    plot_metric(df, "Top-1 Accuracy", "top1_comparison.png")
    plot_metric(df, "MRR", "mrr_comparison.png")

    print("Comparison complete.")
    print("Saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()