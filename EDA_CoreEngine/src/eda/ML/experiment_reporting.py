"""
experiment_reporting.py

Reporting and visualization utilities for ML experiments in the EDA Core Engine (Increment 2).

This module is responsible for:
- creating structured folders for each ML model experiment
- saving evaluation figures (confusion matrix, ROC curve, PR curve)
- exporting metric summaries in JSON and text format

Purpose:
- ensure consistent and reproducible experiment outputs
- support clear comparison between models (Logistic Regression, Random Forest, etc.)
- provide ready-to-use figures and metrics for the final report
"""

from __future__ import annotations

# Standard library imports
from pathlib import Path
import json

# Numerical and plotting libraries
import numpy as np
import matplotlib.pyplot as plt

# Sklearn metrics for evaluation visualization
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve,
)

import pandas as pd

# Create experiment folder structure


def get_experiment_paths(model_name: str) -> dict[str, Path]:
    """
    Creates a standard folder structure for a given ML model.

    Each model will have:
    - figures/  -> saved plots (PNG)
    - metrics/  -> JSON and TXT summaries
    - notes/    -> optional manual notes

    Returns a dictionary of paths for easy access.
    """

    # Base path for this model
    base_path = Path("reports") / "ml_experiments" / model_name

    # Subfolders
    figures_path = base_path / "figures"
    metrics_path = base_path / "metrics"
    notes_path = base_path / "notes"

    # Create folders if they don't exist
    for folder in [base_path, figures_path, metrics_path, notes_path]:
        folder.mkdir(parents=True, exist_ok=True)

    return {
        "base": base_path,
        "figures": figures_path,
        "metrics": metrics_path,
        "notes": notes_path,
    }


# Confusion Matrix Visualization


def save_confusion_matrix_figure(
    y_true,
    y_pred,
    output_path: Path,
    title: str,
) -> None:
    """
    Generates and saves a confusion matrix plot.

    This helps visualize:
    - true positives
    - false positives
    - false negatives
    - true negatives
    """

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, colorbar=False)

    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    # Close figure to avoid memory issues when running multiple plots
    plt.close(fig)


# ROC Curve


def save_roc_curve_figure(
    y_true,
    y_prob,
    output_path: Path,
    title: str,
) -> None:
    """
    Generates and saves ROC curve.

    Shows trade-off between:
    - True Positive Rate
    - False Positive Rate
    """

    fpr, tpr, _ = roc_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(fpr, tpr, label="ROC Curve")
    ax.plot([0, 1], [0, 1], linestyle="--")  # random baseline

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close(fig)    


# Precision-Recall Curve


def save_pr_curve_figure(
    y_true,
    y_prob,
    output_path: Path,
    title: str,
) -> None:
    """
    Generates and saves Precision-Recall curve.

    Especially useful for imbalanced datasets.
    """

    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(recall, precision, label="PR Curve")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close(fig)    


# Save Metrics (JSON + TXT)


def save_metric_summary_json(metrics: dict, output_path: Path) -> None:
    """
    Saves metrics dictionary as JSON file.
    Useful for programmatic access later.
    """
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)


def save_metric_summary_txt(metrics: dict, output_path: Path) -> None:
    """
    Saves metrics in readable text format.
    Useful for quick inspection and documentation.
    """
    with output_path.open("w", encoding="utf-8") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")    

# Save Full Classification Report

def save_classification_report_json(report_dict: dict, output_path: Path) -> None:
    """
    Saves the full sklearn classification report dictionary as JSON.
    """
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=4)


def save_classification_report_csv(report_dict: dict, output_path: Path) -> None:
    """
    Saves the full sklearn classification report as a CSV table.

    This makes it easier to inspect precision / recall / F1 / support
    in spreadsheet-like form.
    """
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(output_path, index=True)      

# Save Simple Metric Bar Chart

def save_metric_bar_chart(
    metric_names: list[str],
    metric_values: list[float],
    output_path: Path,
    title: str,
    ylabel: str = "Score",
) -> None:
    """
    Saves a simple bar chart for selected metrics.
    Useful for precision / recall / F1 visualisation.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.bar(metric_names, metric_values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)      

# Comparison Chart for train, val, and test

def save_split_comparison_chart(
    splits: list[str],
    values: list[float],
    output_path: Path,
    title: str,
    ylabel: str = "Score",
) -> None:
    """
    Saves a bar chart comparing Train / Validation / Test performance.

    Used to:
    - check generalization
    - visualize overfitting
    """

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.bar(splits, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)    