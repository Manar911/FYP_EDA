"""
train_lightgbm.py

Gradient boosting model training for the EDA Core Engine (Increment 2).

This module trains and evaluates a LightGBM classifier using the prepared
ML datasets.

It performs:
- loading ML splits
- preprocessing for boosting models (encoding only)
- model training using LightGBM
- row-level and scenario-level evaluation
- full experiment reporting (figures + metrics + classification reports)

Purpose:
- capture complex non-linear relationships in diversion decision-making
- provide the strongest candidate model for final comparison
- support a lightweight and efficient deployment-oriented ML design
"""

from __future__ import annotations

# Standard / third-party imports

from pathlib import Path
import joblib
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import warnings

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names"
)

# Reporting utilities

from experiment_reporting import (
    get_experiment_paths,
    save_confusion_matrix_figure,
    save_roc_curve_figure,
    save_pr_curve_figure,
    save_metric_summary_json,
    save_metric_summary_txt,
    save_classification_report_json,
    save_classification_report_csv,
    save_metric_bar_chart,
    save_split_comparison_chart,
)

# File paths

TRAIN_CSV = "data_generated/ml_dataset/train_ml.csv"
VAL_CSV = "data_generated/ml_dataset/val_ml.csv"
TEST_CSV = "data_generated/ml_dataset/test_ml.csv"

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "lightgbm_pipeline.joblib"

# Feature Definition

def build_feature_lists():
    """
    Defines numerical and categorical features.

    LightGBM does not require feature scaling,
    so numerical features are passed through directly.
    """

    num_cols = [
        "aircraft_lat",
        "aircraft_lon",
        "required_runway_m",
        "max_range_km",
        "fuel_multiplier",
        "aircraft_adjusted_range_km",
        "usable_range_km",
        "extended_range_km",
        "runway_length_m",
        "runway_width_m",
        "has_ils",
        "has_medical",
        "has_rescue",
        "has_firefighting",
        "has_maintenance",
        "fuel_available",
        "open_24h",
        "is_international",
        "tower_available",
        "weather_reporting",
        "slot_restricted",
        "distance_km",
        "runway_margin_m",
    ]

    cat_cols = [
        "aircraft_type",
        "aircraft_category",
        "emergency_type",
        "fuel_state",
        "binding_side",
        "surface_type",
        "approach_type",
        "medical_level",
        "rescue_category",
        "closure_status",
        "restricted_status",
        "unsafe_status",
        "civil_military",
        "distance_zone",
    ]

    return num_cols + cat_cols, num_cols, cat_cols

# Load Dataset

def load_data():
    """Loads train / validation / test splits."""
    return (
        pd.read_csv(TRAIN_CSV),
        pd.read_csv(VAL_CSV),
        pd.read_csv(TEST_CSV),
    )


# Build Pipeline

def build_pipeline(num_cols, cat_cols):
    """
    Builds LightGBM pipeline.

    Differences from Logistic Regression:
    - No scaling required
    - Uses gradient boosting instead of linear model
    """

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    model = LGBMClassifier(
    n_estimators=300,          # more trees but controlled
    learning_rate=0.05,        # slower learning (less memorization)
    num_leaves=31,             # reduce complexity
    max_depth=6,               # limit tree depth
    min_child_samples=20,      # prevent tiny leaf nodes
    subsample=0.8,             # row sampling
    colsample_bytree=0.8,      # feature sampling
    reg_alpha=0.1,             # L1 regularization
    reg_lambda=0.1,            # L2 regularization
    class_weight="balanced",   # adjusts for class imbalance automatically
    random_state=42,           # ensures same output every run
    n_jobs=-1,                 # uses all CPU cores for faster training
    verbose=-1,                # hide LightGBM training output
)

    return Pipeline([
        ("prep", preprocessor),
        ("clf", model),
    ])

# Row-Level Evaluation

def evaluate_row(model, X, y, name):
    """
    Performs row-level evaluation and returns metrics + predictions.
    """

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    report_text = classification_report(y, y_pred, digits=4)
    report_dict = classification_report(y, y_pred, output_dict=True)

    print(f"\n===== {name} Row-Level Evaluation =====")
    print(report_text)
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))

    roc = roc_auc_score(y, y_prob)
    pr = average_precision_score(y, y_prob)

    print("ROC-AUC:", round(roc, 4))
    print("PR-AUC:", round(pr, 4))

    return {
        "y_pred": y_pred,
        "y_prob": y_prob,
        "roc_auc": round(roc, 4),
        "pr_auc": round(pr, 4),
        "classification_report": report_dict,
    }

# Scenario-Level Evaluation

def evaluate_scenario(df, model, feature_cols, name):
    """
    Evaluates model performance at scenario level (decision quality).
    """

    def top1():
        return sum(
            model.predict_proba(g[feature_cols])[:, 1].argmax() ==
            g["is_top_choice"].values.argmax()
            for _, g in df.groupby("scenario_id")
        ) / df["scenario_id"].nunique()

    def topk(k=3):
        return sum(
            g["is_top_choice"].values.argmax() in
            model.predict_proba(g[feature_cols])[:, 1].argsort()[::-1][:k]
            for _, g in df.groupby("scenario_id")
        ) / df["scenario_id"].nunique()

    def mrr():
        total = 0
        score = 0
        for _, g in df.groupby("scenario_id"):
            probs = model.predict_proba(g[feature_cols])[:, 1]
            ranking = probs.argsort()[::-1]
            idx = g["is_top_choice"].values.argmax()
            score += 1 / (list(ranking).index(idx) + 1)
            total += 1
        return score / total

    results = {
        "top1_accuracy": round(top1(), 4),
        "top3_accuracy": round(topk(), 4),
        "mrr": round(mrr(), 4),
    }

    print(f"\n===== {name} Scenario-Level Evaluation =====")
    for k, v in results.items():
        print(f"{k}: {v}")

    return results

# Main

def main():
    train_df, val_df, test_df = load_data()
    feature_cols, num_cols, cat_cols = build_feature_lists()

    X_train, y_train = train_df[feature_cols], train_df["is_top_choice"]
    X_val, y_val = val_df[feature_cols], val_df["is_top_choice"]
    X_test, y_test = test_df[feature_cols], test_df["is_top_choice"]

    model = build_pipeline(num_cols, cat_cols)

    # Create reporting folders
    report_paths = get_experiment_paths("lightgbm")

    print("Training LightGBM...")
    model.fit(X_train, y_train)
    print("Training complete.")

    # Training evaluation
    train_row = evaluate_row(model, X_train, y_train, "Train")
    train_scenario = evaluate_scenario(train_df, model, feature_cols, "Train")

    # ---------- VALIDATION ----------
    val_row = evaluate_row(model, X_val, y_val, "Validation")
    val_scenario = evaluate_scenario(val_df, model, feature_cols, "Validation")

    save_confusion_matrix_figure(
        y_val,
        val_row["y_pred"],
        report_paths["figures"] / "validation_confusion_matrix.png",
        "LightGBM Validation Confusion Matrix",
    )

    save_roc_curve_figure(
        y_val,
        val_row["y_prob"],
        report_paths["figures"] / "validation_roc_curve.png",
        "LightGBM Validation ROC",
    )

    save_pr_curve_figure(
        y_val,
        val_row["y_prob"],
        report_paths["figures"] / "validation_pr_curve.png",
        "LightGBM Validation PR",
    )

    val_metrics = {
        "split": "validation",
        "roc_auc": val_row["roc_auc"],
        "pr_auc": val_row["pr_auc"],
        **val_scenario,
    }

    save_metric_summary_json(
        val_metrics,
        report_paths["metrics"] / "validation_metrics.json",
    )

    save_metric_summary_txt(
        val_metrics,
        report_paths["metrics"] / "validation_metrics.txt",
    )

    save_classification_report_json(
        val_row["classification_report"],
        report_paths["metrics"] / "validation_classification_report.json",
    )

    save_classification_report_csv(
        val_row["classification_report"],
        report_paths["metrics"] / "validation_classification_report.csv",
    )

    val_class1 = val_row["classification_report"]["1"]

    save_metric_bar_chart(
        ["Precision", "Recall", "F1"],
        [
            val_class1["precision"],
            val_class1["recall"],
            val_class1["f1-score"],
        ],
        report_paths["figures"] / "validation_class1_metrics.png",
        "LightGBM - Validation Class 1 Metrics",
    )

    # ---------- TEST ----------
    test_row = evaluate_row(model, X_test, y_test, "Test")
    test_scenario = evaluate_scenario(test_df, model, feature_cols, "Test")

    save_confusion_matrix_figure(
        y_test,
        test_row["y_pred"],
        report_paths["figures"] / "test_confusion_matrix.png",
        "LightGBM Test Confusion Matrix",
    )

    save_roc_curve_figure(
        y_test,
        test_row["y_prob"],
        report_paths["figures"] / "test_roc_curve.png",
        "LightGBM Test ROC",
    )

    save_pr_curve_figure(
        y_test,
        test_row["y_prob"],
        report_paths["figures"] / "test_pr_curve.png",
        "LightGBM Test PR",
    )

    test_metrics = {
        "split": "test",
        "roc_auc": test_row["roc_auc"],
        "pr_auc": test_row["pr_auc"],
        **test_scenario,
    }

    save_metric_summary_json(
        test_metrics,
        report_paths["metrics"] / "test_metrics.json",
    )

    save_metric_summary_txt(
        test_metrics,
        report_paths["metrics"] / "test_metrics.txt",
    )

    save_classification_report_json(
        test_row["classification_report"],
        report_paths["metrics"] / "test_classification_report.json",
    )

    save_classification_report_csv(
        test_row["classification_report"],
        report_paths["metrics"] / "test_classification_report.csv",
    )

    test_class1 = test_row["classification_report"]["1"]

    save_metric_bar_chart(
        ["Precision", "Recall", "F1"],
        [
            test_class1["precision"],
            test_class1["recall"],
            test_class1["f1-score"],
        ],
        report_paths["figures"] / "test_class1_metrics.png",
        "LightGBM - Test Class 1 Metrics",
    )

    
    # Train vs Validation vs Test comparison charts
    save_split_comparison_chart(
        ["Train", "Validation", "Test"],
        [
            train_scenario["top1_accuracy"],
            val_scenario["top1_accuracy"],
            test_scenario["top1_accuracy"],
        ],
        report_paths["figures"] / "top1_split_comparison.png",
        "LightGBM - Top-1 Accuracy (Train vs Val vs Test)",
    )

    save_split_comparison_chart(
        ["Train", "Validation", "Test"],
        [
            train_scenario["mrr"],
            val_scenario["mrr"],
            test_scenario["mrr"],
        ],
        report_paths["figures"] / "mrr_split_comparison.png",
        "LightGBM - MRR (Train vs Val vs Test)",
    )

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("Model saved:", MODEL_PATH)
    print("LightGBM reports saved to:", report_paths["base"])


if __name__ == "__main__":
    main()