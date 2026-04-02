"""
train_logistic_regression.py

Baseline classification training for the EDA Core Engine (Increment 2).

This module trains and evaluates a Logistic Regression model using the prepared
ML datasets.

It performs:
- loading of train/validation/test ML splits
- feature/target separation
- preprocessing for linear models (scaling + one-hot encoding)
- model training with class balancing
- row-level classification evaluation
- scenario-level Top-1 decision evaluation
- model export for later comparison and deployment studies

Purpose:
- establish a lightweight and interpretable baseline classifier
- provide a reference point for later comparison with tree-based models
- evaluate whether the ML layer can reproduce expert-system top-choice behaviour
"""

from __future__ import annotations


# Standard / third-party imports


from pathlib import Path
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
)


# File paths (ML-ready dataset splits)


TRAIN_CSV = "data_generated/ml_dataset/train_ml.csv"
VAL_CSV = "data_generated/ml_dataset/val_ml.csv"
TEST_CSV = "data_generated/ml_dataset/test_ml.csv"


# Model output path


MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "logistic_regression_pipeline.joblib"



# Feature Definition


def build_feature_lists() -> tuple[list[str], list[str], list[str]]:
    """
    Defines which columns are used as features.

    We explicitly separate:
    - numerical features → require scaling
    - categorical features → require encoding

    Also ensures we DO NOT include any leakage columns such as:
    - target (is_top_choice)
    - identifiers (scenario_id, airport codes)
    """


    # Numerical features (continuous / ordinal)
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

    # Categorical features (must be encoded)
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

    feature_cols = num_cols + cat_cols

    return feature_cols, num_cols, cat_cols



# Load Dataset Splits


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads train / validation / test datasets.

    These splits were created earlier with:
    - scenario-level separation
    - no leakage between splits
    """

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)

    print("Train shape:", train_df.shape)
    print("Val shape:", val_df.shape)
    print("Test shape:", test_df.shape)

    return train_df, val_df, test_df



# Build ML Pipeline


def build_pipeline(num_cols: list[str], cat_cols: list[str]) -> Pipeline:
    """
    Builds the full ML pipeline:

    - Numerical features → StandardScaler
    - Categorical features → OneHotEncoder
    - Model → Logistic Regression

    Important:
    Logistic Regression requires scaling because it is sensitive to feature magnitude.
    """

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,              # ensure convergence
                    class_weight="balanced",    # handle class imbalance
                    random_state=42,
                ),
            ),
        ]
    )

    return pipeline



# Row-Level Evaluation

def evaluate_row_level(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    split_name: str,
) -> dict:
    """
    Evaluates classification performance at row level.

    Metrics:
    - precision / recall / F1
    - confusion matrix
    - ROC-AUC
    - PR-AUC
    - full classification report (dictionary form for saving)
    """

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    # String version for terminal output
    report_text = classification_report(y, y_pred, digits=4)

    # Dictionary version for saving
    report_dict = classification_report(y, y_pred, output_dict=True)

    print(f"\n===== {split_name} Row-Level Evaluation =====")
    print(report_text)
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))

    roc = roc_auc_score(y, y_prob)
    pr_auc = average_precision_score(y, y_prob)

    print(f"ROC-AUC: {roc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")

    return {
        "y_pred": y_pred,
        "y_prob": y_prob,
        "roc_auc": round(roc, 4),
        "pr_auc": round(pr_auc, 4),
        "classification_report": report_dict,
    }



# Scenario-Level Evaluation


def top1_accuracy(df, model, feature_cols):
    """
    Measures how often the model selects the correct airport as the top choice.
    """
    correct = 0
    total = 0

    for scenario_id, group in df.groupby("scenario_id"):
        probs = model.predict_proba(group[feature_cols])[:, 1]

        pred_idx = probs.argmax()
        true_idx = group["is_top_choice"].values.argmax()

        if pred_idx == true_idx:
            correct += 1

        total += 1

    return correct / total if total > 0 else 0.0


def topk_accuracy(df, model, feature_cols, k):
    """
    Measures whether the correct airport appears in top-K predictions.
    """
    correct = 0
    total = 0

    for _, group in df.groupby("scenario_id"):
        probs = model.predict_proba(group[feature_cols])[:, 1]

        top_k = probs.argsort()[::-1][:k]
        true_idx = group["is_top_choice"].values.argmax()

        if true_idx in top_k:
            correct += 1

        total += 1

    return correct / total if total > 0 else 0.0


def reciprocal_rank(df, model, feature_cols):
    """
    Measures ranking quality:
    - higher score means correct airport appears earlier in ranking
    """
    rr_sum = 0.0
    total = 0

    for _, group in df.groupby("scenario_id"):
        probs = model.predict_proba(group[feature_cols])[:, 1]

        ranking = probs.argsort()[::-1]
        true_idx = group["is_top_choice"].values.argmax()

        rank = list(ranking).index(true_idx) + 1
        rr_sum += 1.0 / rank
        total += 1

    return rr_sum / total if total > 0 else 0.0


def evaluate_scenario_level(df, model, feature_cols, split_name) -> dict:
    """
    Evaluates decision quality at scenario level.

    This is the most important evaluation for the EDA system.
    """

    top1 = top1_accuracy(df, model, feature_cols)
    top3 = topk_accuracy(df, model, feature_cols, k=3)
    mrr = reciprocal_rank(df, model, feature_cols)

    print(f"\n===== {split_name} Scenario-Level Evaluation =====")
    print(f"Top-1 Accuracy: {top1:.4f}")
    print(f"Top-3 Accuracy: {top3:.4f}")
    print(f"MRR: {mrr:.4f}")

    return {
        "top1_accuracy": round(top1, 4),
        "top3_accuracy": round(top3, 4),
        "mrr": round(mrr, 4),
    }



# Main Execution


def main() -> None:
    """
    Full pipeline execution:
    - load data
    - prepare features
    - train model
    - evaluate performance
    - save trained model
    """

    train_df, val_df, test_df = load_splits()

    feature_cols, num_cols, cat_cols = build_feature_lists()

    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]
    X_test = test_df[feature_cols]

    y_train = train_df["is_top_choice"]
    y_val = val_df["is_top_choice"]
    y_test = test_df["is_top_choice"]

    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)
    print("X_test shape:", X_test.shape)

    print("y_train positives:", int(y_train.sum()))
    print("y_val positives:", int(y_val.sum()))
    print("y_test positives:", int(y_test.sum()))

    model = build_pipeline(num_cols, cat_cols)
    
    # Create output folders for this experiment
    report_paths = get_experiment_paths("logistic_regression")

    print("\nTraining Logistic Regression baseline...")
    model.fit(X_train, y_train)
    print("Training complete.")

    # Validation evaluation + saved reporting outputs

    val_row = evaluate_row_level(model, X_val, y_val, "Validation")
    val_scenario = evaluate_scenario_level(val_df, model, feature_cols, "Validation")

    save_confusion_matrix_figure(
        y_val,
        val_row["y_pred"],
        report_paths["figures"] / "validation_confusion_matrix.png",
        "Logistic Regression - Validation Confusion Matrix",
    )

    save_roc_curve_figure(
        y_val,
        val_row["y_prob"],
        report_paths["figures"] / "validation_roc_curve.png",
        "Logistic Regression - Validation ROC Curve",
    )

    save_pr_curve_figure(
        y_val,
        val_row["y_prob"],
        report_paths["figures"] / "validation_pr_curve.png",
        "Logistic Regression - Validation Precision-Recall Curve",
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
        "Logistic Regression - Validation Class 1 Metrics",
    )
    # Test evaluation + saved reporting outputs

    test_row = evaluate_row_level(model, X_test, y_test, "Test")
    test_scenario = evaluate_scenario_level(test_df, model, feature_cols, "Test")

    save_confusion_matrix_figure(
        y_test,
        test_row["y_pred"],
        report_paths["figures"] / "test_confusion_matrix.png",
        "Logistic Regression - Test Confusion Matrix",
    )

    save_roc_curve_figure(
        y_test,
        test_row["y_prob"],
        report_paths["figures"] / "test_roc_curve.png",
        "Logistic Regression - Test ROC Curve",
    )

    save_pr_curve_figure(
        y_test,
        test_row["y_prob"],
        report_paths["figures"] / "test_pr_curve.png",
        "Logistic Regression - Test Precision-Recall Curve",
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
        "Logistic Regression - Test Class 1 Metrics",
    )    

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"\nSaved model to: {MODEL_PATH}")

    print("Logistic Regression reports saved to:", report_paths["base"])

if __name__ == "__main__":
    main()