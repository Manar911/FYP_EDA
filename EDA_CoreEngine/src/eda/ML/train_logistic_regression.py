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

# Paths to the pre-split CSVs produced by build_ml_dataset.py
TRAIN_CSV = "data_generated/ml_dataset/train_ml.csv"
VAL_CSV = "data_generated/ml_dataset/val_ml.csv"
TEST_CSV = "data_generated/ml_dataset/test_ml.csv"

# Where to save the trained model pipeline after training
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "logistic_regression_pipeline.joblib"


def build_feature_lists() -> tuple[list[str], list[str], list[str]]:

    
    exclude_from_features = [
        "scenario_id",
        "is_top_choice",
        "target_rank",
        "airport_icao",
        "seed_airport_icao",
    ]

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

    feature_cols = [c for c in (num_cols + cat_cols)]
    return feature_cols, num_cols, cat_cols


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)

    print("Train shape:", train_df.shape)
    print("Val shape:", val_df.shape)
    print("Test shape:", test_df.shape)

    return train_df, val_df, test_df


def build_pipeline(num_cols: list[str], cat_cols: list[str]) -> Pipeline:
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
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    return pipeline


def evaluate_row_level(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    split_name: str,
) -> None:
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    print(f"\n===== {split_name} Row-Level Evaluation =====")
    print(classification_report(y, y_pred, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))

    roc = roc_auc_score(y, y_prob)
    pr_auc = average_precision_score(y, y_prob)

    print(f"ROC-AUC: {roc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")


def top1_accuracy(
    df: pd.DataFrame,
    model: Pipeline,
    feature_cols: list[str],
) -> float:
    correct = 0
    total = 0

    for scenario_id, group in df.groupby("scenario_id"):
        X_group = group[feature_cols]
        probs = model.predict_proba(X_group)[:, 1]

        pred_idx = probs.argmax()
        true_idx = group["is_top_choice"].values.argmax()

        if pred_idx == true_idx:
            correct += 1

        total += 1

    return correct / total if total > 0 else 0.0


def topk_accuracy(
    df: pd.DataFrame,
    model: Pipeline,
    feature_cols: list[str],
    k: int,
) -> float:
    correct = 0
    total = 0

    for scenario_id, group in df.groupby("scenario_id"):
        X_group = group[feature_cols]
        probs = model.predict_proba(X_group)[:, 1]

        top_k_pred_indices = probs.argsort()[::-1][:k]
        true_idx = group["is_top_choice"].values.argmax()

        if true_idx in top_k_pred_indices:
            correct += 1

        total += 1

    return correct / total if total > 0 else 0.0


def reciprocal_rank(
    df: pd.DataFrame,
    model: Pipeline,
    feature_cols: list[str],
) -> float:
    rr_sum = 0.0
    total = 0

    for scenario_id, group in df.groupby("scenario_id"):
        X_group = group[feature_cols]
        probs = model.predict_proba(X_group)[:, 1]

        ranked_indices = probs.argsort()[::-1]
        true_idx = group["is_top_choice"].values.argmax()

        rank_position = list(ranked_indices).index(true_idx) + 1
        rr_sum += 1.0 / rank_position
        total += 1

    return rr_sum / total if total > 0 else 0.0


def evaluate_scenario_level(
    df: pd.DataFrame,
    model: Pipeline,
    feature_cols: list[str],
    split_name: str,
) -> None:
    top1 = top1_accuracy(df, model, feature_cols)
    top3 = topk_accuracy(df, model, feature_cols, k=3)
    mrr = reciprocal_rank(df, model, feature_cols)

    print(f"\n===== {split_name} Scenario-Level Evaluation =====")
    print(f"Top-1 Accuracy: {top1:.4f}")
    print(f"Top-3 Accuracy: {top3:.4f}")
    print(f"MRR: {mrr:.4f}")


def main() -> None:
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

    print("\nTraining Logistic Regression baseline...")
    model.fit(X_train, y_train)
    print("Training complete.")

    evaluate_row_level(model, X_val, y_val, "Validation")
    evaluate_scenario_level(val_df, model, feature_cols, "Validation")

    evaluate_row_level(model, X_test, y_test, "Test")
    evaluate_scenario_level(test_df, model, feature_cols, "Test")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"\nSaved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()