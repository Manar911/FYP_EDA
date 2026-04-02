"""
train_random_forest.py

Non-linear ensemble model training for the EDA Core Engine (Increment 2).

This module trains and evaluates a Random Forest classifier using the prepared
ML datasets.

It performs:
- loading ML splits
- preprocessing for tree-based models (encoding only)
- model training using ensemble decision trees
- evaluation using row-level and scenario-level metrics
- comparison against Logistic Regression baseline

Purpose:
- capture non-linear feature interactions in diversion decision-making
- improve classification precision while maintaining high recall
- provide a stronger benchmark before advanced boosting models
"""

from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TRAIN_CSV = "data_generated/ml_dataset/train_ml.csv"
VAL_CSV = "data_generated/ml_dataset/val_ml.csv"
TEST_CSV = "data_generated/ml_dataset/test_ml.csv"

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "random_forest_pipeline.joblib"


def build_feature_lists():
    num_cols = [
        "aircraft_lat","aircraft_lon","required_runway_m","max_range_km",
        "fuel_multiplier","aircraft_adjusted_range_km","usable_range_km",
        "extended_range_km","runway_length_m","runway_width_m",
        "has_ils","has_medical","has_rescue","has_firefighting",
        "has_maintenance","fuel_available","open_24h","is_international",
        "tower_available","weather_reporting","slot_restricted",
        "distance_km","runway_margin_m"
    ]

    cat_cols = [
        "aircraft_type","aircraft_category","emergency_type","fuel_state",
        "binding_side","surface_type","approach_type","medical_level",
        "rescue_category","closure_status","restricted_status",
        "unsafe_status","civil_military","distance_zone"
    ]

    feature_cols = num_cols + cat_cols
    return feature_cols, num_cols, cat_cols


def load_data():
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)
    return train_df, val_df, test_df


def build_pipeline(num_cols, cat_cols):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("clf", model),
    ])

    return pipeline


def evaluate_row(model, X, y, name):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    print(f"\n===== {name} Row-Level Evaluation =====")
    print(classification_report(y, y_pred, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))

    print("ROC-AUC:", round(roc_auc_score(y, y_prob), 4))
    print("PR-AUC:", round(average_precision_score(y, y_prob), 4))


def top1(df, model, feature_cols):
    correct = 0
    for _, g in df.groupby("scenario_id"):
        probs = model.predict_proba(g[feature_cols])[:, 1]
        if probs.argmax() == g["is_top_choice"].values.argmax():
            correct += 1
    return correct / df["scenario_id"].nunique()


def topk(df, model, feature_cols, k=3):
    correct = 0
    for _, g in df.groupby("scenario_id"):
        probs = model.predict_proba(g[feature_cols])[:, 1]
        if g["is_top_choice"].values.argmax() in probs.argsort()[::-1][:k]:
            correct += 1
    return correct / df["scenario_id"].nunique()


def mrr(df, model, feature_cols):
    total = 0
    rr = 0
    for _, g in df.groupby("scenario_id"):
        probs = model.predict_proba(g[feature_cols])[:, 1]
        ranking = probs.argsort()[::-1]
        true_idx = g["is_top_choice"].values.argmax()
        rank = list(ranking).index(true_idx) + 1
        rr += 1 / rank
        total += 1
    return rr / total


def evaluate_scenario(df, model, feature_cols, name):
    print(f"\n===== {name} Scenario-Level Evaluation =====")
    print("Top-1 Accuracy:", round(top1(df, model, feature_cols), 4))
    print("Top-3 Accuracy:", round(topk(df, model, feature_cols), 4))
    print("MRR:", round(mrr(df, model, feature_cols), 4))


def main():
    train_df, val_df, test_df = load_data()
    feature_cols, num_cols, cat_cols = build_feature_lists()

    X_train = train_df[feature_cols]
    y_train = train_df["is_top_choice"]

    X_val = val_df[feature_cols]
    y_val = val_df["is_top_choice"]

    X_test = test_df[feature_cols]
    y_test = test_df["is_top_choice"]

    model = build_pipeline(num_cols, cat_cols)

    print("Training Random Forest...")
    model.fit(X_train, y_train)
    print("Training complete.")

    evaluate_row(model, X_val, y_val, "Validation")
    evaluate_scenario(val_df, model, feature_cols, "Validation")

    evaluate_row(model, X_test, y_test, "Test")
    evaluate_scenario(test_df, model, feature_cols, "Test")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print("Model saved:", MODEL_PATH)


if __name__ == "__main__":
    main()