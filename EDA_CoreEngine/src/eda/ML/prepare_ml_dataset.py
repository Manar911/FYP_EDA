"""
prepare_ml_dataset.py

Machine learning data preparation pipeline for the EDA Core Engine (Increment 2).

This module transforms the ML-ready dataset into structured inputs suitable for
model training and evaluation.

It performs:
- feature selection and grouping (numerical + categorical)
- leakage-safe dataset splitting by scenario_id
- classification and ranking target preparation
- preprocessing pipeline construction (encoding + scaling)
- generation of train/validation/test datasets

Purpose:
- ensure correct ML problem formulation (classification + ranking)
- prevent data leakage through scenario-level splitting
- prepare data for both linear and tree-based models
- establish a reproducible and traceable ML workflow

This module represents the transition from deterministic decision logic
to a learned decision model (imitation learning from expert system outputs).
"""


from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


INPUT_CSV = "data_generated/ml_dataset/ml_dataset.csv"
OUTPUT_DIR = Path("data_generated/ml_dataset")


def main() -> None:
    
    # 1) Load ML-ready dataset
    
    df = pd.read_csv(INPUT_CSV)

    print("Loaded ML dataset shape:", df.shape)
    print(df.head())

    
    # 2) Define grouping + targets
   
    group_col = "scenario_id"
    target_class_col = "is_top_choice"
    target_rank_col = "target_rank"

    
    # 3) Define columns to exclude from model inputs
    
    exclude_from_features = [
        "scenario_id",       # grouping only
        "is_top_choice",     # classification target
        "target_rank",       # ranking target
        "airport_icao",      # identifier
        "seed_airport_icao", # identifier / shortcut context
    ]

    
    # 4) Define numerical and categorical feature groups
    
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

    
    # 5) Build feature column list and validate groups
    
    feature_cols = [c for c in df.columns if c not in exclude_from_features]

    print("Total feature columns:", len(feature_cols))
    print("Numerical columns:", len(num_cols))
    print("Categorical columns:", len(cat_cols))

    missing_num = [c for c in num_cols if c not in df.columns]
    missing_cat = [c for c in cat_cols if c not in df.columns]

    print("Missing numerical columns:", missing_num)
    print("Missing categorical columns:", missing_cat)

    if missing_num:
        raise ValueError(f"Missing numerical columns: {missing_num}")

    if missing_cat:
        raise ValueError(f"Missing categorical columns: {missing_cat}")

    expected_feature_cols = set(num_cols + cat_cols)
    actual_feature_cols = set(feature_cols)

    unexpected_cols = sorted(actual_feature_cols - expected_feature_cols)
    missing_from_feature_design = sorted(expected_feature_cols - actual_feature_cols)

    print("Unexpected feature columns:", unexpected_cols)
    print("Expected but missing from feature design:", missing_from_feature_design)

    if missing_from_feature_design:
        raise ValueError(
            f"Some expected feature columns are missing from feature_cols: "
            f"{missing_from_feature_design}"
        )

    
    # 6) Re-split by scenario_id (NOT by rows)
    
    scenario_ids = df[group_col].unique()

    train_ids, temp_ids = train_test_split(
        scenario_ids,
        test_size=0.30,
        random_state=42,
    )

    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=0.50,
        random_state=42,
    )

    train_df = df[df[group_col].isin(train_ids)].copy()
    val_df = df[df[group_col].isin(val_ids)].copy()
    test_df = df[df[group_col].isin(test_ids)].copy()

    print("Train shape:", train_df.shape)
    print("Val shape:", val_df.shape)
    print("Test shape:", test_df.shape)

    print("Train scenarios:", train_df[group_col].nunique())
    print("Val scenarios:", val_df[group_col].nunique())
    print("Test scenarios:", test_df[group_col].nunique())

    
    # 7) Leakage check across splits
    
    train_scenarios = set(train_df[group_col].unique())
    val_scenarios = set(val_df[group_col].unique())
    test_scenarios = set(test_df[group_col].unique())

    print("Train ∩ Val:", len(train_scenarios & val_scenarios))
    print("Train ∩ Test:", len(train_scenarios & test_scenarios))
    print("Val ∩ Test:", len(val_scenarios & test_scenarios))

    if train_scenarios & val_scenarios:
        raise ValueError("Leakage detected between train and val scenario IDs.")
    if train_scenarios & test_scenarios:
        raise ValueError("Leakage detected between train and test scenario IDs.")
    if val_scenarios & test_scenarios:
        raise ValueError("Leakage detected between val and test scenario IDs.")

    
    # 8) Prepare X / y
    
    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]
    X_test = test_df[feature_cols]

    y_train = train_df[target_class_col]
    y_val = val_df[target_class_col]
    y_test = test_df[target_class_col]

    rank_train = train_df[target_rank_col]
    rank_val = val_df[target_rank_col]
    rank_test = test_df[target_rank_col]

    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)
    print("X_test shape:", X_test.shape)

    print("y_train positives:", int(y_train.sum()))
    print("y_val positives:", int(y_val.sum()))
    print("y_test positives:", int(y_test.sum()))

    print("rank_train non-null:", int(rank_train.notna().sum()))
    print("rank_val non-null:", int(rank_val.notna().sum()))
    print("rank_test non-null:", int(rank_test.notna().sum()))

    
    # 9) Build preprocessors
    
    # Logistic Regression preprocessor:
    # - scale numeric features
    # - one-hot encode categorical features
    preprocessor_linear = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    # Tree-model preprocessor:
    # - keep numeric features as-is
    # - one-hot encode categorical features
    preprocessor_tree = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    # Fit once here just to validate preprocessing works
    X_train_linear = preprocessor_linear.fit_transform(X_train)
    X_val_linear = preprocessor_linear.transform(X_val)
    X_test_linear = preprocessor_linear.transform(X_test)

    X_train_tree = preprocessor_tree.fit_transform(X_train)
    X_val_tree = preprocessor_tree.transform(X_val)
    X_test_tree = preprocessor_tree.transform(X_test)

    print("Linear preprocessed train shape:", X_train_linear.shape)
    print("Linear preprocessed val shape:", X_val_linear.shape)
    print("Linear preprocessed test shape:", X_test_linear.shape)

    print("Tree preprocessed train shape:", X_train_tree.shape)
    print("Tree preprocessed val shape:", X_val_tree.shape)
    print("Tree preprocessed test shape:", X_test_tree.shape)

    
    # 10) Save split CSVs for traceability
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(OUTPUT_DIR / "train_ml.csv", index=False)
    val_df.to_csv(OUTPUT_DIR / "val_ml.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test_ml.csv", index=False)

    print("Saved ML splits successfully.")
    print("Files created:")
    print("-", OUTPUT_DIR / "train_ml.csv")
    print("-", OUTPUT_DIR / "val_ml.csv")
    print("-", OUTPUT_DIR / "test_ml.csv")


if __name__ == "__main__":
    main()