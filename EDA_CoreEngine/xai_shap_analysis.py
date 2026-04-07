"""
xai_shap_analysis.py

Offline SHAP-based XAI analysis for the EDA Core Engine LightGBM model.

Purpose:
- Provide the second XAI implementation for the project, used offline on a laptop.
- Support dissertation/report writing with stronger ML interpretability evidence.
- Generate both global and local explanation artifacts for the final LightGBM model.

This version samples by scenario_id, not by individual rows, so that SHAP
analysis preserves full scenario context and remains aligned with the
scenario-based ML problem formulation.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap


# -----------------------------
# Config
# -----------------------------

DEFAULT_MODEL_PATH = Path("models") / "lightgbm_pipeline.joblib"
DEFAULT_DATA_PATH = Path("data_generated") / "ml_dataset" / "test_ml.csv"
DEFAULT_OUTPUT_DIR = Path("reports") / "xai_shap_outputs"


def build_feature_lists() -> tuple[list[str], list[str], list[str]]:
    """
    Exact feature schema used by the final LightGBM training pipeline.
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
        "distance_rank",
        "range_coverage_ratio",
        "runway_rank",
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

    feature_cols = num_cols + cat_cols
    return feature_cols, num_cols, cat_cols


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run offline SHAP analysis for the EDA LightGBM model."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the saved LightGBM pipeline joblib file.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the ML CSV split to analyse (default: test_ml.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where SHAP outputs will be saved.",
    )
    parser.add_argument(
        "--max-scenarios",
        type=int,
        default=30,
        help="Maximum number of unique scenarios to analyse with SHAP.",
    )
    parser.add_argument(
        "--scenario-id",
        type=str,
        default=None,
        help="Optional scenario_id to focus on for local explanation.",
    )
    return parser.parse_args()


def clean_feature_name(encoded_name: str) -> str:
    """
    Map sklearn transformed feature names back to original feature groups.

    Examples:
    - 'num__distance_km' -> 'distance_km'
    - 'cat__emergency_type_medical' -> 'emergency_type'
    """
    if encoded_name.startswith("num__"):
        return encoded_name.replace("num__", "", 1)

    if encoded_name.startswith("cat__"):
        raw = encoded_name.replace("cat__", "", 1)
        for base in CAT_COLS:
            prefix = f"{base}_"
            if raw == base or raw.startswith(prefix):
                return base
        return raw

    return encoded_name


def ensure_paths(model_path: Path, data_path: Path, output_dir: Path) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    output_dir.mkdir(parents=True, exist_ok=True)


def load_model_and_data(model_path: Path, data_path: Path) -> tuple:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message="X does not have valid feature names")

    model = joblib.load(model_path)
    df = pd.read_csv(data_path)

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required feature columns: {missing}")

    if "scenario_id" not in df.columns:
        raise ValueError("Dataset must contain a scenario_id column for scenario-based SHAP sampling.")

    return model, df


def prepare_subset_by_scenario(
    df: pd.DataFrame,
    max_scenarios: int,
) -> pd.DataFrame:
    """
    Sample whole scenarios, not individual rows.

    This preserves the full candidate-airport structure of each scenario.
    """
    unique_scenarios = sorted(df["scenario_id"].unique())

    if len(unique_scenarios) <= max_scenarios:
        selected_scenarios = unique_scenarios
    else:
        selected_scenarios = (
            pd.Series(unique_scenarios)
            .sample(n=max_scenarios, random_state=42)
            .sort_values()
            .tolist()
        )

    subset = df[df["scenario_id"].isin(selected_scenarios)].copy()
    return subset


def get_positive_class_shap_values(explainer, X_transformed):
    """
    Handle SHAP output shape robustly across versions.
    """
    shap_values = explainer.shap_values(X_transformed)

    if isinstance(shap_values, list):
        return shap_values[1]

    if getattr(shap_values, "ndim", None) == 3:
        return shap_values[:, :, 1]

    return shap_values


def aggregate_global_importance(
    shap_values_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate one-hot encoded SHAP values to original feature-level importance
    using mean absolute contribution.
    """
    mean_abs = shap_values_df.abs().mean(axis=0)
    agg = mean_abs.groupby([clean_feature_name(c) for c in mean_abs.index]).sum()
    out = (
        agg.sort_values(ascending=False)
        .rename("mean_abs_shap")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    return out


def aggregate_local_explanation(
    shap_row: pd.Series,
    feature_row: pd.Series,
) -> pd.DataFrame:
    """
    Aggregate a single row's SHAP values back to original features.
    Keeps signed contribution for local interpretation.
    """
    contrib_df = pd.DataFrame(
        {
            "encoded_feature": shap_row.index,
            "shap_value": shap_row.values,
        }
    )
    contrib_df["feature"] = contrib_df["encoded_feature"].map(clean_feature_name)

    local = contrib_df.groupby("feature", as_index=False)["shap_value"].sum()
    local["abs_shap"] = local["shap_value"].abs()

    raw_values = {}
    for feature in local["feature"]:
        raw_values[feature] = feature_row.get(feature, None)

    local["raw_value"] = local["feature"].map(raw_values)
    local = local.sort_values("abs_shap", ascending=False).reset_index(drop=True)
    return local


def choose_local_example(
    analysed_df: pd.DataFrame,
    y_prob: pd.Series,
    scenario_id: int | None,
) -> int:
    """
    Pick one row for local explanation.

    Priority:
    1. If scenario_id is provided: explain the highest-scoring row in that scenario.
    2. Otherwise: explain the highest-scoring row in the analysed subset.
    """
    if scenario_id is not None:
        mask = analysed_df["scenario_id"] == scenario_id
        if not mask.any():
            raise ValueError(f"scenario_id {scenario_id} not found in analysed subset.")
        candidate_probs = y_prob.loc[mask]
        return int(candidate_probs.idxmax())

    return int(y_prob.idxmax())


def save_global_plot(global_df: pd.DataFrame, output_path: Path, top_n: int = 15) -> None:
    top = global_df.head(top_n).iloc[::-1]
    plt.figure(figsize=(10, 7))
    plt.barh(top["feature"], top["mean_abs_shap"])
    plt.xlabel("Mean |SHAP value|")
    plt.ylabel("Feature")
    plt.title("Global SHAP Feature Importance (aggregated to original features)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_local_plot(local_df: pd.DataFrame, output_path: Path, top_n: int = 12) -> None:
    top = local_df.head(top_n).iloc[::-1]
    plt.figure(figsize=(10, 7))
    plt.barh(top["feature"], top["shap_value"])
    plt.xlabel("SHAP contribution")
    plt.ylabel("Feature")
    plt.title("Local SHAP Explanation (aggregated to original features)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    ensure_paths(args.model, args.data, args.output_dir)

    model, df = load_model_and_data(args.model, args.data)

    prep = model.named_steps["prep"]
    clf = model.named_steps["clf"]

    analysed_df = prepare_subset_by_scenario(df, args.max_scenarios)
    X_raw = analysed_df[FEATURE_COLS].copy()

    X_transformed = prep.transform(X_raw)
    feature_names_out = prep.get_feature_names_out()

    y_prob = pd.Series(
        model.predict_proba(X_raw)[:, 1],
        index=analysed_df.index,
        name="predicted_top_choice_score",
    )

    explainer = shap.TreeExplainer(clf)
    shap_values = get_positive_class_shap_values(explainer, X_transformed)

    shap_values_df = pd.DataFrame(
        shap_values,
        columns=feature_names_out,
        index=analysed_df.index,
    )

    # Global explanation
    global_df = aggregate_global_importance(shap_values_df)
    global_csv = args.output_dir / "global_feature_importance.csv"
    global_png = args.output_dir / "global_feature_importance.png"
    global_df.to_csv(global_csv, index=False)
    save_global_plot(global_df, global_png)

    # Local explanation
    local_idx = choose_local_example(analysed_df, y_prob, args.scenario_id)
    local_shap_row = shap_values_df.loc[local_idx]
    local_feature_row = X_raw.loc[local_idx]
    local_df = aggregate_local_explanation(local_shap_row, local_feature_row)

    local_csv = args.output_dir / "local_explanation.csv"
    local_png = args.output_dir / "local_explanation.png"
    local_df.to_csv(local_csv, index=False)
    save_local_plot(local_df, local_png)

    analysed_scenarios = sorted(analysed_df["scenario_id"].unique().tolist())

    summary = {
        "model_path": str(args.model),
        "data_path": str(args.data),
        "rows_analysed": int(len(analysed_df)),
        "scenarios_analysed": int(len(analysed_scenarios)),
        "scenario_ids_analysed": analysed_scenarios,
        "local_example_index": int(local_idx),
        "local_example_scenario_id": str(analysed_df.loc[local_idx, "scenario_id"]),
        "local_example_airport_icao": (
            analysed_df.loc[local_idx, "airport_icao"]
            if "airport_icao" in analysed_df.columns
            else None
        ),
        "local_example_predicted_score": float(y_prob.loc[local_idx]),
        "top_10_global_features": global_df.head(10).to_dict(orient="records"),
        "top_10_local_contributors": local_df.head(10).to_dict(orient="records"),
    }

    summary_json = args.output_dir / "xai_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\nSHAP XAI analysis complete.")
    print(f"Outputs saved to: {args.output_dir}")
    print(f"Rows analysed: {len(analysed_df)}")
    print(f"Scenarios analysed: {len(analysed_scenarios)}")
    print(f"Local example scenario_id: {analysed_df.loc[local_idx, 'scenario_id']}")
    print(f"- {global_csv.name}")
    print(f"- {global_png.name}")
    print(f"- {local_csv.name}")
    print(f"- {local_png.name}")
    print(f"- {summary_json.name}")


if __name__ == "__main__":
    FEATURE_COLS, NUM_COLS, CAT_COLS = build_feature_lists()
    args = parse_args()
    main()