"""
pipeline.py

End-to-end deterministic pipeline for the EDA Core Engine.

This module connects:
Scenario → Validation → Airport DB → Feature Engineering → Feasibility Filter → Ranking

It returns a DecisionReport, which is traceable and suitable for logging.

Increment 2 update:
- The redesigned distance model is carried inside Scenario.
- Feasible airports now include a distance-zone label:
    * preferred
    * extended
- Ranking uses that distance zone to apply a soft last-resort penalty.
"""

from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd

from typing import List, Tuple

from eda.airport_db import load_airports, Airport
from eda.features import compute_features, EngineFeatures
from eda.filter import is_feasible
from eda.models import DecisionReport, EvaluatedAirport
from eda.ranking import rank_options
from eda.scenario import Scenario
from eda.validation import validate_scenario
from eda.config import DEFAULT_TOP_K, DEFAULT_MAX_RANGE_KM
from eda.operational_constraints import OperationalConstraints

# ML CONFIG
MODEL_PATH = Path("models") / "lightgbm_pipeline.joblib"
MODEL_NAME = "LightGBM (Final)"

# feature column lists used by training
ML_NUM_COLS = [
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
     "distance_rank",        # rank of this airport by distance within scenario
    "range_coverage_ratio", # usable_range_km / distance_km — fuel urgency signal
    "runway_rank",          # rank of this airport by runway length within scenario
]

ML_CAT_COLS = [
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

ML_FEATURE_COLS = ML_NUM_COLS + ML_CAT_COLS

def _load_ml_model():
    """
    Loads the final selected ML ranking model

    Returns:
        Loaded pipeline object, or None if loading fails
    """
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None

def _build_ml_row(
    scenario: Scenario,
    airport: Airport,
    features: EngineFeatures,
    distance_zone: str,
) -> dict:
    """
    Builds one inference row for the LightGBM pipeline using the exact
    feature schema used during training.
    """
    row = {
        # Scenario / aircraft context
        "aircraft_lat": scenario.aircraft_lat,
        "aircraft_lon": scenario.aircraft_lon,
        "required_runway_m": scenario.required_runway_m,
        "max_range_km": scenario.max_range_km,
        "fuel_multiplier": scenario.fuel_multiplier,
        "aircraft_adjusted_range_km": scenario.aircraft_adjusted_range_km,
        "usable_range_km": scenario.usable_range_km,
        "extended_range_km": scenario.extended_range_km,

        # Airport / facility numeric features
        "runway_length_m": airport.runway_length_m,
        "runway_width_m": airport.runway_width_m,
        "has_ils": int(airport.has_ils),
        "has_medical": int(airport.has_medical),
        "has_rescue": int(airport.has_rescue),
        "has_firefighting": int(airport.has_firefighting),
        "has_maintenance": int(airport.has_maintenance),
        "fuel_available": int(airport.fuel_available),
        "open_24h": int(airport.open_24h),
        "is_international": int(airport.is_international),
        "tower_available": int(airport.tower_available),
        "weather_reporting": int(airport.weather_reporting),
        "slot_restricted": int(airport.slot_restricted),

        # Engineered numeric features
        "distance_km": features.distance_km,
        "runway_margin_m": features.runway_margin_m,

        # Categorical features
        "aircraft_type": scenario.aircraft_type,
        "aircraft_category": scenario.aircraft_category,
        "emergency_type": scenario.emergency_type.value,
        "fuel_state": scenario.fuel_state.value,
        "binding_side": scenario.binding_side.value,
        "surface_type": airport.surface_type,
        "approach_type": airport.approach_type,
        "medical_level": airport.medical_level,
        "rescue_category": airport.rescue_category,
        "closure_status": airport.closure_status,
        "restricted_status": airport.restricted_status,
        "unsafe_status": airport.unsafe_status,
        "civil_military": airport.civil_military,
        "distance_zone": distance_zone,
    }

    return row

def run_pipeline(
    scenario: Scenario,
    *,
    constraints: OperationalConstraints | None = None,
    top_k: int = DEFAULT_TOP_K,
    max_range_km: float = DEFAULT_MAX_RANGE_KM,
) -> DecisionReport:
    """
    Runs the deterministic decision pipeline.

    Args:
        scenario: Scenario input.
        constraints: Optional scenario-specific operational overrides.
        top_k: Number of top airports to return.
        max_range_km: Legacy compatibility parameter from Increment 1.
            It is no longer the final reachability authority once Scenario
            carries the redesigned range model.

    Returns:
        DecisionReport containing evaluated airports, feasible set, and top ranked outputs.
    """
    # 1) System-level validation
    validate_scenario(scenario)
    

    if constraints is None:
        constraints = OperationalConstraints()

    # 2) Load airport candidates
    airports = load_airports()

    evaluated: List[EvaluatedAirport] = []
    feasible_pairs: List[Tuple[Airport, EngineFeatures, str]] = []

    # 3) Evaluate each airport
    for a in airports:
        feats = compute_features(scenario, a)
        feas = is_feasible(
            scenario,
            a,
            feats,
            constraints=constraints,
            max_range_km=max_range_km,  # legacy compatibility only
        )

        evaluated.append(
            EvaluatedAirport(
                airport=a,
                features=feats,
                feasibility=feas,
            )
        )

        if feas.feasible:
            feasible_pairs.append((a, feats, feas.zone))

        # 4) Rank feasible airports (ML + fallback)
    model = _load_ml_model()

    if model is not None and feasible_pairs:
        try:
            rows = []

            for airport, feats, zone in feasible_pairs:
                row = _build_ml_row(scenario, airport, feats, zone)
                rows.append(row)

            df = pd.DataFrame(rows)

            # Compute within-scenario relative features.
            # These must be computed across all candidates together
            # so that ranks are relative to the full candidate set —
            # exactly matching how they were computed during training.

            # Rank by distance (1 = closest)
            df["distance_rank"] = (
                df["distance_km"]
                .rank(method="min", ascending=True)
                .astype(int)
            )

            # Fuel urgency × distance relationship
            df["range_coverage_ratio"] = (
                df["usable_range_km"] / df["distance_km"]
            ).round(4)

            # Rank by runway length (1 = longest)
            df["runway_rank"] = (
                df["runway_length_m"]
                .rank(method="min", ascending=False)
                .astype(int)
            )

            # Ensure column order matches training
            df = df[ML_FEATURE_COLS]

            probs = model.predict_proba(df)[:, 1]

            # Combine ML scores with airport data
            scored = []
            for (airport, feats, zone), prob in zip(feasible_pairs, probs):
                scored.append((airport, feats, zone, float(prob)))

            # Sort by ML probability
            scored.sort(key=lambda x: x[3], reverse=True)

            # Convert back to RankedOption format (reuse deterministic structure)
            from eda.ranking import RankedOption

            ranked_top = [
                RankedOption(
                    airport=a,
                    features=f,
                    score=prob,  # ML score now
                    distance_zone=z,
                )
                for a, f, z, prob in scored[:top_k]
            ]

        except Exception:
            # Fallback to deterministic ranking if ML fails
            ranked_top = rank_options(
                scenario.emergency_type,
                feasible_pairs,
                top_k=top_k,
            )
    else:
        # Fallback if model not loaded
        ranked_top = rank_options(
            scenario.emergency_type,
            feasible_pairs,
            top_k=top_k,
        )

    return DecisionReport(
        scenario=scenario,
        total_airports=len(airports),
        evaluated=evaluated,
        feasible=feasible_pairs,
        ranked_top=ranked_top,
    )