"""
pipeline.py  —  EDA 

End-to-end pipeline for the EDA Core Engine.

"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import joblib
import pandas as pd

from airport_db import load_airports, Airport
from features import compute_features, EngineFeatures
from filter import is_feasible
from models import DecisionReport, EvaluatedAirport
from ranking import rank_options, RankedOption
from scenario import Scenario
from validation import validate_scenario
from config import DEFAULT_TOP_K, DEFAULT_MAX_RANGE_KM
from operational_constraints import OperationalConstraints

#  Model path — absolute, resolved from this file's location 
_HERE       = Path(__file__).resolve().parent   
_ROOT       = _HERE.parent                      
MODEL_PATH  = _ROOT / "model" / "lightgbm_pipeline.joblib"

ML_NUM_COLS = [
    "aircraft_lat", "aircraft_lon", "required_runway_m",
    "max_range_km", "fuel_multiplier", "aircraft_adjusted_range_km",
    "usable_range_km", "extended_range_km", "runway_length_m",
    "runway_width_m", "has_ils", "has_medical", "has_rescue",
    "has_firefighting", "has_maintenance", "fuel_available",
    "open_24h", "is_international", "tower_available",
    "weather_reporting", "slot_restricted", "distance_km",
    "runway_margin_m", "distance_rank", "range_coverage_ratio",
    "runway_rank",
]

ML_CAT_COLS = [
    "aircraft_type", "aircraft_category", "emergency_type",
    "fuel_state", "binding_side", "surface_type", "approach_type",
    "medical_level", "rescue_category", "closure_status",
    "restricted_status", "unsafe_status", "civil_military",
    "distance_zone",
]

ML_FEATURE_COLS = ML_NUM_COLS + ML_CAT_COLS


def _load_ml_model():
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
    return {
        "aircraft_lat":               scenario.aircraft_lat,
        "aircraft_lon":               scenario.aircraft_lon,
        "required_runway_m":          scenario.required_runway_m,
        "max_range_km":               scenario.max_range_km,
        "fuel_multiplier":            scenario.fuel_multiplier,
        "aircraft_adjusted_range_km": scenario.aircraft_adjusted_range_km,
        "usable_range_km":            scenario.usable_range_km,
        "extended_range_km":          scenario.extended_range_km,
        "runway_length_m":            airport.runway_length_m,
        "runway_width_m":             airport.runway_width_m,
        "has_ils":                    int(airport.has_ils),
        "has_medical":                int(airport.has_medical),
        "has_rescue":                 int(airport.has_rescue),
        "has_firefighting":           int(airport.has_firefighting),
        "has_maintenance":            int(airport.has_maintenance),
        "fuel_available":             int(airport.fuel_available),
        "open_24h":                   int(airport.open_24h),
        "is_international":           int(airport.is_international),
        "tower_available":            int(airport.tower_available),
        "weather_reporting":          int(airport.weather_reporting),
        "slot_restricted":            int(airport.slot_restricted),
        "distance_km":                features.distance_km,
        "runway_margin_m":            features.runway_margin_m,
        "aircraft_type":              scenario.aircraft_type,
        "aircraft_category":          scenario.aircraft_category,
        "emergency_type":             scenario.emergency_type.value,
        "fuel_state":                 scenario.fuel_state.value,
        "binding_side":               scenario.binding_side.value,
        "surface_type":               airport.surface_type,
        "approach_type":              airport.approach_type,
        "medical_level":              airport.medical_level,
        "rescue_category":            airport.rescue_category,
        "closure_status":             airport.closure_status,
        "restricted_status":          airport.restricted_status,
        "unsafe_status":              airport.unsafe_status,
        "civil_military":             airport.civil_military,
        "distance_zone":              distance_zone,
    }


def run_pipeline(
    scenario: Scenario,
    *,
    constraints: OperationalConstraints | None = None,
    top_k: int = DEFAULT_TOP_K,
    max_range_km: float = DEFAULT_MAX_RANGE_KM,
    use_ml: bool = True,
) -> DecisionReport:
    validate_scenario(scenario)

    if constraints is None:
        constraints = OperationalConstraints()

    airports = load_airports()
    evaluated:      List[EvaluatedAirport]                  = []
    feasible_pairs: List[Tuple[Airport, EngineFeatures, str]] = []

    for airport in airports:
        feats = compute_features(scenario, airport)
        feas  = is_feasible(
            scenario, airport, feats,
            constraints=constraints,
            max_range_km=max_range_km,
        )
        evaluated.append(EvaluatedAirport(
            airport=airport, features=feats, feasibility=feas,
        ))
        if feas.feasible:
            feasible_pairs.append((airport, feats, feas.zone))

    ranked_top = _rank(scenario, feasible_pairs, top_k, use_ml)

    return DecisionReport(
        scenario=scenario,
        total_airports=len(airports),
        evaluated=evaluated,
        feasible=feasible_pairs,
        ranked_top=ranked_top,
    )


def _rank(
    scenario: Scenario,
    feasible_pairs: List[Tuple[Airport, EngineFeatures, str]],
    top_k: int,
    use_ml: bool,
) -> List[RankedOption]:
    if not feasible_pairs:
        return []

    if not use_ml:
        print("Pipeline: deterministic ranking (ML disabled by integrity check)")
        return rank_options(scenario.emergency_type, feasible_pairs, top_k=top_k)

    model = _load_ml_model()
    if model is None:
        print("Pipeline: ML model not found — using deterministic fallback")
        return rank_options(scenario.emergency_type, feasible_pairs, top_k=top_k)

    try:
        rows = [
            _build_ml_row(scenario, a, f, z)
            for a, f, z in feasible_pairs
        ]
        df = pd.DataFrame(rows)
        df["distance_rank"] = (
            df["distance_km"].rank(method="min", ascending=True).astype(int)
        )
        df["range_coverage_ratio"] = (
            df["usable_range_km"] / df["distance_km"]
        ).round(4)
        df["runway_rank"] = (
            df["runway_length_m"].rank(method="min", ascending=False).astype(int)
        )
        df = df[ML_FEATURE_COLS]
        probs = model.predict_proba(df)[:, 1]

        scored = sorted(
            [(a, f, z, float(p)) for (a, f, z), p in zip(feasible_pairs, probs)],
            key=lambda x: x[3], reverse=True,
        )
        return [
            RankedOption(airport=a, features=f, score=p, distance_zone=z)
            for a, f, z, p in scored[:top_k]
        ]

    except Exception as exc:
        print(f"Pipeline: ML inference failed ({exc}) — using deterministic fallback")
        return rank_options(scenario.emergency_type, feasible_pairs, top_k=top_k)