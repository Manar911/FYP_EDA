"""
dataset_builder.py

ML dataset construction for Increment 2 of the EDA Core Engine.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, List, Sequence

from eda.models import DecisionReport
from eda.pipeline import run_pipeline
from eda.ranking import rank_options
from eda.scenario import EmergencyType, Scenario, FuelState, BindingSide
from eda.scenario_generator import GeneratedScenario, ScenarioGenerator


DATASET_COLUMNS = [
    "scenario_id",
    "seed_airport_icao",
    "aircraft_lat",
    "aircraft_lon",
    "aircraft_type",
    "aircraft_category",
    "required_runway_m",
    "emergency_type",
    "max_range_km",
    "fuel_state",
    "fuel_multiplier",
    "aircraft_adjusted_range_km",
    "usable_range_km",
    "extended_range_km",
    "binding_side",
    "airport_icao",
    "airport_iata",
    "airport_name",
    "airport_city",
    "airport_country",
    "runway_length_m",
    "runway_width_m",
    "surface_type",
    "approach_type",
    "has_ils",
    "has_medical",
    "medical_level",
    "has_rescue",
    "rescue_category",
    "has_firefighting",
    "has_maintenance",
    "fuel_available",
    "open_24h",
    "is_international",
    "tower_available",
    "weather_reporting",
    "closure_status",
    "restricted_status",
    "unsafe_status",
    "civil_military",
    "slot_restricted",
    "distance_km",
    "runway_margin_m",
    "feasible",
    "feasibility_reason",
    "distance_zone",
    "baseline_score",
    "target_rank",
    "is_top_choice",
    "split",
]


def _bool_to_int(value: bool) -> int:
    return 1 if bool(value) else 0


def _generated_to_runtime_scenario(g: GeneratedScenario) -> Scenario:
    return Scenario(
        aircraft_lat=g.aircraft_lat,
        aircraft_lon=g.aircraft_lon,
        required_runway_m=g.required_runway_m,
        emergency_type=EmergencyType(g.emergency_type),
        aircraft_type=g.aircraft_type,
        fuel_state=FuelState(g.fuel_state),
        fuel_multiplier=g.fuel_multiplier,
        max_range_km=g.max_range_km,
        aircraft_adjusted_range_km=g.aircraft_adjusted_range_km,
        usable_range_km=g.usable_range_km,
        extended_range_km=g.extended_range_km,
        binding_side=BindingSide(g.binding_side),
    )


def _build_rank_maps(
    report: DecisionReport,
) -> tuple[Dict[str, float], Dict[str, int], Dict[str, int]]:
    full_ranked = rank_options(
        report.scenario.emergency_type,
        report.feasible,
        top_k=len(report.feasible),
    )

    score_map: Dict[str, float] = {}
    rank_map: Dict[str, int] = {}
    top_choice_map: Dict[str, int] = {}

    for idx, ranked in enumerate(full_ranked, start=1):
        icao = ranked.airport.icao
        score_map[icao] = float(ranked.score)
        rank_map[icao] = idx
        top_choice_map[icao] = 1 if idx == 1 else 0

    return score_map, rank_map, top_choice_map


def _build_rows_for_scenario(
    generated: GeneratedScenario,
    report: DecisionReport,
) -> List[dict]:
    score_map, rank_map, top_choice_map = _build_rank_maps(report)

    rows: List[dict] = []

    for evaluated in report.evaluated:
        airport = evaluated.airport
        feats = evaluated.features
        feas = evaluated.feasibility

        icao = airport.icao

        row = {
            "scenario_id": generated.scenario_id,
            "seed_airport_icao": generated.seed_airport_icao,
            "aircraft_lat": generated.aircraft_lat,
            "aircraft_lon": generated.aircraft_lon,
            "aircraft_type": generated.aircraft_type,
            "aircraft_category": generated.aircraft_category,
            "required_runway_m": generated.required_runway_m,
            "emergency_type": generated.emergency_type,
            "max_range_km": generated.max_range_km,
            "fuel_state": generated.fuel_state,
            "fuel_multiplier": generated.fuel_multiplier,
            "aircraft_adjusted_range_km": generated.aircraft_adjusted_range_km,
            "usable_range_km": generated.usable_range_km,
            "extended_range_km": generated.extended_range_km,
            "binding_side": generated.binding_side,
            "airport_icao": airport.icao,
            "airport_iata": airport.iata,
            "airport_name": airport.name,
            "airport_city": airport.city,
            "airport_country": airport.country,
            "runway_length_m": airport.runway_length_m,
            "runway_width_m": airport.runway_width_m,
            "surface_type": airport.surface_type,
            "approach_type": airport.approach_type,
            "has_ils": _bool_to_int(airport.has_ils),
            "has_medical": _bool_to_int(airport.has_medical),
            "medical_level": airport.medical_level,
            "has_rescue": _bool_to_int(airport.has_rescue),
            "rescue_category": airport.rescue_category,
            "has_firefighting": _bool_to_int(airport.has_firefighting),
            "has_maintenance": _bool_to_int(airport.has_maintenance),
            "fuel_available": _bool_to_int(airport.fuel_available),
            "open_24h": _bool_to_int(airport.open_24h),
            "is_international": _bool_to_int(airport.is_international),
            "tower_available": _bool_to_int(airport.tower_available),
            "weather_reporting": _bool_to_int(airport.weather_reporting),
            "closure_status": airport.closure_status,
            "restricted_status": airport.restricted_status,
            "unsafe_status": airport.unsafe_status,
            "civil_military": airport.civil_military,
            "slot_restricted": _bool_to_int(airport.slot_restricted),
            "distance_km": round(float(feats.distance_km), 6),
            "runway_margin_m": int(feats.runway_margin_m),
            "feasible": _bool_to_int(feas.feasible),
            "feasibility_reason": feas.reason,
            "distance_zone": feas.zone,
            "baseline_score": score_map.get(icao, None),
            "target_rank": rank_map.get(icao, None),
            "is_top_choice": top_choice_map.get(icao, 0),
            "split": "",
        }

        rows.append(row)

    return rows


def _split_scenarios(
    scenario_ids: Sequence[str],
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[set[str], set[str], set[str]]:
    if not scenario_ids:
        raise ValueError("No scenario_ids provided for splitting.")

    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    unique_ids = sorted(set(scenario_ids))
    rng = random.Random(seed)
    rng.shuffle(unique_ids)

    n = len(unique_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(
            f"Split sizes are invalid for {n} scenarios. "
            f"Got train={n_train}, val={n_val}, test={n_test}."
        )

    train_ids = set(unique_ids[:n_train])
    val_ids = set(unique_ids[n_train:n_train + n_val])
    test_ids = set(unique_ids[n_train + n_val:])

    return train_ids, val_ids, test_ids


def _assign_split_labels(rows: List[dict], *, split_seed: int = 42) -> None:
    scenario_ids = [row["scenario_id"] for row in rows]
    train_ids, val_ids, test_ids = _split_scenarios(scenario_ids, seed=split_seed)

    for row in rows:
        sid = row["scenario_id"]

        if sid in train_ids:
            row["split"] = "train"
        elif sid in val_ids:
            row["split"] = "val"
        elif sid in test_ids:
            row["split"] = "test"
        else:
            raise RuntimeError(f"Scenario ID {sid} was not assigned to any split.")


def validate_no_leakage(rows: Sequence[dict]) -> None:
    split_map: Dict[str, set[str]] = {}

    for row in rows:
        sid = row["scenario_id"]
        split = row["split"]
        split_map.setdefault(sid, set()).add(split)

    leaked = {sid: splits for sid, splits in split_map.items() if len(splits) > 1}
    if leaked:
        raise ValueError(f"Leakage detected: scenario_ids appear in multiple splits: {leaked}")


def validate_dataset_quality(rows: Sequence[dict]) -> None:
    if not rows:
        raise ValueError("Dataset is empty.")

    scenario_ids = {row["scenario_id"] for row in rows}
    if not scenario_ids:
        raise ValueError("No scenario IDs found in dataset.")

    splits = {row["split"] for row in rows}
    expected_splits = {"train", "val", "test"}
    if splits != expected_splits:
        raise ValueError(f"Expected splits {expected_splits}, but found {splits}")

    top_choice_count = sum(int(row["is_top_choice"]) for row in rows)
    if top_choice_count <= 0:
        raise ValueError("Dataset contains no positive top-choice labels.")

    feasible_count = sum(int(row["feasible"]) for row in rows)
    if feasible_count <= 0:
        raise ValueError("Dataset contains no feasible rows.")

    by_scenario_top_count: Dict[str, int] = {}
    for row in rows:
        sid = row["scenario_id"]
        by_scenario_top_count.setdefault(sid, 0)
        by_scenario_top_count[sid] += int(row["is_top_choice"])

    bad = {sid: c for sid, c in by_scenario_top_count.items() if c > 1}
    if bad:
        raise ValueError(f"Some scenarios have more than one top choice: {bad}")


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DATASET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _filter_rows_by_split(rows: Sequence[dict], split_name: str) -> List[dict]:
    return [row for row in rows if row["split"] == split_name]


def summarize_dataset(rows: Sequence[dict]) -> dict:
    scenario_ids = sorted({row["scenario_id"] for row in rows})
    train_ids = {row["scenario_id"] for row in rows if row["split"] == "train"}
    val_ids = {row["scenario_id"] for row in rows if row["split"] == "val"}
    test_ids = {row["scenario_id"] for row in rows if row["split"] == "test"}

    emergency_counts: Dict[str, int] = {}
    for row in rows:
        emergency = row["emergency_type"]
        emergency_counts[emergency] = emergency_counts.get(emergency, 0) + 1

    return {
        "total_rows": len(rows),
        "total_scenarios": len(scenario_ids),
        "train_scenarios": len(train_ids),
        "val_scenarios": len(val_ids),
        "test_scenarios": len(test_ids),
        "feasible_rows": sum(int(row["feasible"]) for row in rows),
        "top_choice_rows": sum(int(row["is_top_choice"]) for row in rows),
        "emergency_row_counts": emergency_counts,
    }


def build_dataset(
    *,
    scenario_count: int,
    scenario_seed: int = 42,
    split_seed: int = 42,
) -> List[dict]:
    generator = ScenarioGenerator(seed=scenario_seed)

    all_rows: List[dict] = []
    accepted_scenarios = 0
    scenario_index = 1

    while accepted_scenarios < scenario_count:
        generated = generator.generate_one(scenario_index)
        runtime_scenario = _generated_to_runtime_scenario(generated)

        report = run_pipeline(
            runtime_scenario,
            top_k=3,
            max_range_km=float(generated.max_range_km),
        )

        if len(report.feasible) == 0:
            scenario_index += 1
            continue

        scenario_rows = _build_rows_for_scenario(generated, report)
        all_rows.extend(scenario_rows)

        accepted_scenarios += 1
        scenario_index += 1

    _assign_split_labels(all_rows, split_seed=split_seed)
    validate_no_leakage(all_rows)
    validate_dataset_quality(all_rows)

    return all_rows


def build_and_save_dataset(
    *,
    scenario_count: int,
    output_dir: str = "generated_data",
    scenario_seed: int = 42,
    split_seed: int = 42,
) -> dict:
    rows = build_dataset(
        scenario_count=scenario_count,
        scenario_seed=scenario_seed,
        split_seed=split_seed,
    )

    out_dir = Path(output_dir)
    full_path = out_dir / "dataset_full.csv"
    train_path = out_dir / "train.csv"
    val_path = out_dir / "val.csv"
    test_path = out_dir / "test.csv"

    train_rows = _filter_rows_by_split(rows, "train")
    val_rows = _filter_rows_by_split(rows, "val")
    test_rows = _filter_rows_by_split(rows, "test")

    _write_csv(full_path, rows)
    _write_csv(train_path, train_rows)
    _write_csv(val_path, val_rows)
    _write_csv(test_path, test_rows)

    summary = summarize_dataset(rows)
    summary["output_dir"] = str(out_dir)
    summary["dataset_full"] = str(full_path)
    summary["train_csv"] = str(train_path)
    summary["val_csv"] = str(val_path)
    summary["test_csv"] = str(test_path)

    return summary


if __name__ == "__main__":
    summary = build_and_save_dataset(
        scenario_count=80,
        output_dir="generated_data",
        scenario_seed=42,
        split_seed=42,
    )

    print("Dataset build complete.")
    for key, value in summary.items():
        print(f"{key}: {value}")