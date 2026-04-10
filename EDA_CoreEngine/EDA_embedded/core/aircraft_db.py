"""
aircraft_db.py

Aircraft profile loading and validation for the EDA Core Engine.

This module is responsible for:
- Loading the packaged aircraft dataset (aircraft_profiles.csv)
- Validating structural and data integrity constraints
- Producing immutable AircraftProfile objects for deterministic scenario generation

The aircraft database supports:
- structured aircraft selection
- consistent runway requirement generation
- future range and fuel abstraction logic
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class AircraftProfile:
    aircraft_type: str
    aircraft_category: str
    operator_class: str
    runway_min_m: int
    runway_max_m: int
    nominal_diversion_capability_km: int
    nominal_endurance_minutes: int
    selection_weight: int


REQUIRED_COLUMNS = {
    "aircraft_type",
    "aircraft_category",
    "operator_class",
    "runway_min_m",
    "runway_max_m",
    "nominal_diversion_capability_km",
    "nominal_endurance_minutes",
    "selection_weight",
}


ALLOWED_CATEGORIES = {
    "regional_jet",
    "narrow_body",
    "wide_body",
    "very_large",
}

ALLOWED_OPERATOR_CLASSES = {
    "civilian_commercial",
}


class AircraftDbError(ValueError):
    pass


def _parse_int(value: str, field: str, row_num: int) -> int:
    try:
        return int(float(value.strip()))
    except ValueError as e:
        raise AircraftDbError(
            f"Row {row_num}: {field} must be numeric, got '{value}'"
        ) from e


def load_aircraft_profiles() -> List[AircraftProfile]:
    """
    Loads aircraft profiles from the packaged CSV file:
    eda/data/aircraft_profiles.csv

    This avoids path issues when running tests from different working directories.
    """
    try:
        csv_path = Path(__file__).resolve().parent / "data" / "aircraft_profiles.csv"
    except Exception as e:
        raise AircraftDbError(
            "Could not locate eda/data/aircraft_profiles.csv"
        ) from e

    if not csv_path.is_file():
        raise AircraftDbError(f"Aircraft CSV not found at: {csv_path}")

    profiles: List[AircraftProfile] = []
    seen_types: set[str] = set()

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise AircraftDbError("CSV has no header row.")

        header = {h.strip() for h in reader.fieldnames}
        missing = REQUIRED_COLUMNS - header
        if missing:
            raise AircraftDbError(f"Missing columns: {sorted(missing)}")

        for row_num, row in enumerate(reader, start=2):
            aircraft_type = row["aircraft_type"].strip()
            aircraft_category = row["aircraft_category"].strip().lower()
            operator_class = row["operator_class"].strip().lower()

            if not aircraft_type:
                raise AircraftDbError(f"Row {row_num}: aircraft_type is empty")

            if aircraft_type in seen_types:
                raise AircraftDbError(
                    f"Row {row_num}: duplicate aircraft_type '{aircraft_type}'"
                )
            seen_types.add(aircraft_type)

            if aircraft_category not in ALLOWED_CATEGORIES:
                raise AircraftDbError(
                    f"Row {row_num}: invalid aircraft_category '{aircraft_category}'"
                )

            if operator_class not in ALLOWED_OPERATOR_CLASSES:
                raise AircraftDbError(
                    f"Row {row_num}: invalid operator_class '{operator_class}'"
                )

            runway_min_m = _parse_int(row["runway_min_m"], "runway_min_m", row_num)
            runway_max_m = _parse_int(row["runway_max_m"], "runway_max_m", row_num)
            nominal_diversion_capability_km = _parse_int(
                row["nominal_diversion_capability_km"],
                "nominal_diversion_capability_km",
                row_num,
            )
            nominal_endurance_minutes = _parse_int(
                row["nominal_endurance_minutes"],
                "nominal_endurance_minutes",
                row_num,
            )
            selection_weight = _parse_int(
                row["selection_weight"],
                "selection_weight",
                row_num,
            )

            if runway_min_m <= 0:
                raise AircraftDbError(
                    f"Row {row_num}: runway_min_m must be > 0"
                )

            if runway_max_m <= 0:
                raise AircraftDbError(
                    f"Row {row_num}: runway_max_m must be > 0"
                )

            if runway_min_m > runway_max_m:
                raise AircraftDbError(
                    f"Row {row_num}: runway_min_m cannot exceed runway_max_m"
                )

            if nominal_diversion_capability_km <= 0:
                raise AircraftDbError(
                    f"Row {row_num}: nominal_diversion_capability_km must be > 0"
                )

            if nominal_endurance_minutes <= 0:
                raise AircraftDbError(
                    f"Row {row_num}: nominal_endurance_minutes must be > 0"
                )

            if selection_weight <= 0:
                raise AircraftDbError(
                    f"Row {row_num}: selection_weight must be > 0"
                )

            profiles.append(
                AircraftProfile(
                    aircraft_type=aircraft_type,
                    aircraft_category=aircraft_category,
                    operator_class=operator_class,
                    runway_min_m=runway_min_m,
                    runway_max_m=runway_max_m,
                    nominal_diversion_capability_km=nominal_diversion_capability_km,
                    nominal_endurance_minutes=nominal_endurance_minutes,
                    selection_weight=selection_weight,
                )
            )

    if not profiles:
        raise AircraftDbError("CSV contains no data rows.")

    return profiles