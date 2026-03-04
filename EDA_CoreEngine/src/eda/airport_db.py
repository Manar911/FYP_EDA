"""
airport_db.py

Airport data loading and integrity validation for the EDA Core Engine (Increment 1).

This module is responsible for:
- Loading the packaged airport dataset (airports.csv)
- Validating structural and data integrity constraints
- Producing immutable Airport objects for deterministic processing

The loader enforces strict schema, range, and uniqueness checks to prevent
corrupted or unsafe data from entering the decision pipeline.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from importlib import resources
from typing import List


@dataclass(frozen=True)
class Airport:
    icao: str
    name: str
    lat: float
    lon: float
    runway_length_m: int
    has_medical: bool
    has_rescue: bool
    approach_type: str
    country: str


REQUIRED_COLUMNS = {
    "icao",
    "name",
    "lat",
    "lon",
    "runway_length_m",
    "has_medical",
    "has_rescue",
    "approach_type",
    "country",
}


class AirportDbError(ValueError):
    pass


def _parse_bool_01(value: str, field: str, row_num: int) -> bool:
    v = value.strip()
    if v not in {"0", "1"}:
        raise AirportDbError(f"Row {row_num}: {field} must be 0/1, got '{value}'")
    return v == "1"


def load_airports() -> List[Airport]:
    """
    Loads airports from the packaged CSV file: eda/data/airports.csv
    This avoids path issues when running tests from different working directories.
    """
    try:
        csv_path = resources.files("eda").joinpath("data/airports.csv")
    except Exception as e:
        raise AirportDbError("Could not locate eda/data/airports.csv") from e

    if not csv_path.is_file():
        raise AirportDbError(f"Airport CSV not found at: {csv_path}")

    airports: List[Airport] = []
    seen: set[str] = set()

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise AirportDbError("CSV has no header row.")

        header = {h.strip() for h in reader.fieldnames}
        missing = REQUIRED_COLUMNS - header
        if missing:
            raise AirportDbError(f"Missing columns: {sorted(missing)}")

        for row_num, row in enumerate(reader, start=2):
            icao = row["icao"].strip().upper()
            if len(icao) != 4:
                raise AirportDbError(f"Row {row_num}: invalid ICAO '{icao}'")

            if icao in seen:
                raise AirportDbError(f"Row {row_num}: duplicate ICAO '{icao}'")
            seen.add(icao)

            name = row["name"].strip()
            lat = float(row["lat"])
            lon = float(row["lon"])
            runway_length_m = int(float(row["runway_length_m"]))

            if not (-90 <= lat <= 90):
                raise AirportDbError(f"Row {row_num}: lat out of range {lat}")
            if not (-180 <= lon <= 180):
                raise AirportDbError(f"Row {row_num}: lon out of range {lon}")
            if runway_length_m <= 0:
                raise AirportDbError(f"Row {row_num}: runway_length_m must be > 0")

            has_medical = _parse_bool_01(row["has_medical"], "has_medical", row_num)
            has_rescue = _parse_bool_01(row["has_rescue"], "has_rescue", row_num)

            airports.append(
                Airport(
                    icao=icao,
                    name=name,
                    lat=lat,
                    lon=lon,
                    runway_length_m=runway_length_m,
                    has_medical=has_medical,
                    has_rescue=has_rescue,
                    approach_type=row["approach_type"].strip(),
                    country=row["country"].strip().upper(),
                )
            )

    if not airports:
        raise AirportDbError("CSV contains no data rows.")

    return airports