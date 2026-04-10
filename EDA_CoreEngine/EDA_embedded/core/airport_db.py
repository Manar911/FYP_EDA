"""
airport_db.py

Airport data loading and integrity validation for the EDA Core Engine.

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
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class Airport:
    icao: str
    iata: str
    name: str
    city: str
    country: str
    lat: float
    lon: float
    elevation_ft: int
    runway_length_m: int
    runway_width_m: int
    surface_type: str
    approach_type: str
    has_ils: bool
    has_medical: bool
    medical_level: str
    has_rescue: bool
    rescue_category: str
    has_firefighting: bool
    has_maintenance: bool
    fuel_available: bool
    open_24h: bool
    is_international: bool
    tower_available: bool
    weather_reporting: bool
    closure_status: str
    restricted_status: str
    unsafe_status: str
    civil_military: str
    slot_restricted: bool


REQUIRED_COLUMNS = {
    "icao",
    "iata",
    "name",
    "city",
    "country",
    "lat",
    "lon",
    "elevation_ft",
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
}


class AirportDbError(ValueError):
    pass


def _parse_bool_01(value: str, field: str, row_num: int) -> bool:
    v = value.strip()
    if v not in {"0", "1"}:
        raise AirportDbError(f"Row {row_num}: {field} must be 0/1, got '{value}'")
    return v == "1"


def _parse_int(value: str, field: str, row_num: int) -> int:
    try:
        return int(float(value.strip()))
    except ValueError as e:
        raise AirportDbError(f"Row {row_num}: {field} must be numeric, got '{value}'") from e


def _parse_float(value: str, field: str, row_num: int) -> float:
    try:
        return float(value.strip())
    except ValueError as e:
        raise AirportDbError(f"Row {row_num}: {field} must be numeric, got '{value}'") from e


def load_airports() -> List[Airport]:
    """
    Loads airports from the packaged CSV file: eda/data/airports.csv
    This avoids path issues when running tests from different working directories.
    """
    try:
        csv_path = Path(__file__).resolve().parent / "data" / "airports.csv"
    except Exception as e:
        raise AirportDbError("Could not locate eda/data/airports.csv") from e

    if not csv_path.is_file():
        raise AirportDbError(f"Airport CSV not found at: {csv_path}")

    airports: List[Airport] = []
    seen: set[str] = set()

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise AirportDbError("CSV has no header row.")

        header = {h.strip() for h in reader.fieldnames}
        missing = REQUIRED_COLUMNS - header
        if missing:
            raise AirportDbError(f"Missing columns: {sorted(missing)}")

        for row_num, row in enumerate(reader, start=2):
            icao = row["icao"].strip().upper()
            iata = row["iata"].strip().upper()
            name = row["name"].strip()
            city = row["city"].strip()
            country = row["country"].strip()

            if len(icao) != 4:
                raise AirportDbError(f"Row {row_num}: invalid ICAO '{icao}'")

            if icao in seen:
                raise AirportDbError(f"Row {row_num}: duplicate ICAO '{icao}'")
            seen.add(icao)

            lat = _parse_float(row["lat"], "lat", row_num)
            lon = _parse_float(row["lon"], "lon", row_num)
            elevation_ft = _parse_int(row["elevation_ft"], "elevation_ft", row_num)
            runway_length_m = _parse_int(row["runway_length_m"], "runway_length_m", row_num)
            runway_width_m = _parse_int(row["runway_width_m"], "runway_width_m", row_num)

            if not (-90 <= lat <= 90):
                raise AirportDbError(f"Row {row_num}: lat out of range {lat}")
            if not (-180 <= lon <= 180):
                raise AirportDbError(f"Row {row_num}: lon out of range {lon}")
            if runway_length_m <= 0:
                raise AirportDbError(f"Row {row_num}: runway_length_m must be > 0")
            if runway_width_m <= 0:
                raise AirportDbError(f"Row {row_num}: runway_width_m must be > 0")

            surface_type = row["surface_type"].strip().lower()
            approach_type = row["approach_type"].strip().upper()
            medical_level = row["medical_level"].strip().lower()
            rescue_category = row["rescue_category"].strip().lower()
            closure_status = row["closure_status"].strip().lower()
            restricted_status = row["restricted_status"].strip().lower()
            unsafe_status = row["unsafe_status"].strip().lower()
            civil_military = row["civil_military"].strip().lower()

            has_ils = _parse_bool_01(row["has_ils"], "has_ils", row_num)
            has_medical = _parse_bool_01(row["has_medical"], "has_medical", row_num)
            has_rescue = _parse_bool_01(row["has_rescue"], "has_rescue", row_num)
            has_firefighting = _parse_bool_01(row["has_firefighting"], "has_firefighting", row_num)
            has_maintenance = _parse_bool_01(row["has_maintenance"], "has_maintenance", row_num)
            fuel_available = _parse_bool_01(row["fuel_available"], "fuel_available", row_num)
            open_24h = _parse_bool_01(row["open_24h"], "open_24h", row_num)
            is_international = _parse_bool_01(row["is_international"], "is_international", row_num)
            tower_available = _parse_bool_01(row["tower_available"], "tower_available", row_num)
            weather_reporting = _parse_bool_01(row["weather_reporting"], "weather_reporting", row_num)
            slot_restricted = _parse_bool_01(row["slot_restricted"], "slot_restricted", row_num)

            airports.append(
                Airport(
                    icao=icao,
                    iata=iata,
                    name=name,
                    city=city,
                    country=country,
                    lat=lat,
                    lon=lon,
                    elevation_ft=elevation_ft,
                    runway_length_m=runway_length_m,
                    runway_width_m=runway_width_m,
                    surface_type=surface_type,
                    approach_type=approach_type,
                    has_ils=has_ils,
                    has_medical=has_medical,
                    medical_level=medical_level,
                    has_rescue=has_rescue,
                    rescue_category=rescue_category,
                    has_firefighting=has_firefighting,
                    has_maintenance=has_maintenance,
                    fuel_available=fuel_available,
                    open_24h=open_24h,
                    is_international=is_international,
                    tower_available=tower_available,
                    weather_reporting=weather_reporting,
                    closure_status=closure_status,
                    restricted_status=restricted_status,
                    unsafe_status=unsafe_status,
                    civil_military=civil_military,
                    slot_restricted=slot_restricted,
                )
            )

    if not airports:
        raise AirportDbError("CSV contains no data rows.")

    return airports