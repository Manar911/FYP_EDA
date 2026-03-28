"""
scenario_generator.py

Scenario generation for Increment 2 dataset construction in the EDA Core Engine.

This module is responsible for:
- Generating realistic emergency diversion scenarios for supervised learning
- Selecting aircraft profiles and deriving runway requirements in a logically consistent order
- Assigning emergency types and diversion range limits according to defined generation policy
- Placing aircraft positions near real airports to preserve geographic plausibility
- Producing structured scenario records for downstream evaluation by the deterministic pipeline

The generator enforces realism and policy consistency so that the training dataset
reflects plausible aviation diversion conditions and remains aligned with the same
decision logic used by the operational EDA system.
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass, asdict
from importlib import resources
from typing import Iterable, List, Optional


# ============================================================
# Aircraft policy
# ============================================================

@dataclass(frozen=True)
class AircraftProfile:
    aircraft_type: str
    aircraft_category: str
    runway_min_m: int
    runway_max_m: int


AIRCRAFT_PROFILES: List[AircraftProfile] = [
    # Regional jets
    AircraftProfile("E170", "regional_jet", 1600, 2000),
    AircraftProfile("E175", "regional_jet", 1700, 2050),
    AircraftProfile("E190", "regional_jet", 1800, 2150),
    AircraftProfile("CRJ900", "regional_jet", 1700, 2100),

    # Narrow-body
    AircraftProfile("A220-300", "narrow_body", 2000, 2350),
    AircraftProfile("A319", "narrow_body", 2000, 2350),
    AircraftProfile("A320", "narrow_body", 2200, 2600),
    AircraftProfile("A321", "narrow_body", 2300, 2700),
    AircraftProfile("B737-800", "narrow_body", 2200, 2600),
    AircraftProfile("B737 MAX 8", "narrow_body", 2300, 2650),

    # Wide-body
    AircraftProfile("A330-300", "wide_body", 2600, 3100),
    AircraftProfile("A350-900", "wide_body", 2800, 3300),
    AircraftProfile("B767-300ER", "wide_body", 2600, 3050),
    AircraftProfile("B777-300ER", "wide_body", 2800, 3400),
    AircraftProfile("B787-9", "wide_body", 2700, 3200),

    # Very large
    AircraftProfile("A380-800", "very_large", 3200, 3800),
    AircraftProfile("B747-8", "very_large", 3200, 3700),
]

CATEGORY_WEIGHTS = {
    "regional_jet": 0.20,
    "narrow_body": 0.45,
    "wide_body": 0.28,
    "very_large": 0.07,
}


# ============================================================
# Emergency policy
# ============================================================

EMERGENCY_TYPES = [
    "medical",
    "technical",
    "mechanical",
    "weather",
    "security",
    "fuel",
    "operational_constraints",
]

EMERGENCY_WEIGHTS = {
    "medical": 0.18,
    "technical": 0.18,
    "mechanical": 0.16,
    "weather": 0.14,
    "security": 0.10,
    "fuel": 0.14,
    "operational_constraints": 0.10,
}

CRITICAL_EMERGENCIES = {
    "medical",
    "technical",
    "mechanical",
    "security",
    "fuel",
}

NON_CRITICAL_EMERGENCIES = {
    "weather",
    "operational_constraints",
}

EMERGENCY_RANGE_KM = {
    "fuel": (200, 650),
    "medical": (250, 850),
    "technical": (300, 900),
    "mechanical": (300, 900),
    "security": (300, 950),
    "weather": (350, 1000),
    "operational_constraints": (350, 950),
}


# ============================================================
# Scenario templates for variation / balance
# ============================================================

SCENARIO_TEMPLATES = (
    "balanced",
    "short_range",
    "tight_runway",
    "competitive",
    "critical_pressure",
)

TEMPLATE_WEIGHTS = {
    "balanced": 0.45,
    "short_range": 0.12,
    "tight_runway": 0.18,
    "competitive": 0.15,
    "critical_pressure": 0.10,
}


# ============================================================
# Airport row model
# ============================================================

@dataclass(frozen=True)
class AirportRow:
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
    has_ils: int
    has_medical: int
    medical_level: str
    has_rescue: int
    rescue_category: str
    has_firefighting: int
    has_maintenance: int
    fuel_available: int
    open_24h: int
    is_international: int
    tower_available: int
    weather_reporting: int
    closure_status: str
    restricted_status: str
    unsafe_status: str
    civil_military: str
    slot_restricted: int


# ============================================================
# Generated scenario model
# ============================================================

@dataclass(frozen=True)
class GeneratedScenario:
    scenario_id: str
    aircraft_lat: float
    aircraft_lon: float
    aircraft_type: str
    aircraft_category: str
    required_runway_m: int
    emergency_type: str
    max_range_km: int
    seed_airport_icao: str

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================
# CSV loading
# ============================================================

_REQUIRED_AIRPORT_COLUMNS = {
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


def _read_airports_from_package_csv(filename: str = "airports.csv") -> List[AirportRow]:
    """
    Reads airports from src/eda/data/airports.csv using importlib.resources.
    This keeps the generator aligned with your packaged project structure.
    """
    package = "eda.data"

    with resources.files(package).joinpath(filename).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError("airports.csv is empty or missing a header row.")

        missing = _REQUIRED_AIRPORT_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"airports.csv missing required columns: {sorted(missing)}")

        airports: List[AirportRow] = []
        for row in reader:
            airports.append(
                AirportRow(
                    icao=row["icao"].strip(),
                    iata=row["iata"].strip(),
                    name=row["name"].strip(),
                    city=row["city"].strip(),
                    country=row["country"].strip(),
                    lat=float(row["lat"]),
                    lon=float(row["lon"]),
                    elevation_ft=int(float(row["elevation_ft"])),
                    runway_length_m=int(float(row["runway_length_m"])),
                    runway_width_m=int(float(row["runway_width_m"])),
                    surface_type=row["surface_type"].strip(),
                    approach_type=row["approach_type"].strip(),
                    has_ils=int(row["has_ils"]),
                    has_medical=int(row["has_medical"]),
                    medical_level=row["medical_level"].strip(),
                    has_rescue=int(row["has_rescue"]),
                    rescue_category=row["rescue_category"].strip(),
                    has_firefighting=int(row["has_firefighting"]),
                    has_maintenance=int(row["has_maintenance"]),
                    fuel_available=int(row["fuel_available"]),
                    open_24h=int(row["open_24h"]),
                    is_international=int(row["is_international"]),
                    tower_available=int(row["tower_available"]),
                    weather_reporting=int(row["weather_reporting"]),
                    closure_status=row["closure_status"].strip(),
                    restricted_status=row["restricted_status"].strip(),
                    unsafe_status=row["unsafe_status"].strip(),
                    civil_military=row["civil_military"].strip(),
                    slot_restricted=int(row["slot_restricted"]),
                )
            )

    if not airports:
        raise ValueError("No airports were loaded from airports.csv.")

    return airports


# ============================================================
# Geometry helpers
# ============================================================

EARTH_RADIUS_KM = 6371.0


def _destination_point(lat_deg: float, lon_deg: float, distance_km: float, bearing_deg: float) -> tuple[float, float]:
    """
    Computes a destination point from a start point, distance, and bearing.
    """
    lat1 = math.radians(lat_deg)
    lon1 = math.radians(lon_deg)
    bearing = math.radians(bearing_deg)
    angular_distance = distance_km / EARTH_RADIUS_KM

    lat2 = math.asin(
        math.sin(lat1) * math.cos(angular_distance)
        + math.cos(lat1) * math.sin(angular_distance) * math.cos(bearing)
    )

    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(angular_distance) * math.cos(lat1),
        math.cos(angular_distance) - math.sin(lat1) * math.sin(lat2),
    )

    lon2_deg = (math.degrees(lon2) + 540.0) % 360.0 - 180.0
    lat2_deg = math.degrees(lat2)

    return lat2_deg, lon2_deg


# ============================================================
# Selection helpers
# ============================================================

def _weighted_choice(items: List[str], weights_map: dict[str, float], rng: random.Random) -> str:
    weights = [weights_map[item] for item in items]
    return rng.choices(items, weights=weights, k=1)[0]


def _choose_aircraft_profile(rng: random.Random, template: str) -> AircraftProfile:
    """
    Template can bias aircraft selection to improve dataset variation.
    """
    if template == "tight_runway":
        preferred_categories = ["wide_body", "very_large", "narrow_body"]
        preferred_weights = [0.40, 0.25, 0.35]
        chosen_category = rng.choices(preferred_categories, weights=preferred_weights, k=1)[0]
    elif template == "short_range":
        preferred_categories = ["regional_jet", "narrow_body", "wide_body"]
        preferred_weights = [0.35, 0.45, 0.20]
        chosen_category = rng.choices(preferred_categories, weights=preferred_weights, k=1)[0]
    else:
        chosen_category = _weighted_choice(list(CATEGORY_WEIGHTS.keys()), CATEGORY_WEIGHTS, rng)

    candidates = [p for p in AIRCRAFT_PROFILES if p.aircraft_category == chosen_category]
    return rng.choice(candidates)


def _choose_emergency_type(rng: random.Random, template: str) -> str:
    if template == "critical_pressure":
        critical = list(CRITICAL_EMERGENCIES)
        return rng.choice(critical)

    return _weighted_choice(EMERGENCY_TYPES, EMERGENCY_WEIGHTS, rng)


def _choose_required_runway_m(profile: AircraftProfile, rng: random.Random, template: str) -> int:
    span = profile.runway_max_m - profile.runway_min_m

    if template == "tight_runway":
        # Still hard, but not always near-extreme.
        low = int(profile.runway_min_m + 0.45 * span)
        high = int(profile.runway_min_m + 0.85 * span)
        return rng.randint(low, high)

    if template == "competitive":
        # Mid-range values help produce multiple feasible competitors.
        low = int(profile.runway_min_m + 0.25 * span)
        high = int(profile.runway_min_m + 0.65 * span)
        return rng.randint(low, high)

    if template == "short_range":
        # Slightly easier runway requirement to avoid double-hard scenarios.
        low = profile.runway_min_m
        high = int(profile.runway_min_m + 0.55 * span)
        return rng.randint(low, high)

    return rng.randint(profile.runway_min_m, int(profile.runway_min_m + 0.75 * span))


def _choose_max_range_km(emergency_type: str, rng: random.Random, template: str) -> int:
    min_km, max_km = EMERGENCY_RANGE_KM[emergency_type]

    if template == "short_range":
        upper = max(min_km, min_km + int(0.45 * (max_km - min_km)))
        return rng.randint(min_km, upper)

    if template == "competitive":
        # Bias toward mid-to-high range so several airports may compete.
        lower = min_km + int(0.35 * (max_km - min_km))
        return rng.randint(lower, max_km)

    return rng.randint(min_km, max_km)


def _choose_seed_airport(airports: List[AirportRow], rng: random.Random) -> AirportRow:
    """
    Seed airport should be a real airport with valid coordinates.
    Bias away from hard-closed/unsafe seeds to keep aircraft positions plausible.
    """
    candidates = [
        a for a in airports
        if a.closure_status == "open"
        and a.unsafe_status != "unsafe"
    ]

    if not candidates:
        raise ValueError("No suitable seed airports available for scenario generation.")

    return rng.choice(candidates)


def _choose_aircraft_position(seed_airport: AirportRow, rng: random.Random, template: str) -> tuple[float, float]:
    """
    Generate aircraft within a realistic radius of the seed airport.
    Tuned to improve feasible-airport density without becoming unrealistic.
    """
    if template == "short_range":
        radius_km = rng.uniform(50, 180)
    elif template == "competitive":
        radius_km = rng.uniform(80, 250)
    elif template == "critical_pressure":
        radius_km = rng.uniform(80, 320)
    elif template == "tight_runway":
        radius_km = rng.uniform(70, 280)
    else:
        radius_km = rng.uniform(50, 350)

    bearing_deg = rng.uniform(0, 360)
    return _destination_point(seed_airport.lat, seed_airport.lon, radius_km, bearing_deg)


def _choose_template(rng: random.Random) -> str:
    return _weighted_choice(list(TEMPLATE_WEIGHTS.keys()), TEMPLATE_WEIGHTS, rng)


# ============================================================
# Public API
# ============================================================

class ScenarioGenerator:
    """
    Generates realistic diversion scenarios using your airport DB.

    Core realism rules:
    - chooses a real airport first
    - places aircraft near that airport within a controlled radius
    - aircraft type chosen before runway requirement
    - max_range_km depends on emergency type
    """

    def __init__(
        self,
        airports: Optional[List[AirportRow]] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.airports = airports if airports is not None else _read_airports_from_package_csv()
        self.rng = random.Random(seed)

    def generate_one(
        self,
        scenario_index: int,
        template: Optional[str] = None,
    ) -> GeneratedScenario:
        """
        Generate one scenario.

        scenario_index:
            used to create scenario_id like S00001
        template:
            optional scenario style:
            - balanced
            - short_range
            - tight_runway
            - competitive
            - critical_pressure
        """
        chosen_template = template or _choose_template(self.rng)
        if chosen_template not in SCENARIO_TEMPLATES:
            raise ValueError(f"Unknown template '{chosen_template}'. Must be one of {SCENARIO_TEMPLATES}.")

        seed_airport = _choose_seed_airport(self.airports, self.rng)
        aircraft_profile = _choose_aircraft_profile(self.rng, chosen_template)
        emergency_type = _choose_emergency_type(self.rng, chosen_template)
        required_runway_m = _choose_required_runway_m(aircraft_profile, self.rng, chosen_template)
        max_range_km = _choose_max_range_km(emergency_type, self.rng, chosen_template)
        aircraft_lat, aircraft_lon = _choose_aircraft_position(seed_airport, self.rng, chosen_template)

        return GeneratedScenario(
            scenario_id=f"S{scenario_index:05d}",
            aircraft_lat=round(aircraft_lat, 6),
            aircraft_lon=round(aircraft_lon, 6),
            aircraft_type=aircraft_profile.aircraft_type,
            aircraft_category=aircraft_profile.aircraft_category,
            required_runway_m=required_runway_m,
            emergency_type=emergency_type,
            max_range_km=max_range_km,
            seed_airport_icao=seed_airport.icao,
        )

    def generate_many(
        self,
        count: int,
        start_index: int = 1,
        templates: Optional[Iterable[str]] = None,
    ) -> List[GeneratedScenario]:
        """
        Generate multiple scenarios.

        If templates is provided, it should contain one template per scenario.
        """
        if count <= 0:
            raise ValueError("count must be > 0")

        scenarios: List[GeneratedScenario] = []

        if templates is None:
            for i in range(count):
                scenarios.append(self.generate_one(start_index + i))
            return scenarios

        templates_list = list(templates)
        if len(templates_list) != count:
            raise ValueError("If templates is provided, its length must equal count.")

        for i, template in enumerate(templates_list):
            scenarios.append(self.generate_one(start_index + i, template=template))

        return scenarios


# ============================================================
# Convenience functions
# ============================================================

def generate_scenarios(
    count: int,
    start_index: int = 1,
    seed: Optional[int] = None,
) -> List[dict]:
    """
    Convenience function that returns plain dictionaries.
    Useful for dataset_builder.py later.
    """
    generator = ScenarioGenerator(seed=seed)
    return [scenario.to_dict() for scenario in generator.generate_many(count=count, start_index=start_index)]


def preview_scenarios(count: int = 5, seed: int = 42) -> None:
    """
    Small debug helper for manual inspection.
    """
    scenarios = generate_scenarios(count=count, seed=seed)
    for s in scenarios:
        print(s)