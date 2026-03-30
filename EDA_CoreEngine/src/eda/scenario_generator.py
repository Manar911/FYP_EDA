"""
scenario_generator.py

Scenario generation for Increment 2 dataset construction in the EDA Core Engine.

This module is responsible for:
- Generating realistic emergency diversion scenarios for supervised learning
- Selecting aircraft profiles from the aircraft database and deriving runway requirements
  in a logically consistent order
- Assigning emergency types and diversion range limits according to defined generation policy
- Assigning fuel state and deriving the redesigned distance model
- Placing aircraft positions near real airports while preserving some harder,
  farther cases for realism
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

from eda.aircraft_db import AircraftProfile, load_aircraft_profiles


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
    "fuel": (275, 850),
    "medical": (325, 1000),
    "technical": (375, 1050),
    "mechanical": (375, 1050),
    "security": (375, 1100),
    "weather": (425, 1150),
    "operational_constraints": (425, 1100),
}


# ============================================================
# Fuel policy
# ============================================================

FUEL_STATES = [
    "critical",
    "low",
    "normal",
]

FUEL_MULTIPLIERS = {
    "critical": 0.75,
    "low": 0.90,
    "normal": 1.00,
}

FUEL_STATE_WEIGHTS = {
    "critical": 0.16,
    "low": 0.34,
    "normal": 0.50,
}


# ============================================================
# Endurance / extension policy
# ============================================================

def _lookup_endurance_extension_factor(nominal_endurance_minutes: int) -> float:
    """
    Returns the extension factor based on endurance class.

    Tuned version:
    - < 180 min   -> 1.25
    - 180–240 min -> 1.35
    - > 240 min   -> 1.45

    This remains more permissive than the earlier baseline so the extended zone
    becomes visible in the generated dataset.
    """
    if nominal_endurance_minutes < 180:
        return 1.25
    if nominal_endurance_minutes <= 240:
        return 1.35
    return 1.45


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
    "balanced": 0.38,
    "short_range": 0.10,
    "tight_runway": 0.14,
    "competitive": 0.28,
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

    # Emergency-envelope input
    max_range_km: int

    # Fuel / range redesign
    fuel_state: str
    fuel_multiplier: float
    aircraft_adjusted_range_km: float
    usable_range_km: float
    extended_range_km: float
    binding_side: str

    # Traceability
    nominal_diversion_capability_km: int
    nominal_endurance_minutes: int
    endurance_extension_factor: float

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


def _destination_point(
    lat_deg: float,
    lon_deg: float,
    distance_km: float,
    bearing_deg: float,
) -> tuple[float, float]:
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


def _choose_aircraft_profile(
    aircraft_profiles: List[AircraftProfile],
    rng: random.Random,
    template: str,
) -> AircraftProfile:
    """
    Select an aircraft profile from the aircraft DB, optionally biased by template.
    Uses the CSV-backed selection_weight field for realism.
    """
    if template == "tight_runway":
        allowed_categories = {"narrow_body", "wide_body", "very_large"}
        candidates = [p for p in aircraft_profiles if p.aircraft_category in allowed_categories]
    elif template == "short_range":
        allowed_categories = {"regional_jet", "narrow_body", "wide_body"}
        candidates = [p for p in aircraft_profiles if p.aircraft_category in allowed_categories]
    else:
        candidates = list(aircraft_profiles)

    if not candidates:
        raise ValueError("No aircraft profiles available after template filtering.")

    weights = [p.selection_weight for p in candidates]
    return rng.choices(candidates, weights=weights, k=1)[0]


def _choose_emergency_type(rng: random.Random, template: str) -> str:
    if template == "critical_pressure":
        critical = list(CRITICAL_EMERGENCIES)
        return rng.choice(critical)

    return _weighted_choice(EMERGENCY_TYPES, EMERGENCY_WEIGHTS, rng)


def _choose_fuel_state(rng: random.Random, template: str, emergency_type: str) -> str:
    """
    Selects fuel state with mild template-aware bias.
    This keeps the distribution realistic without making every critical template
    become a fuel-starved scenario.
    """
    if template == "critical_pressure":
        weights = {
            "critical": 0.28,
            "low": 0.40,
            "normal": 0.32,
        }
        return _weighted_choice(FUEL_STATES, weights, rng)

    if emergency_type == "fuel":
        weights = {
            "critical": 0.30,
            "low": 0.45,
            "normal": 0.25,
        }
        return _weighted_choice(FUEL_STATES, weights, rng)

    if template == "competitive":
        weights = {
            "critical": 0.08,
            "low": 0.26,
            "normal": 0.66,
        }
        return _weighted_choice(FUEL_STATES, weights, rng)

    return _weighted_choice(FUEL_STATES, FUEL_STATE_WEIGHTS, rng)


def _choose_required_runway_m(profile: AircraftProfile, rng: random.Random, template: str) -> int:
    """
    Slightly softened runway requirements to reduce over-constrained scenarios.
    """
    span = profile.runway_max_m - profile.runway_min_m

    if template == "tight_runway":
        low = int(profile.runway_min_m + 0.30 * span)
        high = int(profile.runway_min_m + 0.70 * span)
        return rng.randint(low, high)

    if template == "competitive":
        low = int(profile.runway_min_m + 0.10 * span)
        high = int(profile.runway_min_m + 0.45 * span)
        return rng.randint(low, high)

    if template == "short_range":
        low = profile.runway_min_m
        high = int(profile.runway_min_m + 0.35 * span)
        return rng.randint(low, high)

    return rng.randint(profile.runway_min_m, int(profile.runway_min_m + 0.55 * span))


def _choose_max_range_km(emergency_type: str, rng: random.Random, template: str) -> int:
    min_km, max_km = EMERGENCY_RANGE_KM[emergency_type]

    if template == "short_range":
        upper = max(min_km, min_km + int(0.60 * (max_km - min_km)))
        base = rng.randint(min_km, upper)
        return int(base * rng.uniform(1.05, 1.12))

    if template == "competitive":
        lower = min_km + int(0.50 * (max_km - min_km))
        base = rng.randint(lower, max_km)
        return int(base * rng.uniform(1.08, 1.15))

    if template == "critical_pressure":
        lower = min_km + int(0.25 * (max_km - min_km))
        upper = min_km + int(0.85 * (max_km - min_km))
        base = rng.randint(lower, upper)
        return int(base * rng.uniform(1.04, 1.10))

    base = rng.randint(min_km, max_km)
    return int(base * rng.uniform(1.05, 1.15))


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


def _choose_aircraft_position(
    seed_airport: AirportRow,
    rng: random.Random,
    template: str,
) -> tuple[float, float]:
    """
    Generate aircraft positions with a realistic density bias:
    - most scenarios are kept closer to airport clusters
    - some remain moderately spread
    - a small minority are intentionally harder / farther cases

    This keeps the dataset realistic without making it artificially easy.
    """
    if template == "short_range":
        base_radius = rng.uniform(20, 100)
    elif template == "competitive":
        base_radius = rng.uniform(25, 120)
    elif template == "critical_pressure":
        base_radius = rng.uniform(50, 180)
    elif template == "tight_runway":
        base_radius = rng.uniform(30, 140)
    else:
        base_radius = rng.uniform(20, 140)

    roll = rng.random()

    # 70%: closer to airport clusters
    if roll < 0.70:
        radius_km = base_radius * rng.uniform(0.55, 0.80)

    # 15%: keep as originally sampled
    elif roll < 0.85:
        radius_km = base_radius

    # 15%: deliberately harder / farther cases
    else:
        radius_km = base_radius * rng.uniform(1.15, 1.45)

    # Safety clamp so cases remain difficult but not absurd
    radius_km = min(radius_km, 220.0)

    bearing_deg = rng.uniform(0, 360)
    return _destination_point(seed_airport.lat, seed_airport.lon, radius_km, bearing_deg)


def _choose_template(rng: random.Random) -> str:
    return _weighted_choice(list(TEMPLATE_WEIGHTS.keys()), TEMPLATE_WEIGHTS, rng)


# ============================================================
# Range-model helpers
# ============================================================

def _compute_aircraft_adjusted_range_km(
    profile: AircraftProfile,
    fuel_state: str,
) -> float:
    return float(profile.nominal_diversion_capability_km) * FUEL_MULTIPLIERS[fuel_state]


def _compute_binding_side(
    max_range_km: int,
    aircraft_adjusted_range_km: float,
) -> str:
    if aircraft_adjusted_range_km <= float(max_range_km):
        return "aircraft_fuel"
    return "emergency"


def _compute_usable_range_km(
    max_range_km: int,
    aircraft_adjusted_range_km: float,
) -> float:
    """
    Usable range must remain the strict conservative min() rule.
    This is enforced by Scenario validation and is part of the core design.
    """
    return min(float(max_range_km), float(aircraft_adjusted_range_km))


def _compute_extended_range_km(
    usable_range_km: float,
    endurance_extension_factor: float,
    rng: random.Random,
    template: str,
) -> float:
    """
    Extended range is widened further so the extended zone becomes visible in
    the dataset and supports fallback ranking behaviour.

    Important:
    - usable_range_km remains strict min()
    - only the extended layer is widened
    """
    if template == "competitive":
        extra = rng.uniform(1.18, 1.34)
    elif template == "critical_pressure":
        extra = rng.uniform(1.12, 1.25)
    else:
        extra = rng.uniform(1.12, 1.28)

    return usable_range_km * endurance_extension_factor * extra


# ============================================================
# Public API
# ============================================================

class ScenarioGenerator:
    """
    Generates realistic diversion scenarios using your airport DB and aircraft DB.

    Core realism rules:
    - chooses a real airport first
    - places aircraft near that airport within a controlled radius
    - selects aircraft from the structured aircraft database
    - aircraft type chosen before runway requirement
    - max_range_km depends on emergency type
    - fuel state modifies aircraft diversion capability
    - final usable range is determined by interaction between:
        1) emergency max range
        2) aircraft/fuel adjusted capability
    """

    def __init__(
        self,
        airports: Optional[List[AirportRow]] = None,
        aircraft_profiles: Optional[List[AircraftProfile]] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.airports = airports if airports is not None else _read_airports_from_package_csv()
        self.aircraft_profiles = aircraft_profiles if aircraft_profiles is not None else load_aircraft_profiles()
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
            raise ValueError(
                f"Unknown template '{chosen_template}'. Must be one of {SCENARIO_TEMPLATES}."
            )

        seed_airport = _choose_seed_airport(self.airports, self.rng)
        aircraft_profile = _choose_aircraft_profile(self.aircraft_profiles, self.rng, chosen_template)
        emergency_type = _choose_emergency_type(self.rng, chosen_template)
        fuel_state = _choose_fuel_state(self.rng, chosen_template, emergency_type)
        required_runway_m = _choose_required_runway_m(aircraft_profile, self.rng, chosen_template)
        max_range_km = _choose_max_range_km(emergency_type, self.rng, chosen_template)
        aircraft_lat, aircraft_lon = _choose_aircraft_position(seed_airport, self.rng, chosen_template)

        fuel_multiplier = FUEL_MULTIPLIERS[fuel_state]
        aircraft_adjusted_range_km = _compute_aircraft_adjusted_range_km(aircraft_profile, fuel_state)

        usable_range_km = _compute_usable_range_km(
            max_range_km,
            aircraft_adjusted_range_km,
        )

        endurance_extension_factor = _lookup_endurance_extension_factor(
            aircraft_profile.nominal_endurance_minutes
        )

        extended_range_km = _compute_extended_range_km(
            usable_range_km,
            endurance_extension_factor,
            self.rng,
            chosen_template,
        )

        binding_side = _compute_binding_side(
            max_range_km,
            aircraft_adjusted_range_km,
        )

        return GeneratedScenario(
            scenario_id=f"S{scenario_index:05d}",
            aircraft_lat=round(aircraft_lat, 6),
            aircraft_lon=round(aircraft_lon, 6),
            aircraft_type=aircraft_profile.aircraft_type,
            aircraft_category=aircraft_profile.aircraft_category,
            required_runway_m=required_runway_m,
            emergency_type=emergency_type,
            max_range_km=max_range_km,
            fuel_state=fuel_state,
            fuel_multiplier=fuel_multiplier,
            aircraft_adjusted_range_km=round(aircraft_adjusted_range_km, 6),
            usable_range_km=round(usable_range_km, 6),
            extended_range_km=round(extended_range_km, 6),
            binding_side=binding_side,
            nominal_diversion_capability_km=aircraft_profile.nominal_diversion_capability_km,
            nominal_endurance_minutes=aircraft_profile.nominal_endurance_minutes,
            endurance_extension_factor=endurance_extension_factor,
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
    return [
        scenario.to_dict()
        for scenario in generator.generate_many(count=count, start_index=start_index)
    ]


def preview_scenarios(count: int = 5, seed: int = 42) -> None:
    """
    Small debug helper for manual inspection.
    """
    scenarios = generate_scenarios(count=count, seed=seed)
    for s in scenarios:
        print(s)