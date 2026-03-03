"""
features.py

Feature engineering for the EDA Core Engine (Increment 1).

This module converts raw inputs (Scenario + Airport) into deterministic numerical
features that can be:
- validated with unit tests
- used by the feasibility filter (hard constraints)
- used by the baseline ranking (weighted scoring)
"""

from __future__ import annotations

from dataclasses import dataclass
from math import asin, cos, radians, sin, sqrt

from eda.airport_db import Airport
from eda.scenario import Scenario


EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Great-circle distance between two points on Earth (in kilometers).
    Deterministic and widely used in aviation/navigation calculations.
    """
    lat1_r = radians(lat1)
    lon1_r = radians(lon1)
    lat2_r = radians(lat2)
    lon2_r = radians(lon2)

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = sin(dlat / 2) ** 2 + cos(lat1_r) * cos(lat2_r) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return EARTH_RADIUS_KM * c


def runway_margin_m(required_runway_m: int, available_runway_m: int) -> int:
    """
    Positive margin = runway is longer than required (good).
    Negative margin = runway is too short (infeasible for filter.py later).
    """
    return int(available_runway_m) - int(required_runway_m)


@dataclass(frozen=True)
class EngineFeatures:
    distance_km: float
    runway_margin_m: int
    has_medical: bool
    has_rescue: bool


def compute_features(scenario: Scenario, airport: Airport) -> EngineFeatures:
    """
    Computes the core feature set for Increment 1.
    """
    dist = haversine_km(
        scenario.aircraft_lat, scenario.aircraft_lon, airport.lat, airport.lon
    )
    margin = runway_margin_m(scenario.required_runway_m, airport.runway_length_m)

    return EngineFeatures(
        distance_km=dist,
        runway_margin_m=margin,
        has_medical=airport.has_medical,
        has_rescue=airport.has_rescue,
    )