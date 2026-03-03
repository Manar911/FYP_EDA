"""
scenario.py

Defines the Scenario input model for the EDA Core Engine (Increment 1).

A Scenario represents the "current situation" that the deterministic pipeline must
process (e.g., aircraft position, runway requirement, and emergency type).

This module intentionally keeps inputs structured and validated, so later stages
(feature engineering, feasibility filtering, ranking, explanations, and logging)
can rely on correct, consistent data.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EmergencyType(str, Enum):
    MEDICAL = "medical"
    TECHNICAL = "technical"
    MECHANICAL = "mechanical"
    WEATHER = "weather"
    SECURITY = "security"
    FUEL = "fuel"
    OPERATIONAL_CONSTRAINTS = "operational_constraints"


@dataclass(frozen=True)
class Scenario:
    aircraft_lat: float
    aircraft_lon: float
    required_runway_m: int
    emergency_type: EmergencyType

    def __post_init__(self) -> None:
        # Basic validation to prevent invalid system states early
        if not (-90 <= self.aircraft_lat <= 90):
            raise ValueError("Invalid latitude (must be between -90 and 90).")

        if not (-180 <= self.aircraft_lon <= 180):
            raise ValueError("Invalid longitude (must be between -180 and 180).")

        if self.required_runway_m <= 0:
            raise ValueError("Runway requirement must be a positive integer.")