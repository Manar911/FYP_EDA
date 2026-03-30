"""
scenario.py

Defines the Scenario input model for the EDA Core Engine.

A Scenario represents the current emergency diversion situation that the
deterministic pipeline must process.

This version extends the original Increment 1 scenario model so the system can
carry the redesigned distance logic end-to-end for Increment 2 dataset
generation and ML preparation.

Key design principle:
- max_range_km is NOT the final reachability truth.
- It is the emergency-envelope input.
- Final usable reachability is determined by the interaction between:
    1) emergency envelope (max_range_km)
    2) aircraft capability adjusted by fuel state
- The resulting usable/extended ranges are computed upstream and stored here
  so downstream stages remain traceable and dataset-friendly.
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


class FuelState(str, Enum):
    CRITICAL = "critical"
    LOW = "low"
    NORMAL = "normal"


class BindingSide(str, Enum):
    AIRCRAFT_FUEL = "aircraft_fuel"
    EMERGENCY = "emergency"


@dataclass(frozen=True)
class Scenario:
    # Core Increment 1 inputs
    aircraft_lat: float
    aircraft_lon: float
    required_runway_m: int
    emergency_type: EmergencyType

    # Increment 2 aircraft / range context
    aircraft_type: str
    fuel_state: FuelState
    fuel_multiplier: float

    # Emergency-envelope input
    max_range_km: float

    # Derived range-model fields
    aircraft_adjusted_range_km: float
    usable_range_km: float
    extended_range_km: float
    binding_side: BindingSide

    def __post_init__(self) -> None:
        # ---------------------------------------------------------
        # Core validation
        # ---------------------------------------------------------
        if not (-90 <= self.aircraft_lat <= 90):
            raise ValueError("Invalid latitude (must be between -90 and 90).")

        if not (-180 <= self.aircraft_lon <= 180):
            raise ValueError("Invalid longitude (must be between -180 and 180).")

        if self.required_runway_m <= 0:
            raise ValueError("Runway requirement must be a positive integer.")

        if not self.aircraft_type.strip():
            raise ValueError("aircraft_type must not be empty.")

        # ---------------------------------------------------------
        # Fuel validation
        # ---------------------------------------------------------
        if self.fuel_multiplier <= 0:
            raise ValueError("fuel_multiplier must be > 0.")

        # Conservative guardrails based on the approved design
        allowed_multipliers = {
            FuelState.CRITICAL: 0.75,
            FuelState.LOW: 0.90,
            FuelState.NORMAL: 1.00,
        }

        expected_multiplier = allowed_multipliers[self.fuel_state]
        if abs(self.fuel_multiplier - expected_multiplier) > 1e-9:
            raise ValueError(
                f"fuel_multiplier {self.fuel_multiplier} does not match "
                f"fuel_state '{self.fuel_state.value}' "
                f"(expected {expected_multiplier})."
            )

        # ---------------------------------------------------------
        # Range validation
        # ---------------------------------------------------------
        if self.max_range_km <= 0:
            raise ValueError("max_range_km must be > 0.")

        if self.aircraft_adjusted_range_km <= 0:
            raise ValueError("aircraft_adjusted_range_km must be > 0.")

        if self.usable_range_km <= 0:
            raise ValueError("usable_range_km must be > 0.")

        if self.extended_range_km <= 0:
            raise ValueError("extended_range_km must be > 0.")

        # usable_range_km must be the minimum of:
        # - emergency envelope
        # - aircraft/fuel adjusted capability
        expected_usable = min(self.max_range_km, self.aircraft_adjusted_range_km)
        if abs(self.usable_range_km - expected_usable) > 1e-6:
            raise ValueError(
                f"usable_range_km must equal min(max_range_km, aircraft_adjusted_range_km). "
                f"Expected {expected_usable}, got {self.usable_range_km}."
            )

        if self.extended_range_km < self.usable_range_km:
            raise ValueError("extended_range_km must be >= usable_range_km.")

        # ---------------------------------------------------------
        # Binding-side consistency
        # ---------------------------------------------------------
        expected_binding = (
            BindingSide.AIRCRAFT_FUEL
            if self.aircraft_adjusted_range_km <= self.max_range_km
            else BindingSide.EMERGENCY
        )

        if self.binding_side != expected_binding:
            raise ValueError(
                f"binding_side '{self.binding_side.value}' is inconsistent with "
                f"max_range_km={self.max_range_km} and "
                f"aircraft_adjusted_range_km={self.aircraft_adjusted_range_km}. "
                f"Expected '{expected_binding.value}'."
            )