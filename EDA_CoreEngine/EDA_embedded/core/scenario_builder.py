"""
scenario_builder.py

Runtime scenario construction for the EDA Core Engine.

Builds a complete Scenario object from minimal user inputs by
automatically loading aircraft performance data from the aircraft
database. The user only needs to provide aircraft type, position,
fuel state, and emergency type. All derived values are computed
automatically and consistently with the training data pipeline.
"""

from __future__ import annotations

from aircraft_db import load_aircraft_profiles
from scenario import Scenario, EmergencyType, FuelState, BindingSide


FUEL_MULTIPLIERS = {
    FuelState.NORMAL:   1.00,
    FuelState.LOW:      0.90,
    FuelState.CRITICAL: 0.75,
}

EXTENDED_RANGE_FACTOR = 1.25


def build_scenario(
    *,
    aircraft_type: str,
    aircraft_lat: float,
    aircraft_lon: float,
    fuel_state: FuelState,
    emergency_type: EmergencyType,
) -> Scenario:
    """
    Builds a complete Scenario from minimal user inputs.

    Looks up the aircraft profile from the database and computes
    all derived range and runway values automatically. This ensures
    runtime scenarios are consistent with training data generation
    and eliminates manual hardcoding of aircraft parameters.

    Args:
        aircraft_type:  Aircraft type string e.g. 'A320', 'B737-800'
        aircraft_lat:   Current aircraft latitude
        aircraft_lon:   Current aircraft longitude
        fuel_state:     Current fuel state (NORMAL / LOW / CRITICAL)
        emergency_type: Type of emergency declared

    Returns:
        A fully constructed and validated Scenario object.

    Raises:
        ValueError: If aircraft_type is not found in the database.
    """

    # Load aircraft profile from database
    profiles = load_aircraft_profiles()
    profile_map = {p.aircraft_type: p for p in profiles}

    if aircraft_type not in profile_map:
        available = sorted(profile_map.keys())
        raise ValueError(
            f"Aircraft type '{aircraft_type}' not found in database. "
            f"Available: {available}"
        )

    profile = profile_map[aircraft_type]

    # Compute all derived values
    fuel_multiplier = FUEL_MULTIPLIERS[fuel_state]
    max_range_km = float(profile.nominal_diversion_capability_km)
    aircraft_adjusted_range_km = round(max_range_km * fuel_multiplier, 6)
    usable_range_km = min(max_range_km, aircraft_adjusted_range_km)
    extended_range_km = round(usable_range_km * EXTENDED_RANGE_FACTOR, 6)

    binding_side = (
        BindingSide.AIRCRAFT_FUEL
        if aircraft_adjusted_range_km <= max_range_km
        else BindingSide.EMERGENCY
    )

    return Scenario(
        aircraft_lat=aircraft_lat,
        aircraft_lon=aircraft_lon,
        required_runway_m=profile.runway_max_m,
        emergency_type=emergency_type,
        aircraft_type=aircraft_type,
        aircraft_category=profile.aircraft_category,
        fuel_state=fuel_state,
        fuel_multiplier=fuel_multiplier,
        max_range_km=max_range_km,
        aircraft_adjusted_range_km=aircraft_adjusted_range_km,
        usable_range_km=usable_range_km,
        extended_range_km=extended_range_km,
        binding_side=binding_side,
    )


def list_available_aircraft() -> list[str]:
    """Returns all aircraft types available in the database."""
    profiles = load_aircraft_profiles()
    return sorted(p.aircraft_type for p in profiles)