"""
filter.py

Feasibility filtering (hard constraints) for the EDA Core Engine.

Purpose:
- Reject airports that are unsafe or infeasible BEFORE ranking.
- Act as a safety gate that removes clearly invalid candidates.
- Apply the redesigned distance model using:
    1) usable_range_km as the preferred boundary
    2) extended_range_km as the last-resort boundary
- Integrate static airport restrictions and dynamic operational constraints.

Distance redesign:
- Preferred: distance <= usable_range_km
- Extended: usable_range_km < distance <= extended_range_km
- Reject: distance > extended_range_km

Important:
- max_range_km is no longer the final reachability truth.
- It is one input into the range model already carried inside Scenario.
"""

from __future__ import annotations

from dataclasses import dataclass

from airport_db import Airport
from features import EngineFeatures
from operational_constraints import (
    OperationalConstraints,
    check_operational_constraints,
)
from scenario import Scenario, EmergencyType


@dataclass(frozen=True)
class FeasibilityResult:
    feasible: bool
    reason: str
    zone: str  # "preferred", "extended", or "reject"


def _classify_distance_zone(
    distance_km: float,
    usable_range_km: float,
    extended_range_km: float,
) -> str:
    """
    Classifies the airport into one of the three approved distance zones.
    """
    if distance_km <= usable_range_km:
        return "preferred"

    if distance_km <= extended_range_km:
        return "extended"

    return "reject"


def is_feasible(
    scenario: Scenario,
    airport: Airport,
    features: EngineFeatures,
    *,
    constraints: OperationalConstraints | None = None,
    max_range_km: float = 800.0,  # kept temporarily for compatibility
) -> FeasibilityResult:
    """
    Determines whether an airport is feasible for the given scenario based on
    hard safety and operational constraints.

    Args:
        scenario: Current emergency scenario input.
        airport: Airport candidate being evaluated.
        features: Precomputed features for the airport.
        constraints: Optional scenario-specific operational overrides.
        max_range_km: Legacy compatibility parameter. No longer used as the
            final reachability authority once Scenario carries the redesigned
            range model.

    Returns:
        FeasibilityResult: feasible flag + explanation + distance zone.
    """

    # Explicitly acknowledge the legacy parameter but do not use it as the
    # real source of truth anymore. The redesigned range model lives in Scenario.
    _ = max_range_km

    if constraints is None:
        constraints = OperationalConstraints()

    # 1) Distance-zone classification using redesigned range model
    distance_zone = _classify_distance_zone(
        distance_km=float(features.distance_km),
        usable_range_km=float(scenario.usable_range_km),
        extended_range_km=float(scenario.extended_range_km),
    )

    if distance_zone == "reject":
        return FeasibilityResult(
            False,
            (
                "Rejected: unreachable beyond extended range "
                f"(distance={features.distance_km:.2f} km, "
                f"usable={scenario.usable_range_km:.2f} km, "
                f"extended={scenario.extended_range_km:.2f} km)"
            ),
            "reject",
        )

    # 2) Runway constraint
    if features.runway_margin_m < 0:
        return FeasibilityResult(False, "Rejected: runway too short", "reject")

    # 3) Medical service constraint
    if scenario.emergency_type == EmergencyType.MEDICAL and not features.has_medical:
        return FeasibilityResult(False, "Rejected: no medical capability", "reject")

    # 4) Operational restrictions constraint
    op_result = check_operational_constraints(airport, scenario, constraints)
    if not op_result.allowed:
        return FeasibilityResult(False, op_result.reason, "reject")

    # 5) Accepted candidate
    if distance_zone == "preferred":
        return FeasibilityResult(True, "Accepted: preferred range", "preferred")

    return FeasibilityResult(True, "Accepted: extended range", "extended")