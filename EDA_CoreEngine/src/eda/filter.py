"""
filter.py

Feasibility filtering (hard constraints) for the EDA Core Engine.

Purpose:
- Reject airports that are unsafe or infeasible BEFORE ranking.
- This acts as a safety gate that reduces hazard risk (e.g., selecting a runway
  that is too short or an airport that is unreachable).
- Integrate static airport restrictions and dynamic operational constraints.

Current hard constraints:
1) Reachability: airport distance must be within an assumed maximum range.
2) Runway feasibility: available runway must meet required runway.
3) Service constraint: if emergency is MEDICAL, airport must have medical capability.
4) Operational restrictions:
   - closed / temporary closed airports are rejected
   - unsafe airports are rejected
   - restricted / military airports are rejected for non-critical emergencies
   - dynamic scenario overrides are applied through OperationalConstraints
"""

from __future__ import annotations

from dataclasses import dataclass

from eda.airport_db import Airport
from eda.features import EngineFeatures
from eda.operational_constraints import (
    OperationalConstraints,
    check_operational_constraints,
)
from eda.scenario import Scenario, EmergencyType


@dataclass(frozen=True)
class FeasibilityResult:
    feasible: bool
    reason: str  # short explanation for reject/accept


def is_feasible(
    scenario: Scenario,
    airport: Airport,
    features: EngineFeatures,
    *,
    constraints: OperationalConstraints | None = None,
    max_range_km: float = 800.0,
) -> FeasibilityResult:
    """
    Determines whether an airport is feasible for the given scenario based on
    hard safety and operational constraints.

    Args:
        scenario: Current emergency scenario input.
        airport: Airport candidate being evaluated.
        features: Precomputed features for the airport.
        constraints: Optional scenario-specific operational overrides.
        max_range_km: Simplified reachability assumption.

    Returns:
        FeasibilityResult: feasible flag + reason.
    """

    if constraints is None:
        constraints = OperationalConstraints()

    # 1) Reachability constraint
    if features.distance_km > max_range_km:
        return FeasibilityResult(False, f"Rejected: unreachable (> {max_range_km} km)")

    # 2) Runway constraint
    if features.runway_margin_m < 0:
        return FeasibilityResult(False, "Rejected: runway too short")

    # 3) Medical service constraint
    if scenario.emergency_type == EmergencyType.MEDICAL and not features.has_medical:
        return FeasibilityResult(False, "Rejected: no medical capability")

    # 4) Operational restrictions constraint
    op_result = check_operational_constraints(airport, scenario, constraints)
    if not op_result.allowed:
        return FeasibilityResult(False, op_result.reason)

    return FeasibilityResult(True, "Accepted")