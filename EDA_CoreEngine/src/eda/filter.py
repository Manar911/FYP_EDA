"""
filter.py

Feasibility filtering (hard constraints) for the EDA Core Engine (Increment 1).

Purpose:
- Reject airports that are unsafe or infeasible BEFORE ranking.
- This acts as a safety gate that reduces hazard risk (e.g., selecting a runway
  that is too short or an airport that is unreachable).

Increment 1 baseline constraints:
1) Runway feasibility: available runway must meet required runway.
2) Reachability: airport distance must be within an assumed maximum range.
3) Service constraint (baseline): if emergency is MEDICAL, airport must have medical capability.
"""

from __future__ import annotations

from dataclasses import dataclass

from eda.features import EngineFeatures
from eda.scenario import Scenario, EmergencyType


@dataclass(frozen=True)
class FeasibilityResult:
    feasible: bool
    reason: str  # short explanation for reject/accept


def is_feasible(
    scenario: Scenario,
    features: EngineFeatures,
    *,
    max_range_km: float = 800.0,
) -> FeasibilityResult:
    """
    Determines whether an airport is feasible for the given scenario based on
    hard safety constraints.

    Args:
        scenario: Current emergency scenario input.
        features: Precomputed features for a specific airport.
        max_range_km: Simplified reachability assumption for Increment 1.

    Returns:
        FeasibilityResult: feasible flag + reason (useful for debugging/explanations/logging).
    """

    # 1) Reachability constraint
    if features.distance_km > max_range_km:
        return FeasibilityResult(False, f"Rejected: unreachable (> {max_range_km} km)")

    # 2) Runway constraint
    if features.runway_margin_m < 0:
        return FeasibilityResult(False, "Rejected: runway too short")

    # 3) Baseline service constraint (for medical emergencies)
    if scenario.emergency_type == EmergencyType.MEDICAL and not features.has_medical:
        return FeasibilityResult(False, "Rejected: no medical capability")

    return FeasibilityResult(True, "Accepted")