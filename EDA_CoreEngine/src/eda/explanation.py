"""
explanation.py

Explanation generation for the EDA Core Engine (Increment 1).

Purpose:
- Provide transparent, human-readable reasons for why an airport was recommended.
- Support trust, traceability, and interpretability in a safety-critical
  decision-support context.

This baseline version uses deterministic rule-based explanations derived from:
- distance
- runway margin
- medical capability
- rescue capability
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from eda.ranking import RankedOption
from eda.scenario import EmergencyType


@dataclass(frozen=True)
class Explanation:
    airport_icao: str
    reasons: List[str]


def generate_explanation(
    ranked_option: RankedOption,
    emergency_type: EmergencyType,
) -> Explanation:
    """
    Generate a short explanation for a ranked airport option.
    """
    airport = ranked_option.airport
    features = ranked_option.features

    reasons: List[str] = []

    # Distance-based explanation
    if features.distance_km < 50:
        reasons.append("Very close to the aircraft position")
    elif features.distance_km < 200:
        reasons.append("Relatively close diversion option")
    else:
        reasons.append("Feasible diversion option within range")

    # Runway explanation
    if features.runway_margin_m > 1500:
        reasons.append(f"Runway exceeds requirement by {features.runway_margin_m} m")
    elif features.runway_margin_m > 0:
        reasons.append(f"Runway margin available: +{features.runway_margin_m} m")

    # Capability explanations
    if emergency_type == EmergencyType.MEDICAL and features.has_medical:
        reasons.append("Medical capability available")

    if features.has_rescue:
        reasons.append("Rescue capability available")

    return Explanation(
        airport_icao=airport.icao,
        reasons=reasons,
    )