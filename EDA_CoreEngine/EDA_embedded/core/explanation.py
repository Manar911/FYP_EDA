"""
explanation.py

Explanation generation for the EDA Core Engine.

Purpose:
- Provide transparent, human-readable reasons for why an airport was recommended.
- Support trust, traceability, and interpretability in a safety-critical
  decision-support context.

This version provides a lightweight embedded XAI layer using:
- feasibility explanations from the safety filter
- ranking explanations from model input features and airport capabilities
- a short human-readable summary
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ranking import RankedOption
from scenario import EmergencyType


@dataclass(frozen=True)
class Explanation:
    airport_icao: str
    feasibility_reasons: List[str]
    ranking_reasons: List[str]
    summary: str
    caution: str


def generate_explanation(
    ranked_option: RankedOption,
    emergency_type: EmergencyType,
    feasibility_reason: str,
) -> Explanation:
    """
    Generate a structured explanation for a ranked airport option.
    """
    airport = ranked_option.airport
    features = ranked_option.features
    zone = ranked_option.distance_zone

    # -------------------------
    # Feasibility explanation
    # -------------------------
    feasibility_reasons: List[str] = [feasibility_reason]

    if zone == "preferred":
        feasibility_reasons.append("Within preferred diversion range")
    elif zone == "extended":
        feasibility_reasons.append("Within extended diversion range as a fallback option")

    # -------------------------
    # Ranking explanation
    # -------------------------
    ranking_reasons: List[str] = []

    # Distance
    if features.distance_km < 50:
        ranking_reasons.append("Very close to the aircraft position")
    elif features.distance_km < 200:
        ranking_reasons.append("Relatively close diversion option")
    else:
        ranking_reasons.append("Within reachable diversion range for the current scenario")

    # Runway
    if features.runway_margin_m > 1500:
        ranking_reasons.append("Strong runway margin for safe landing")
    elif features.runway_margin_m > 0:
        ranking_reasons.append(f"Runway margin available: +{features.runway_margin_m} m")

    # Emergency-specific explanations
    if emergency_type == EmergencyType.MEDICAL and features.has_medical:
        if airport.medical_level == "advanced":
            ranking_reasons.append("Advanced medical capability available")
        else:
            ranking_reasons.append("Medical capability available")

    if emergency_type in (EmergencyType.MECHANICAL, EmergencyType.TECHNICAL):
        if airport.has_maintenance:
            ranking_reasons.append("Maintenance capability available")

    if emergency_type == EmergencyType.WEATHER:
        if airport.has_ils:
            ranking_reasons.append("ILS available for low-visibility landing")

    if emergency_type == EmergencyType.FUEL:
        if airport.fuel_available:
            ranking_reasons.append("Fuel available after landing")

    # General support services
    if features.has_rescue:
        if airport.rescue_category == "major":
            ranking_reasons.append("Major rescue capability available")
        else:
            ranking_reasons.append("Rescue capability available")

    if airport.has_firefighting:
        ranking_reasons.append("Firefighting services available")

    if airport.open_24h:
        ranking_reasons.append("Airport operates 24 hours")

    # -------------------------
    # Summary sentence
    # -------------------------
    summary_parts: List[str] = []

    if features.runway_margin_m > 0:
        summary_parts.append("sufficient runway")

    if features.distance_km < 200:
        summary_parts.append("acceptable proximity")

    if features.has_rescue:
        summary_parts.append("emergency support")

    if emergency_type == EmergencyType.MEDICAL and features.has_medical:
        summary_parts.append("medical capability")

    if emergency_type in (EmergencyType.MECHANICAL, EmergencyType.TECHNICAL) and airport.has_maintenance:
        summary_parts.append("maintenance support")

    if emergency_type == EmergencyType.WEATHER and airport.has_ils:
        summary_parts.append("instrument landing support")

    if summary_parts:
        if len(summary_parts) == 1:
            summary = f"Recommended because it offers {summary_parts[0]}."
        else:
            summary = (
                "Recommended because it offers "
                + ", ".join(summary_parts[:-1])
                + f" and {summary_parts[-1]}."
            )
    else:
        summary = "Recommended as a feasible diversion option."

    return Explanation(
        airport_icao=airport.icao,
        feasibility_reasons=feasibility_reasons,
        ranking_reasons=ranking_reasons,
        summary=summary,
        caution="Final diversion decision remains with the pilot.",
    )