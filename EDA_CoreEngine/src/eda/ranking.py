"""
ranking.py

Deterministic ranking for the EDA Core Engine.

Purpose:
- Score feasible airports using explainable rules
- Include soft operational penalties
- Include a last-resort penalty for extended-range candidates
- Prepare realistic labels for ML (Increment 2)

Important:
- Preferred-range airports should generally outrank otherwise similar
  extended-range airports.
- Extended-range airports remain feasible, but are treated as fallback options.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

from eda.airport_db import Airport
from eda.features import EngineFeatures
from eda.scenario import EmergencyType

from eda.config import (
    DISTANCE_WEIGHT,
    RUNWAY_WEIGHT,
    MEDICAL_BONUS,
    RESCUE_BONUS,
)


# ---------------------------------------------------------
# Distance-zone penalty calibration
# ---------------------------------------------------------

EXTENDED_RANGE_PENALTY = 0.15


@dataclass(frozen=True)
class RankedOption:
    airport: Airport
    features: EngineFeatures
    score: float
    distance_zone: str  # "preferred" or "extended"


# ---------------------------
# Base scoring
# ---------------------------

def _score_distance(distance_km: float) -> float:
    return 1.0 / (1.0 + float(distance_km))


def _score_runway_margin(margin_m: int) -> float:
    return float(margin_m) / 1000.0


# ---------------------------
# Distance-zone penalties
# ---------------------------

def _distance_zone_penalty(distance_zone: str) -> float:
    """
    Returns a penalty value (to subtract from score) based on the approved
    distance-zone design.

    preferred -> no penalty
    extended  -> last-resort penalty
    """
    zone = distance_zone.strip().lower()

    if zone == "preferred":
        return 0.0

    if zone == "extended":
        return EXTENDED_RANGE_PENALTY

    raise ValueError(
        f"Unsupported distance_zone '{distance_zone}'. "
        "Expected 'preferred' or 'extended'."
    )


# ---------------------------
# Operational penalties
# ---------------------------

def _operational_penalty(
    airport: Airport,
    emergency_type: EmergencyType,
) -> float:
    """
    Returns a penalty value (to subtract from score).
    """

    penalty = 0.0
    emergency = emergency_type.value.lower()

    # Identify critical emergencies
    critical = any(k in emergency for k in [
        "medical", "fuel", "technical", "engine", "fire", "smoke", "security"
    ])

    # Normalize values
    unsafe_status = airport.unsafe_status.lower()
    civil_military = airport.civil_military.lower()
    restricted_status = airport.restricted_status.lower()

    # ---------------------------
    # Caution airports
    # ---------------------------
    if unsafe_status == "caution":
        penalty += 0.15

    # ---------------------------
    # Joint-use airports
    # ---------------------------
    if civil_military == "joint":
        penalty += 0.10

    # ---------------------------
    # Military airports
    # ---------------------------
    if civil_military == "military":
        if critical:
            penalty += 0.40   # allowed but strong penalty
        else:
            penalty += 1.0    # should already be filtered, but safe fallback

    # ---------------------------
    # Restricted airports
    # ---------------------------
    if restricted_status in {"restricted", "military_restricted"}:
        if critical:
            penalty += 0.30   # last resort
        else:
            penalty += 1.0    # should already be filtered

    return penalty


# ---------------------------
# Final scoring
# ---------------------------

def score_airport(
    emergency_type: EmergencyType,
    airport: Airport,
    features: EngineFeatures,
    *,
    distance_zone: str,
    w_distance: float = DISTANCE_WEIGHT,
    w_runway: float = RUNWAY_WEIGHT,
    bonus_medical: float = MEDICAL_BONUS,
    bonus_rescue: float = RESCUE_BONUS,
) -> float:

    score = 0.0

    # Base score
    score += w_distance * _score_distance(features.distance_km)
    score += w_runway * _score_runway_margin(features.runway_margin_m)

    # Bonuses
    if emergency_type == EmergencyType.MEDICAL and features.has_medical:
        score += bonus_medical

    if features.has_rescue:
        score += bonus_rescue

    # Soft last-resort penalty for extended-range candidates
    score -= _distance_zone_penalty(distance_zone)

    # Operational penalties
    score -= _operational_penalty(airport, emergency_type)

    return float(score)


# ---------------------------
# Ranking
# ---------------------------

def rank_options(
    emergency_type: EmergencyType,
    options: Iterable[Tuple[Airport, EngineFeatures, str]],
    *,
    top_k: int = 3,
) -> List[RankedOption]:
    """
    Ranks feasible airports.

    Expected option tuple:
        (airport, features, distance_zone)

    distance_zone must be:
        - "preferred"
        - "extended"
    """

    ranked: List[RankedOption] = []

    for airport, feats, distance_zone in options:
        s = score_airport(
            emergency_type,
            airport,
            feats,
            distance_zone=distance_zone,
        )
        ranked.append(
            RankedOption(
                airport=airport,
                features=feats,
                score=s,
                distance_zone=distance_zone,
            )
        )

    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked[:top_k]