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
    MAINTENANCE_BONUS,
    ILS_BONUS,
)



# Distance-zone penalty calibration


EXTENDED_RANGE_PENALTY = 0.15


@dataclass(frozen=True)
class RankedOption:
    airport: Airport
    features: EngineFeatures
    score: float
    distance_zone: str  # "preferred" or "extended"



# Base scoring


def _score_distance(distance_km: float) -> float:
    """
    Normalised distance score.
    0 km  → 1.0 (best possible)
    500 km → ~0.5 (moderate)
    1500 km → ~0.25 (far but reachable)
    
    Uses a reference scale of 500km as the 
    mid-point — typical preferred diversion range.
    """
    return 500.0 / (500.0 + float(distance_km))


def _score_runway_margin(margin_m: int) -> float:
    """
    Normalised runway margin score.
    A margin of 500m is considered adequate.
    A margin of 1500m is considered excellent.
    Beyond 1500m adds no additional value.
    
    Capped so oversized runways don't dominate.
    """
    capped = min(float(margin_m), 1500.0)
    return capped / 1500.0



# Distance-zone penalties


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



# Operational penalties


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

    
    # Caution airports
    
    if unsafe_status == "caution":
        penalty += 0.15

    
    # Joint-use airports
    
    if civil_military == "joint":
        penalty += 0.10

    
    # Military airports
    
    if civil_military == "military":
        if critical:
            penalty += 0.40   # allowed but strong penalty
        else:
            penalty += 1.0    # should already be filtered, but safe fallback

    
    # Restricted airports
    
    if restricted_status in {"restricted", "military_restricted"}:
        if critical:
            penalty += 0.30   # last resort
        else:
            penalty += 1.0    # should already be filtered

    return penalty



# Final scoring


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
    """
    Emergency-type aware weight adjustment.

    Different emergencies have different operational
    priorities. The default weights from config.py are
    overridden here based on what matters most for each
    emergency category.

    FUEL: Every extra kilometre burns critically low fuel.
          Distance is the dominant priority. Infrastructure
          advantages of farther airports do not justify the
          additional fuel burn.

    MEDICAL: Medical facility availability is the dominant
             priority. A nearby airport with no medical
             capability is less useful than a slightly
             farther airport with full medical support.
             The MEDICAL_BONUS handles the facility signal.

    MECHANICAL / TECHNICAL: Maintenance capability matters
             most. Distance and runway are secondary to
             whether the airport can support the aircraft.

    WEATHER: Instrument approach quality (ILS) and runway
             length matter more in degraded visibility.
             Weather reporting capability is also rewarded
             as it gives the crew accurate approach data.

    SECURITY: Default weights apply. Security emergencies
              require rapid diversion but do not have a
              single dominant infrastructure requirement
              beyond the standard filter constraints.

    OPERATIONAL_CONSTRAINTS: Default weights apply.
              These are generally lower-urgency diversions
              where the standard balance is appropriate.
    """

    score = 0.0

    if emergency_type == EmergencyType.FUEL:
        # Distance is critical — land as soon as safely possible
        w_distance = 0.80
        w_runway = 0.20

    elif emergency_type == EmergencyType.MEDICAL:
        # Medical facility bonus dominates — reduce base weights
        # to give the bonus more relative influence
        w_distance = 0.50
        w_runway = 0.20

    elif emergency_type in (EmergencyType.MECHANICAL, EmergencyType.TECHNICAL):
        # Maintenance capability matters — moderate distance,
        # slightly increased runway weight for safety margin
        w_distance = 0.60
        w_runway = 0.25

    elif emergency_type == EmergencyType.WEATHER:
        # Instrument approach and runway length matter more
        # in degraded visibility conditions
        w_distance = 0.55
        w_runway = 0.30

    # Base score
    score += w_distance * _score_distance(features.distance_km)
    score += w_runway * _score_runway_margin(features.runway_margin_m)

    
    # Capability bonuses
    

    # Medical bonus — critical for medical emergencies
    if emergency_type == EmergencyType.MEDICAL and features.has_medical:
        score += bonus_medical

    # Rescue capability — always beneficial
    if features.has_rescue:
        score += bonus_rescue

    # Maintenance bonus — critical for mechanical/technical
    if emergency_type in (EmergencyType.MECHANICAL, EmergencyType.TECHNICAL):
        if airport.has_maintenance:
            score += MAINTENANCE_BONUS

    # ILS bonus — critical for weather emergencies
    if emergency_type == EmergencyType.WEATHER:
        if airport.has_ils:
            score += ILS_BONUS
        # Weather reporting gives crew accurate approach data
        if airport.weather_reporting:
            score += 0.10

    
    # Distance zone and operational penalties
    

    # Soft last-resort penalty for extended-range candidates
    score -= _distance_zone_penalty(distance_zone)

    # Operational penalties
    score -= _operational_penalty(airport, emergency_type)

    return float(score)



# Ranking


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