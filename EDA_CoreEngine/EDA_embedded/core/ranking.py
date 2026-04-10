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

from airport_db import Airport
from features import EngineFeatures
from scenario import EmergencyType

from config import (
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
    Emergency-type aware scoring.

    Each emergency type has a distinct priority profile
    controlling both base weights and which capability
    bonuses are active.

    FUEL: w_distance=0.80, w_runway=0.20.
          Distance is dominant — every extra km burns
          critically low fuel. Firefighting and fuel
          availability bonuses apply. Rescue and 24h
          bonuses also apply.

    MEDICAL: w_distance=0.50, w_runway=0.20.
          Medical facility availability is dominant.
          Full medical and rescue bonuses apply.
          A nearby airport with no medical capability
          is less useful than a slightly farther one
          with full medical support.

    MECHANICAL / TECHNICAL: w_distance=0.60, w_runway=0.25.
          Maintenance capability is dominant. Full
          firefighting, rescue, and maintenance bonuses
          apply. Distance and runway are secondary to
          whether the airport can support the aircraft.

    WEATHER: w_distance=0.65, w_runway=0.30.
          ILS and runway length dominate in degraded
          visibility. Distance weight is elevated so
          proximity is not overridden by capability.
          Rescue, firefighting, and 24h bonuses are
          suppressed — only ILS (+0.25) and weather
          reporting (+0.10) bonuses apply.

    SECURITY: w_distance=0.70, w_runway=0.25.
          Rapid diversion is the priority — distance
          is strongly dominant. Standard rescue and
          24h bonuses still apply since ground response
          teams are needed for security incidents.

    OPERATIONAL_CONSTRAINTS: default weights (0.60/0.40).
          Lower-urgency diversion. Balanced distance
          and runway scoring. Standard bonuses apply
          except rescue and firefighting which are
          suppressed as infrastructure is not critical.
    """

    score = 0.0

    if emergency_type == EmergencyType.FUEL:
        w_distance = 0.80
        w_runway = 0.20

    elif emergency_type == EmergencyType.MEDICAL:
        w_distance = 0.50
        w_runway = 0.20

    elif emergency_type in (EmergencyType.MECHANICAL, EmergencyType.TECHNICAL):
        w_distance = 0.60
        w_runway = 0.25

    elif emergency_type == EmergencyType.WEATHER:
        w_distance = 0.65
        w_runway = 0.30

    elif emergency_type == EmergencyType.SECURITY:
        w_distance = 0.70
        w_runway = 0.25

    # Base score
    score += w_distance * _score_distance(features.distance_km)
    score += w_runway * _score_runway_margin(features.runway_margin_m)

    # --------------------------------------------------
    # Graduated medical capability bonus.
    # Advanced medical facilities are significantly more
    # valuable than basic ones for medical emergencies.
    # Basic is still better than none — which is already
    # enforced by the hard filter in filter.py.
    # --------------------------------------------------
    if emergency_type == EmergencyType.MEDICAL and features.has_medical:
        if airport.medical_level == "advanced":
            score += bonus_medical            # 0.50 — full bonus
        else:
            score += bonus_medical * 0.60    # 0.30 — basic level

    # --------------------------------------------------
    # Graduated rescue capability bonus.
    # Major rescue category provides significantly better
    # emergency response than standard category.
    # --------------------------------------------------
    if features.has_rescue:
        if emergency_type not in (
            EmergencyType.WEATHER,
            EmergencyType.OPERATIONAL_CONSTRAINTS,
        ):
            if airport.rescue_category == "major":
                score += bonus_rescue
            else:
                score += bonus_rescue * 0.50
        else:
            score += 0.03

    # --------------------------------------------------
    # Firefighting capability bonus.
    # Critical for fuel and mechanical emergencies where
    # fire risk is elevated. Always beneficial otherwise.
    # --------------------------------------------------
    if airport.has_firefighting:
        if emergency_type in (
            EmergencyType.FUEL,
            EmergencyType.MECHANICAL,
            EmergencyType.TECHNICAL,
        ):
            score += 0.15
        elif emergency_type not in (
            EmergencyType.WEATHER,
            EmergencyType.OPERATIONAL_CONSTRAINTS,
        ):
            score += 0.05

    # --------------------------------------------------
    # Fuel availability bonus.
    # Important for fuel emergencies — confirms the
    # aircraft can be refuelled after landing.
    # --------------------------------------------------
    if emergency_type == EmergencyType.FUEL:
        if airport.fuel_available:
            score += 0.10

    # --------------------------------------------------
    # 24h operations bonus.
    # An airport open at all hours is more reliable for
    # emergency diversions regardless of time of day.
    # --------------------------------------------------
    if airport.open_24h:
        if emergency_type != EmergencyType.WEATHER:
            score += 0.05
        else:
            score += 0.01

    # --------------------------------------------------
    # Maintenance bonus — critical for mechanical/technical
    # --------------------------------------------------
    if emergency_type in (EmergencyType.MECHANICAL, EmergencyType.TECHNICAL):
        if airport.has_maintenance:
            score += MAINTENANCE_BONUS

    # --------------------------------------------------
    # ILS and weather reporting bonuses for weather emergencies
    # --------------------------------------------------
    if emergency_type == EmergencyType.WEATHER:
        if airport.has_ils:
            score += ILS_BONUS
        if airport.weather_reporting:
            score += 0.10

    # --------------------------------------------------
    # Slot restriction soft penalty.
    # Slot-restricted airports add coordination complexity
    # during emergencies — a soft discouragement only,
    # not a hard rejection.
    # --------------------------------------------------
    if airport.slot_restricted:
        score -= 0.05

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