"""
ranking.py

Baseline deterministic ranking for the EDA Core Engine (Increment 1).

Purpose:
- Score feasible airports using simple, explainable rules (no AI).
- Return results as (Airport, EngineFeatures, score) to support:
  * explanations
  * logging
  * debugging
  * future ML labeling (Increment 2)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

from eda.airport_db import Airport
from eda.features import EngineFeatures
from eda.scenario import EmergencyType


@dataclass(frozen=True)
class RankedOption:
    airport: Airport
    features: EngineFeatures
    score: float


def _score_distance(distance_km: float) -> float:
    """
    Higher is better. Closer airports should score higher.
    Simple inverse relationship (bounded).
    """
    return 1.0 / (1.0 + float(distance_km))


def _score_runway_margin(margin_m: int) -> float:
    """
    Higher is better. More spare runway is safer/more comfortable.
    We only rank feasible airports, so margin is >= 0.
    """
    return float(margin_m) / 1000.0  # scale meters into km-like units


def score_airport(
    emergency_type: EmergencyType,
    features: EngineFeatures,
    *,
    w_distance: float = 0.6,
    w_runway: float = 0.4,
    bonus_medical: float = 0.5,
    bonus_rescue: float = 0.2,
) -> float:
    """
    Compute a deterministic baseline score from features.
    Weights are simple and tunable for Increment 1.
    """

    score = 0.0
    score += w_distance * _score_distance(features.distance_km)
    score += w_runway * _score_runway_margin(features.runway_margin_m)

    # Bonuses (small, explainable)
    if emergency_type == EmergencyType.MEDICAL and features.has_medical:
        score += bonus_medical

    # Rescue capability is generally useful for many emergencies
    if features.has_rescue:
        score += bonus_rescue

    return float(score)


def rank_options(
    emergency_type: EmergencyType,
    options: Iterable[Tuple[Airport, EngineFeatures]],
    *,
    top_k: int = 3,
) -> List[RankedOption]:
    """
    Rank feasible options and return top_k highest-scoring.
    """
    ranked: List[RankedOption] = []

    for airport, feats in options:
        s = score_airport(emergency_type, feats)
        ranked.append(RankedOption(airport=airport, features=feats, score=s))

    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked[:top_k]