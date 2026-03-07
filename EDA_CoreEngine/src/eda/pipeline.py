"""
pipeline.py

End-to-end deterministic pipeline for the EDA Core Engine (Increment 1).

This module connects:
Scenario → Validation → Airport DB → Feature Engineering → Feasibility Filter → Ranking

It returns a DecisionReport, which is traceable and suitable for logging.
"""

from __future__ import annotations

from typing import List, Tuple

from eda.airport_db import load_airports, Airport
from eda.features import compute_features, EngineFeatures
from eda.filter import is_feasible
from eda.models import DecisionReport, EvaluatedAirport
from eda.ranking import rank_options
from eda.scenario import Scenario
from eda.validation import validate_scenario

from eda.config import DEFAULT_TOP_K, DEFAULT_MAX_RANGE_KM

def run_pipeline(
    scenario: Scenario,
    *,
    top_k: int = DEFAULT_TOP_K,
    max_range_km: float = DEFAULT_MAX_RANGE_KM,
) -> DecisionReport:
    """
    Runs the Increment 1 deterministic decision pipeline.

    Args:
        scenario: Scenario input.
        top_k: number of top airports to return.
        max_range_km: simplified reachability assumption (Increment 1).

    Returns:
        DecisionReport containing evaluated airports, feasible set, and top ranked outputs.
    """
    # 1) System-level validation
    validate_scenario(scenario)

    # 2) Load airport candidates
    airports = load_airports()

    evaluated: List[EvaluatedAirport] = []
    feasible_pairs: List[Tuple[Airport, EngineFeatures]] = []

    # 3) Evaluate each airport
    for a in airports:
        feats = compute_features(scenario, a)
        feas = is_feasible(scenario, feats, max_range_km=max_range_km)

        evaluated.append(EvaluatedAirport(airport=a, features=feats, feasibility=feas))

        if feas.feasible:
            feasible_pairs.append((a, feats))

    # 4) Rank feasible airports (top_k)
    ranked_top = rank_options(scenario.emergency_type, feasible_pairs, top_k=top_k)

    return DecisionReport(
        scenario=scenario,
        total_airports=len(airports),
        evaluated=evaluated,
        feasible=feasible_pairs,
        ranked_top=ranked_top,
    )