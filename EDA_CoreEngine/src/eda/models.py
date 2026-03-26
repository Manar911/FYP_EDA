"""
models.py

Shared data models for the EDA Core Engine (Increment 1).

These models represent the outputs of pipeline stages in a structured way.
They are designed to support:
- traceability
- unit testing
- future logging (Increment 1)
- future ML dataset generation (Increment 2)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from eda.airport_db import Airport
from eda.features import EngineFeatures
from eda.filter import FeasibilityResult
from eda.scenario import Scenario
from eda.ranking import RankedOption


@dataclass(frozen=True)
class EvaluatedAirport:
    """
    Holds a single airport evaluated against a scenario:
    - features computed for it
    - feasibility decision + reason
    """
    airport: Airport
    features: EngineFeatures
    feasibility: FeasibilityResult


@dataclass(frozen=True)
class DecisionReport:
    """
    Full deterministic decision report for one scenario run.
    """
    scenario: Scenario
    total_airports: int
    evaluated: List[EvaluatedAirport]
    feasible: List[Tuple[Airport, EngineFeatures]]
    ranked_top: List[RankedOption]