"""
operational_constraints.py

Defines optional operational constraints for the EDA Core Engine.

Purpose:
- Reserve a clean extension point for external operational restrictions
  such as airport closures, restricted countries, and unsafe zones.
- Keep these constraints separate from the aircraft emergency Scenario.

Note:
- In the current baseline, constraints are optional and not yet enforced.
- This module is added now to avoid future architectural refactoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class OperationalConstraints:
    """
    External operational restrictions affecting diversion feasibility.

    Current placeholder fields:
    - closed_airports: airports unavailable for diversion
    - restricted_countries: countries/regions to exclude
    """
    closed_airports: List[str] = field(default_factory=list)
    restricted_countries: List[str] = field(default_factory=list)