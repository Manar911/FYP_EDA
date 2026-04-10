"""
validation.py

System-level validation for the EDA Core Engine (Increment 1).

This module provides an explicit validation layer separate from the Scenario model.

- Keeps the Scenario dataclass lightweight and focused on basic invariants.
- Allows additional "system policy" checks to evolve without changing the model.
- Improves testability and traceability (validation rules can be linked to test cases).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from scenario import Scenario


@dataclass(frozen=True)
class ValidationIssue:
    field: str
    message: str


class ScenarioValidationError(ValueError):
    """Raised when a scenario fails system-level validation."""

    def __init__(self, issues: list[ValidationIssue]):
        super().__init__("Scenario failed validation.")
        self.issues = issues


def validate_scenario(s: Scenario) -> None:
    """
    Validates a Scenario against system-level rules.

    Raises:
        ScenarioValidationError: if one or more rules are violated.

    Note:
        This function intentionally raises with a list of issues rather than failing fast.
        This is helpful for debugging and (later) UI feedback.
    """
    issues: list[ValidationIssue] = []

    # Example system-level bounds (adjust later as spec evolves)
    # These are "sanity limits" for a prototype, not aviation-certified limits.
    if s.required_runway_m > 6000:
        issues.append(
            ValidationIssue(
                field="required_runway_m",
                message="Runway requirement is unusually high (> 6000m).",
            )
        )

    # Optional: you can add more system policies here later, e.g.:
    # - emergency_type-specific constraints
    # - minimum runway for certain emergency types
    # - etc.

    if issues:
        raise ScenarioValidationError(issues)