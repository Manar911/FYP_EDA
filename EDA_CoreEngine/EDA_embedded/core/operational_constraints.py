"""
operational_constraints.py

Operational restriction rules for the EDA Core Engine.

Purpose:
- Keep temporary scenario-specific restrictions separate from the airport database
- Centralise operational restriction logic in one place
- Support both static airport DB statuses and dynamic scenario overrides

Design:
- Static baseline restrictions live in the airport database
- Dynamic overrides live in OperationalConstraints
- Hard operational rejection is enforced before ranking
- Softer operational discouragement is handled later in ranking
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from airport_db import Airport
from scenario import Scenario


@dataclass(frozen=True)
class OperationalConstraints:
    """
    Scenario-specific temporary operational restrictions.

    These are dynamic overrides in addition to the static fields stored
    in the airport database.
    """
    closed_airports: List[str] = field(default_factory=list)
    restricted_countries: List[str] = field(default_factory=list)
    unsafe_airports: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class OperationalCheckResult:
    allowed: bool
    reason: str


def is_critical_emergency(scenario: Scenario) -> bool:
    """
    Returns True for emergency categories where restricted or military
    airports may be considered as a last resort.

    Critical examples:
    - medical
    - fuel
    - technical / mechanical
    - engine / fire / smoke
    - security / hijack
    """
    emergency_value = scenario.emergency_type.value.strip().lower()

    critical_keywords = (
        "medical",
        "fuel",
        "technical",
        "mechanical",
        "engine",
        "fire",
        "smoke",
        "security",
        "hijack",
    )

    return any(keyword in emergency_value for keyword in critical_keywords)


def check_operational_constraints(
    airport: Airport,
    scenario: Scenario,
    constraints: OperationalConstraints,
) -> OperationalCheckResult:
    """
    Applies hard operational constraint checks before ranking.

    Returns:
        OperationalCheckResult:
            allowed = False if the airport must be rejected before scoring
            allowed = True if the airport may continue to ranking
    """

    icao = airport.icao.strip().upper()
    country = airport.country.strip().lower()

    closure_status = airport.closure_status.strip().lower()
    restricted_status = airport.restricted_status.strip().lower()
    unsafe_status = airport.unsafe_status.strip().lower()
    civil_military = airport.civil_military.strip().lower()

    closed_airports = {x.strip().upper() for x in constraints.closed_airports}
    restricted_countries = {x.strip().lower() for x in constraints.restricted_countries}
    unsafe_airports = {x.strip().upper() for x in constraints.unsafe_airports}

    critical = is_critical_emergency(scenario)

    # -------------------------------------------------
    # 1) Dynamic scenario-specific overrides
    # -------------------------------------------------
    if icao in closed_airports:
        return OperationalCheckResult(
            False,
            "Rejected: airport temporarily closed by scenario constraints",
        )

    if icao in unsafe_airports:
        return OperationalCheckResult(
            False,
            "Rejected: airport marked unsafe by scenario constraints",
        )

    if country in restricted_countries:
        return OperationalCheckResult(
            False,
            "Rejected: country restricted by scenario constraints",
        )

    # -------------------------------------------------
    # 2) Static airport database restriction fields
    # -------------------------------------------------

    # Only fully open airports are acceptable
    if closure_status != "open":
        return OperationalCheckResult(
            False,
            f"Rejected: airport status is {closure_status}",
        )

    # Unsafe airports are always rejected
    if unsafe_status == "unsafe":
        return OperationalCheckResult(
            False,
            "Rejected: airport marked unsafe",
        )

    # Restricted or military-restricted airports:
    # allowed only for critical emergencies
    if restricted_status in {"restricted", "military_restricted"} and not critical:
        return OperationalCheckResult(
            False,
            f"Rejected: airport is {restricted_status}",
        )

    # Military-only airports:
    # allowed only for critical emergencies
    if civil_military == "military" and not critical:
        return OperationalCheckResult(
            False,
            "Rejected: military-only airport not permitted for this emergency",
        )

    return OperationalCheckResult(True, "Operationally accepted")