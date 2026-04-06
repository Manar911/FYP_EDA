"""
run_demo_scenarios.py

Multi-scenario demonstration for the EDA Core Engine (Increment 2).

Covers 9 realistic emergency diversion scenarios across all 6 emergency types:

    1. FUEL                    — A320,        critical fuel,  over Bahrain Gulf
    2. FUEL                    — B777-300ER,  low fuel,       over Arabian Sea
    3. MEDICAL                 — B737-800,    normal fuel,    over Red Sea
    4. MEDICAL                 — A380-800,    normal fuel,    over Indian Ocean
    5. MECHANICAL              — B737-800,    low fuel,       over Persian Gulf
    6. WEATHER                 — A320,        normal fuel,    over Eastern Mediterranean
    7. TECHNICAL               — B777-300ER,  low fuel,       over East Africa
    8. SECURITY                — A320,        normal fuel,    over Arabian Gulf
    9. OPERATIONAL_CONSTRAINTS — B737-800,    normal fuel,    over North Africa

Each scenario prints:
    - Scenario context (aircraft, position, emergency, fuel)
    - Top 3 diversion airports with scores, distances, and explanations
    - Decision log saved to logs/
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from eda.pipeline import run_pipeline
from eda.scenario import EmergencyType, FuelState
from eda.scenario_builder import build_scenario
from eda.explanation import generate_explanation
from eda.logger import save_decision_report_json


SCENARIOS = [
    {
        "id": 1,
        "label": "Fuel Emergency — A320 over Bahrain Gulf (Critical Fuel)",
        "aircraft_type": "A320",
        "aircraft_lat": 26.2708,
        "aircraft_lon": 50.6336,
        "fuel_state": FuelState.CRITICAL,
        "emergency_type": EmergencyType.FUEL,
    },
    {
        "id": 2,
        "label": "Fuel Emergency — B777-300ER over Arabian Sea (Low Fuel)",
        "aircraft_type": "B777-300ER",
        "aircraft_lat": 20.5,
        "aircraft_lon": 62.0,
        "fuel_state": FuelState.LOW,
        "emergency_type": EmergencyType.FUEL,
    },
    {
        "id": 3,
        "label": "Medical Emergency — B737-800 over Red Sea (Normal Fuel)",
        "aircraft_type": "B737-800",
        "aircraft_lat": 22.0,
        "aircraft_lon": 37.5,
        "fuel_state": FuelState.NORMAL,
        "emergency_type": EmergencyType.MEDICAL,
    },
    {
        "id": 4,
        "label": "Medical Emergency — A380-800 over Indian Ocean (Normal Fuel)",
        "aircraft_type": "A380-800",
        "aircraft_lat": 5.0,
        "aircraft_lon": 73.0,
        "fuel_state": FuelState.NORMAL,
        "emergency_type": EmergencyType.MEDICAL,
    },
    {
        "id": 5,
        "label": "Mechanical Emergency — B737-800 over Persian Gulf (Low Fuel)",
        "aircraft_type": "B737-800",
        "aircraft_lat": 25.5,
        "aircraft_lon": 55.0,
        "fuel_state": FuelState.LOW,
        "emergency_type": EmergencyType.MECHANICAL,
    },
    {
        "id": 6,
        "label": "Weather Emergency — A320 over Eastern Mediterranean (Normal Fuel)",
        "aircraft_type": "A320",
        "aircraft_lat": 34.0,
        "aircraft_lon": 33.0,
        "fuel_state": FuelState.NORMAL,
        "emergency_type": EmergencyType.WEATHER,
    },
    {
        "id": 7,
        "label": "Technical Emergency — B777-300ER over East Africa (Low Fuel)",
        "aircraft_type": "B777-300ER",
        "aircraft_lat": -1.5,
        "aircraft_lon": 40.0,
        "fuel_state": FuelState.LOW,
        "emergency_type": EmergencyType.TECHNICAL,
    },
    {
        "id": 8,
        "label": "Security Emergency — A320 over Arabian Gulf (Normal Fuel)",
        "aircraft_type": "A320",
        "aircraft_lat": 24.5,
        "aircraft_lon": 54.0,
        "fuel_state": FuelState.NORMAL,
        "emergency_type": EmergencyType.SECURITY,
    },
    {
        "id": 9,
        "label": "Operational Constraints — B737-800 over North Africa (Normal Fuel)",
        "aircraft_type": "B737-800",
        "aircraft_lat": 30.0,
        "aircraft_lon": 20.0,
        "fuel_state": FuelState.NORMAL,
        "emergency_type": EmergencyType.OPERATIONAL_CONSTRAINTS,
    },
]


def run_scenario(scenario_def: dict) -> None:
    """Runs a single scenario, prints results, and saves decision log."""

    print()
    print("=" * 70)
    print(f"  SCENARIO {scenario_def['id']}: {scenario_def['label']}")
    print("=" * 70)

    scenario = build_scenario(
        aircraft_type=scenario_def["aircraft_type"],
        aircraft_lat=scenario_def["aircraft_lat"],
        aircraft_lon=scenario_def["aircraft_lon"],
        fuel_state=scenario_def["fuel_state"],
        emergency_type=scenario_def["emergency_type"],
    )

    print(f"  Aircraft:        {scenario.aircraft_type} ({scenario.aircraft_category})")
    print(f"  Position:        ({scenario.aircraft_lat}, {scenario.aircraft_lon})")
    print(f"  Emergency:       {scenario.emergency_type.value.upper()}")
    print(f"  Fuel state:      {scenario.fuel_state.value.upper()}")
    print(f"  Usable range:    {scenario.usable_range_km:.0f} km")
    print(f"  Required runway: {scenario.required_runway_m} m")
    print("-" * 70)

    report = run_pipeline(scenario)

    # Save decision log using existing logger
    log_path = save_decision_report_json(report)
    print(f"  Log saved to:    {log_path}")

    print(f"  Airports loaded:   {report.total_airports}")
    print(f"  Feasible airports: {len(report.feasible)}")
    print("-" * 70)

    if not report.ranked_top:
        print("  No feasible diversion airports found.")
        print("=" * 70)
        return

    print("  TOP DIVERSION OPTIONS:")
    print()

    for i, opt in enumerate(report.ranked_top, start=1):
        a = opt.airport
        f = opt.features
        explanation = generate_explanation(opt, scenario.emergency_type)

        print(f"  {i}. {a.icao} — {a.name} ({a.city}, {a.country})")
        print(f"     Score:          {opt.score:.4f}")
        print(f"     Distance:       {f.distance_km:.1f} km")
        print(f"     Zone:           {opt.distance_zone}")
        print(f"     Runway:         {a.runway_length_m} m (margin: {f.runway_margin_m:+d} m)")
        print(f"     ILS:            {'Yes' if a.has_ils else 'No'}")
        print(f"     Medical:        {'Yes' if a.has_medical else 'No'} ({a.medical_level})")
        print(f"     Rescue:         {'Yes' if a.has_rescue else 'No'} ({a.rescue_category})")
        print(f"     Firefighting:   {'Yes' if a.has_firefighting else 'No'}")
        print(f"     Maintenance:    {'Yes' if a.has_maintenance else 'No'}")
        print(f"     Fuel available: {'Yes' if a.fuel_available else 'No'}")
        print(f"     Weather rep:    {'Yes' if a.weather_reporting else 'No'}")
        print(f"     Open 24h:       {'Yes' if a.open_24h else 'No'}")
        print(f"     Why selected:")
        for reason in explanation.reasons:
            print(f"       • {reason}")
        print()

    print("=" * 70)


def main() -> None:
    print()
    print("=" * 70)
    print("  EDA CORE ENGINE — MULTI-SCENARIO DEMONSTRATION")
    print("  Emergency Diversion Assistant — Increment 2")
    print(f"  Total scenarios: {len(SCENARIOS)}")
    print("=" * 70)

    passed = 0
    failed = 0

    for scenario_def in SCENARIOS:
        try:
            run_scenario(scenario_def)
            passed += 1
        except Exception as e:
            print(f"\n  ERROR in Scenario {scenario_def['id']}: {e}")
            failed += 1

    print()
    print("=" * 70)
    print(f"  DEMO COMPLETE — {passed} passed, {failed} failed")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()