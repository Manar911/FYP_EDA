"""
run_demo.py

Small demo runner for the EDA Core Engine (Increment 1).

This script creates a sample Scenario, runs the pipeline, saves a JSON
decision log, and prints:
- Top K diversion airport options
- Their score + key features
- Their explanation reasons
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "src"))


from eda.airport_db import load_airports
from eda.pipeline import run_pipeline
from eda.scenario import EmergencyType, FuelState
from eda.scenario_builder import build_scenario, list_available_aircraft 
from eda.explanation import generate_explanation
from eda.logger import save_decision_report_json


def main() -> None:
    airports = load_airports()

    # Build scenario from minimal user inputs.
    # Aircraft performance data is loaded automatically
    # from the aircraft profiles database — no manual
    # hardcoding of range, runway, or fuel values required.
    print(f"Available aircraft: {list_available_aircraft()}")
    print()

    scenario = build_scenario(
        aircraft_type="A320",
        aircraft_lat=26.2708,
        aircraft_lon=50.6336,
        fuel_state=FuelState.LOW,
        emergency_type=EmergencyType.FUEL,
    )

    print(f"Aircraft:        {scenario.aircraft_type} ({scenario.aircraft_category})")
    print(f"Fuel state:      {scenario.fuel_state.value}")
    print(f"Usable range:    {scenario.usable_range_km} km")
    print(f"Required runway: {scenario.required_runway_m} m")
    print()

    report = run_pipeline(scenario)
    log_path = save_decision_report_json(report)

    print("=" * 60)
    print("EDA Core Engine Demo (Increment 2)")
    print("=" * 60)
    print(f"Emergency: {report.scenario.emergency_type.value}")
    print(f"Aircraft position: ({report.scenario.aircraft_lat}, {report.scenario.aircraft_lon})")
    print(f"Required runway: {report.scenario.required_runway_m} m")
    print(f"Decision log saved to: {log_path}")
    print("-" * 60)
    print(f"Total airports loaded: {report.total_airports}")
    print(f"Feasible airports: {len(report.feasible)}")
    print("-" * 60)

    if not report.ranked_top:
        print("No feasible diversion airports found under current constraints.")
        print("=" * 60)
        return

    print("Top diversion options:")
    for i, opt in enumerate(report.ranked_top, start=1):
        a = opt.airport
        f = opt.features
        explanation = generate_explanation(opt, report.scenario.emergency_type)

        print()
        print(f"{i}. {a.icao} — {a.name} ({a.country})")
        print(f"   Score: {opt.score:.4f}")
        print(f"   Distance: {f.distance_km:.1f} km")
        print(f"   Runway length: {a.runway_length_m} m")
        print(f"   Runway margin: {f.runway_margin_m:+d} m")
        print(f"   Medical: {'Yes' if a.has_medical else 'No'} | Rescue: {'Yes' if a.has_rescue else 'No'}")
        print("   Why this airport:")
        for reason in explanation.reasons:
            print(f"   • {reason}")

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()