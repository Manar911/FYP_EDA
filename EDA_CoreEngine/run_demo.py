"""
run_demo.py

Small demo runner for the EDA Core Engine (Increment 1).

This script creates a sample Scenario, runs the pipeline, and prints:
- Top K diversion airport options
- Their score + key features
"""

from __future__ import annotations

from eda.pipeline import run_pipeline
from eda.scenario import Scenario, EmergencyType


def main() -> None:
    # Demo scenario
    scenario = Scenario(
        aircraft_lat=26.2708,        # Bahrain area example
        aircraft_lon=50.6336,
        required_runway_m=3000,
        emergency_type=EmergencyType.FUEL,
    )

    report = run_pipeline(
        scenario,
        top_k=3,
        max_range_km=800.0,  # Increment 1 assumption (tune for demo)
    )

    print("=" * 60)
    print("EDA Core Engine Demo (Increment 1)")
    print("=" * 60)
    print(f"Emergency: {report.scenario.emergency_type.value}")
    print(f"Aircraft position: ({report.scenario.aircraft_lat}, {report.scenario.aircraft_lon})")
    print(f"Required runway: {report.scenario.required_runway_m} m")
    print("-" * 60)
    print(f"Total airports loaded: {report.total_airports}")
    print(f"Feasible airports: {len(report.feasible)}")
    print("-" * 60)

    if not report.ranked_top:
        print("No feasible diversion airports found under current constraints.")
        return

    print("Top diversion options:")
    for i, opt in enumerate(report.ranked_top, start=1):
        a = opt.airport
        f = opt.features

        print()
        print(f"{i}. {a.icao} — {a.name} ({a.country})")
        print(f"   Score: {opt.score:.4f}")
        print(f"   Distance: {f.distance_km:.1f} km")
        print(f"   Runway length: {a.runway_length_m} m")
        print(f"   Runway margin: {f.runway_margin_m:+d} m")
        print(f"   Medical: {int(a.has_medical)} | Rescue: {int(a.has_rescue)}")

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()