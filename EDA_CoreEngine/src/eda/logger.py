"""
logger.py

JSON logging for the EDA Core Engine (Increment 1).

Purpose:
- Save a full structured decision log for each scenario run.
- Support traceability, reproducibility, debugging, and future dataset use.

Each scenario run is stored as one JSON file containing:
- scenario input
- evaluated airports
- feasible airports
- ranked results
- explanations
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from eda.explanation import generate_explanation
from eda.models import DecisionReport


def _timestamp_for_filename() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H-%M-%S")


def decision_report_to_dict(report: DecisionReport) -> dict[str, Any]:
    """
    Convert DecisionReport into a JSON-serializable dictionary.
    """
    ranked_with_explanations = []

    for ranked in report.ranked_top:
        explanation = generate_explanation(ranked, report.scenario.emergency_type)

        ranked_with_explanations.append(
            {
                "airport": asdict(ranked.airport),
                "features": asdict(ranked.features),
                "score": ranked.score,
                "explanation": explanation.reasons,
            }
        )

    evaluated = []
    for item in report.evaluated:
        evaluated.append(
            {
                "airport": asdict(item.airport),
                "features": asdict(item.features),
                "feasibility": asdict(item.feasibility),
            }
        )

    feasible = []
    for airport, features in report.feasible:
        feasible.append(
            {
                "airport": asdict(airport),
                "features": asdict(features),
            }
        )

    return {
        "scenario": {
            "aircraft_lat": report.scenario.aircraft_lat,
            "aircraft_lon": report.scenario.aircraft_lon,
            "required_runway_m": report.scenario.required_runway_m,
            "emergency_type": report.scenario.emergency_type.value,
        },
        "total_airports": report.total_airports,
        "evaluated": evaluated,
        "feasible": feasible,
        "ranked_top": ranked_with_explanations,
    }


def save_decision_report_json(
    report: DecisionReport,
    *,
    output_dir: str = "logs",
) -> str:
    """
    Save one scenario decision report to a JSON file.

    Returns:
        The path to the saved JSON file.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = _timestamp_for_filename()
    emergency = report.scenario.emergency_type.value
    filename = f"{timestamp}_{emergency}.json"
    path = out_dir / filename

    payload = decision_report_to_dict(report)

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return str(path)