"""
logger.py

JSON logging for the EDA Core Engine.

Purpose:
- Save structured decision logs for each scenario run.
- Support traceability, reproducibility, debugging, and future analysis.
- Provide two logging modes:
    1. full    -> detailed development/debug log
    2. compact -> lightweight embedded log for Raspberry Pi runtime

Design note:
- Full mode is intended for laptop development, testing, and academic evidence.
- Compact mode is intended for embedded runtime use where storage efficiency matters.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from eda.explanation import generate_explanation
from eda.models import DecisionReport

LogMode = Literal["full", "compact"]


def _timestamp_for_filename() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H-%M-%S")


def _build_explanation(report: DecisionReport, ranked) -> dict[str, Any]:
    """
    Build the structured explanation dictionary for one ranked option.
    """
    feasibility_reason = next(
        item.feasibility.reason
        for item in report.evaluated
        if item.airport.icao == ranked.airport.icao
    )

    explanation = generate_explanation(
        ranked,
        report.scenario.emergency_type,
        feasibility_reason,
    )

    return {
        "airport_icao": explanation.airport_icao,
        "feasibility_reasons": explanation.feasibility_reasons,
        "ranking_reasons": explanation.ranking_reasons,
        "summary": explanation.summary,
        "caution": explanation.caution,
    }


def _scenario_dict(report: DecisionReport) -> dict[str, Any]:
    """
    Serialize the scenario in a consistent reusable format.
    """
    return {
        "aircraft_lat": report.scenario.aircraft_lat,
        "aircraft_lon": report.scenario.aircraft_lon,
        "required_runway_m": report.scenario.required_runway_m,
        "emergency_type": report.scenario.emergency_type.value,
        "aircraft_type": report.scenario.aircraft_type,
        "aircraft_category": report.scenario.aircraft_category,
        "fuel_state": report.scenario.fuel_state.value,
        "fuel_multiplier": report.scenario.fuel_multiplier,
        "max_range_km": report.scenario.max_range_km,
        "aircraft_adjusted_range_km": report.scenario.aircraft_adjusted_range_km,
        "usable_range_km": report.scenario.usable_range_km,
        "extended_range_km": report.scenario.extended_range_km,
        "binding_side": report.scenario.binding_side.value,
    }


def decision_report_to_full_dict(report: DecisionReport) -> dict[str, Any]:
    """
    Convert DecisionReport into a full JSON-serializable dictionary.

    Intended for:
    - laptop development
    - debugging
    - traceability
    - dissertation evidence
    """
    ranked_with_explanations = []

    for ranked in report.ranked_top:
        ranked_with_explanations.append(
            {
                "airport": asdict(ranked.airport),
                "features": asdict(ranked.features),
                "score": ranked.score,
                "distance_zone": ranked.distance_zone,
                "explanation": _build_explanation(report, ranked),
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
    for airport, features, distance_zone in report.feasible:
        feasible.append(
            {
                "airport": asdict(airport),
                "features": asdict(features),
                "distance_zone": distance_zone,
            }
        )

    return {
        "timestamp": datetime.now().isoformat(),
        "log_mode": "full",
        "scenario": _scenario_dict(report),
        "selected_option": ranked_with_explanations[0] if ranked_with_explanations else None,
        "total_airports": report.total_airports,
        "evaluated": evaluated,
        "feasible": feasible,
        "ranked_top": ranked_with_explanations,
    }


def decision_report_to_compact_dict(report: DecisionReport) -> dict[str, Any]:
    """
    Convert DecisionReport into a compact JSON-serializable dictionary.

    Intended for:
    - Raspberry Pi runtime
    - storage-efficient embedded logging
    - UI history / lightweight traceability
    """
    top_ranked = []

    for ranked in report.ranked_top:
        explanation = _build_explanation(report, ranked)

        top_ranked.append(
            {
                "airport_icao": ranked.airport.icao,
                "airport_name": ranked.airport.name,
                "city": ranked.airport.city,
                "country": ranked.airport.country,
                "score": ranked.score,
                "distance_km": ranked.features.distance_km,
                "runway_margin_m": ranked.features.runway_margin_m,
                "distance_zone": ranked.distance_zone,
                "summary": explanation["summary"],
            }
        )

    selected = None
    if report.ranked_top:
        best = report.ranked_top[0]
        best_explanation = _build_explanation(report, best)

        selected = {
            "airport_icao": best.airport.icao,
            "airport_name": best.airport.name,
            "city": best.airport.city,
            "country": best.airport.country,
            "score": best.score,
            "distance_km": best.features.distance_km,
            "runway_margin_m": best.features.runway_margin_m,
            "distance_zone": best.distance_zone,
            "explanation": best_explanation,
        }

    return {
        "timestamp": datetime.now().isoformat(),
        "log_mode": "compact",
        "scenario": {
            "emergency_type": report.scenario.emergency_type.value,
            "aircraft_type": report.scenario.aircraft_type,
            "aircraft_category": report.scenario.aircraft_category,
            "fuel_state": report.scenario.fuel_state.value,
            "aircraft_lat": report.scenario.aircraft_lat,
            "aircraft_lon": report.scenario.aircraft_lon,
            "usable_range_km": report.scenario.usable_range_km,
            "required_runway_m": report.scenario.required_runway_m,
        },
        "total_airports": report.total_airports,
        "feasible_airports_count": len(report.feasible),
        "selected_option": selected,
        "top_ranked": top_ranked,
    }


def _enforce_log_retention(output_dir: Path, max_logs: int) -> None:
    """
    Keep only the newest max_logs JSON files in the output directory.
    Older log files are deleted.

    This is mainly useful for embedded runtime logging.
    """
    if max_logs <= 0:
        return

    json_files = sorted(
        output_dir.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    for old_file in json_files[max_logs:]:
        try:
            old_file.unlink()
        except OSError:
            # Ignore deletion failures to avoid crashing runtime logging
            pass


def save_decision_report_json(
    report: DecisionReport,
    *,
    output_dir: str = "logs",
    mode: LogMode = "full", # "full" = detailed logs (dev), "compact" = lightweight logs (embedded)
    max_logs: int | None = None,
) -> str:
    """
    Save one scenario decision report to a JSON file.

    Args:
        report: Decision report to log.
        output_dir: Directory where logs are saved.
        mode:
            - "full"    -> detailed development log
            - "compact" -> lightweight embedded log
        max_logs:
            Optional retention limit. If provided, only the newest max_logs
            JSON files are kept in output_dir.

    Returns:
        The path to the saved JSON file.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = _timestamp_for_filename()
    emergency = report.scenario.emergency_type.value
    filename = f"{timestamp}_{emergency}_{mode}.json"
    path = out_dir / filename

    if mode == "full":
        payload = decision_report_to_full_dict(report)
    elif mode == "compact":
        payload = decision_report_to_compact_dict(report)
    else:
        raise ValueError(f"Unsupported logging mode '{mode}'. Expected 'full' or 'compact'.")

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if max_logs is not None:
        _enforce_log_retention(out_dir, max_logs)

    return str(path)