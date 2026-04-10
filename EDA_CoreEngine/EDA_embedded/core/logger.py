"""
logger.py  —  EDA Embedded

Compact logging for Raspberry Pi runtime.

Changes from development version:
- Default mode is 'compact' (not 'full')
- Default output_dir points to EDA_embedded/logs/ using absolute path
- Max 200 log files retained automatically
- Full mode still available for development use

"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from explanation import generate_explanation
from models import DecisionReport

LogMode = Literal["full", "compact"]

#  Default log directory — absolute path to EDA_embedded/logs/ 
_HERE        = Path(__file__).resolve().parent   # EDA_embedded/core/
_ROOT        = _HERE.parent                      # EDA_embedded/
DEFAULT_LOGS = _ROOT / "logs"


def _timestamp_for_filename() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H-%M-%S")


def _build_explanation(report: DecisionReport, ranked) -> dict[str, Any]:
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
        "airport_icao":       explanation.airport_icao,
        "feasibility_reasons": explanation.feasibility_reasons,
        "ranking_reasons":    explanation.ranking_reasons,
        "summary":            explanation.summary,
        "caution":            explanation.caution,
    }


def _scenario_dict(report: DecisionReport) -> dict[str, Any]:
    return {
        "aircraft_lat":               report.scenario.aircraft_lat,
        "aircraft_lon":               report.scenario.aircraft_lon,
        "required_runway_m":          report.scenario.required_runway_m,
        "emergency_type":             report.scenario.emergency_type.value,
        "aircraft_type":              report.scenario.aircraft_type,
        "aircraft_category":          report.scenario.aircraft_category,
        "fuel_state":                 report.scenario.fuel_state.value,
        "fuel_multiplier":            report.scenario.fuel_multiplier,
        "max_range_km":               report.scenario.max_range_km,
        "aircraft_adjusted_range_km": report.scenario.aircraft_adjusted_range_km,
        "usable_range_km":            report.scenario.usable_range_km,
        "extended_range_km":          report.scenario.extended_range_km,
        "binding_side":               report.scenario.binding_side.value,
    }


def decision_report_to_full_dict(report: DecisionReport) -> dict[str, Any]:
    """Full log — development/debugging use."""
    ranked_with_explanations = []
    for ranked in report.ranked_top:
        ranked_with_explanations.append({
            "airport":       asdict(ranked.airport),
            "features":      asdict(ranked.features),
            "score":         ranked.score,
            "distance_zone": ranked.distance_zone,
            "explanation":   _build_explanation(report, ranked),
        })

    evaluated = [
        {"airport": asdict(i.airport), "features": asdict(i.features),
         "feasibility": asdict(i.feasibility)}
        for i in report.evaluated
    ]
    feasible = [
        {"airport": asdict(a), "features": asdict(f), "distance_zone": z}
        for a, f, z in report.feasible
    ]

    return {
        "timestamp":       datetime.now().isoformat(),
        "log_mode":        "full",
        "scenario":        _scenario_dict(report),
        "selected_option": ranked_with_explanations[0] if ranked_with_explanations else None,
        "total_airports":  report.total_airports,
        "evaluated":       evaluated,
        "feasible":        feasible,
        "ranked_top":      ranked_with_explanations,
    }


def decision_report_to_compact_dict(report: DecisionReport) -> dict[str, Any]:
    """Compact log — embedded Pi runtime use."""
    top_ranked = []
    for ranked in report.ranked_top:
        explanation = _build_explanation(report, ranked)
        top_ranked.append({
            "airport_icao":   ranked.airport.icao,
            "airport_name":   ranked.airport.name,
            "city":           ranked.airport.city,
            "country":        ranked.airport.country,
            "score":          ranked.score,
            "distance_km":    ranked.features.distance_km,
            "runway_margin_m": ranked.features.runway_margin_m,
            "distance_zone":  ranked.distance_zone,
            "summary":        explanation["summary"],
        })

    selected = None
    if report.ranked_top:
        best = report.ranked_top[0]
        selected = {
            "airport_icao":   best.airport.icao,
            "airport_name":   best.airport.name,
            "city":           best.airport.city,
            "country":        best.airport.country,
            "score":          best.score,
            "distance_km":    best.features.distance_km,
            "runway_margin_m": best.features.runway_margin_m,
            "distance_zone":  best.distance_zone,
            "explanation":    _build_explanation(report, best),
        }

    return {
        "timestamp":             datetime.now().isoformat(),
        "log_mode":              "compact",
        "scenario": {
            "emergency_type":  report.scenario.emergency_type.value,
            "aircraft_type":   report.scenario.aircraft_type,
            "aircraft_category": report.scenario.aircraft_category,
            "fuel_state":      report.scenario.fuel_state.value,
            "aircraft_lat":    report.scenario.aircraft_lat,
            "aircraft_lon":    report.scenario.aircraft_lon,
            "usable_range_km": report.scenario.usable_range_km,
            "required_runway_m": report.scenario.required_runway_m,
        },
        "total_airports":        report.total_airports,
        "feasible_airports_count": len(report.feasible),
        "selected_option":       selected,
        "top_ranked":            top_ranked,
    }


def _enforce_log_retention(output_dir: Path, max_logs: int) -> None:
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
            pass


def save_decision_report_json(
    report: DecisionReport,
    *,
    output_dir: Path | str | None = None,
    mode: LogMode = "compact",
    max_logs: int | None = 200,
) -> str:
    """
    Save one decision report to a JSON file.

    Defaults:
        mode      = 'compact'  (lightweight for Pi)
        output_dir = EDA_embedded/logs/  (absolute path)
        max_logs  = 200        (auto-evict oldest beyond 200)
    """
    if output_dir is None:
        out_dir = DEFAULT_LOGS
    else:
        out_dir = Path(output_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = _timestamp_for_filename()
    emergency = report.scenario.emergency_type.value
    filename  = f"{timestamp}_{emergency}_{mode}.json"
    path      = out_dir / filename

    if mode == "full":
        payload = decision_report_to_full_dict(report)
    elif mode == "compact":
        payload = decision_report_to_compact_dict(report)
    else:
        raise ValueError(f"Unsupported mode '{mode}'. Use 'full' or 'compact'.")

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if max_logs is not None:
        _enforce_log_retention(out_dir, max_logs)

    return str(path)