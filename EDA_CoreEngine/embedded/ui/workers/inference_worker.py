"""
inference_worker.py

Background QThread worker for ML inference.
Updated to accept OperationalConstraints for dynamic exclusions.
"""

from __future__ import annotations

from typing import List

from PySide6.QtCore import QThread, Signal

from eda.pipeline import run_pipeline
from eda.scenario import Scenario
from eda.explanation import generate_explanation, Explanation
from eda.models import DecisionReport
from eda.operational_constraints import OperationalConstraints


class InferenceWorker(QThread):

    finished = Signal(object, list)
    error    = Signal(str)

    def __init__(
        self,
        scenario: Scenario,
        constraints: OperationalConstraints | None = None,
    ) -> None:
        super().__init__()
        self.scenario    = scenario
        self.constraints = constraints or OperationalConstraints()

    def run(self) -> None:
        try:
            report: DecisionReport = run_pipeline(
                self.scenario,
                constraints=self.constraints,
            )

            explanations: List[Explanation] = []
            for ranked in report.ranked_top:
                feasibility_reason = next(
                    (
                        item.feasibility.reason
                        for item in report.evaluated
                        if item.airport.icao == ranked.airport.icao
                    ),
                    "Accepted as feasible",
                )
                explanations.append(
                    generate_explanation(
                        ranked,
                        report.scenario.emergency_type,
                        feasibility_reason,
                    )
                )

            self.finished.emit(report, explanations)

        except Exception as exc:
            self.error.emit(str(exc))