"""
inference_worker.py  —  EDA 

Background QThread worker for ML inference.

Updated:
    - Accepts use_ml flag from app.py integrity check
    - When use_ml=False: passes flag to run_pipeline which skips
      model loading and uses deterministic ranking directly
"""

from __future__ import annotations

from typing import List

from PySide6.QtCore import QThread, Signal

from core.pipeline import run_pipeline
from core.scenario import Scenario
from core.explanation import generate_explanation, Explanation
from core.models import DecisionReport
from core.operational_constraints import OperationalConstraints


class InferenceWorker(QThread):

    finished = Signal(object, list)
    error    = Signal(str)

    def __init__(
        self,
        scenario: Scenario,
        constraints: OperationalConstraints | None = None,
        use_ml: bool = True,
    ) -> None:
        super().__init__()
        self.scenario    = scenario
        self.constraints = constraints or OperationalConstraints()
        self.use_ml      = use_ml

    def run(self) -> None:
        try:
            # Pass use_ml to pipeline — pipeline enforces the actual behaviour
            report: DecisionReport = run_pipeline(
                self.scenario,
                constraints=self.constraints,
                use_ml=self.use_ml,
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