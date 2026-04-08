"""
inference_worker.py

Background QThread worker for ML inference.

Runs the full EDA pipeline in a separate thread so the UI
remains responsive during model inference and explanation generation.

Emits:
    finished(report, explanations) — inference completed successfully
    error(message)                 — inference failed with error message
"""

from __future__ import annotations

from typing import List, Tuple, Any

from PySide6.QtCore import QThread, Signal

from eda.pipeline import run_pipeline
from eda.scenario import Scenario
from eda.explanation import generate_explanation, Explanation
from eda.models import DecisionReport


class InferenceWorker(QThread):
    """
    Runs run_pipeline() and generate_explanation() in a background thread.

    Usage:
        worker = InferenceWorker(scenario)
        worker.finished.connect(on_results)
        worker.error.connect(on_error)
        worker.start()
    """

    # Emitted when inference completes successfully
    # Arguments: (report, list of Explanation objects)
    finished = Signal(object, list)

    # Emitted when inference fails
    # Argument: error message string
    error = Signal(str)

    def __init__(self, scenario: Scenario) -> None:
        super().__init__()
        self.scenario = scenario

    def run(self) -> None:
        """
        Executes in the background thread.
        Runs pipeline then generates explanations for all ranked options.
        """
        try:
            # Run the full ML pipeline
            report: DecisionReport = run_pipeline(self.scenario)

            # Generate explanations for each ranked option
            explanations: List[Explanation] = []

            for ranked in report.ranked_top:
                # Find the feasibility reason for this airport
                feasibility_reason = next(
                    (
                        item.feasibility.reason
                        for item in report.evaluated
                        if item.airport.icao == ranked.airport.icao
                    ),
                    "Accepted as feasible",
                )

                explanation = generate_explanation(
                    ranked,
                    report.scenario.emergency_type,
                    feasibility_reason,
                )
                explanations.append(explanation)

            self.finished.emit(report, explanations)

        except Exception as exc:
            self.error.emit(str(exc))