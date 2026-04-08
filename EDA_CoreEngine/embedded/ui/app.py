"""
app.py

Main application window for the EDA embedded UI.

Manages:
- QStackedWidget screen navigation (Input → Results → Logs)
- InferenceWorker lifecycle
- Compact logging to LogsScreen buffer
- Loading overlay during inference
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QMainWindow, QStackedWidget, QWidget,
    QVBoxLayout, QLabel,
)
from PySide6.QtCore import Qt, QTimer

from eda.scenario_builder import build_scenario
from eda.logger import decision_report_to_compact_dict

from ui.theme import Colour, Font, Spacing, get_stylesheet
from ui.screens.input_screen import InputScreen
from ui.screens.results_screen import ResultsScreen
from ui.screens.logs_screen import LogsScreen
from ui.workers.inference_worker import InferenceWorker


# ── Screen indices ─────────────────────────────────────────────────────────────
SCREEN_INPUT   = 0
SCREEN_RESULTS = 1
SCREEN_LOGS    = 2


class EDAMainWindow(QMainWindow):
    """
    Root application window.
    Fixed size matching the 7-inch display at 1024x600.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("EDA — Emergency Diversion Assistant")
        self.setFixedSize(1024, 600)
        self.setStyleSheet(get_stylesheet())

        self._worker: InferenceWorker | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        # ── Stack ──────────────────────────────────────────────────
        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        # ── Screens ────────────────────────────────────────────────
        self._input_screen   = InputScreen()
        self._results_screen = ResultsScreen()
        self._logs_screen    = LogsScreen()

        self._stack.addWidget(self._input_screen)   # index 0
        self._stack.addWidget(self._results_screen) # index 1
        self._stack.addWidget(self._logs_screen)    # index 2

        # ── Loading overlay ────────────────────────────────────────
        self._loading_overlay = self._build_loading_overlay()
        self._loading_overlay.hide()

        # ── Connections ────────────────────────────────────────────
        self._input_screen.run_requested.connect(self._on_run_requested)
        self._input_screen.logs_requested.connect(
            lambda: self._stack.setCurrentIndex(SCREEN_LOGS)
        )

        self._results_screen.back_requested.connect(
            lambda: self._stack.setCurrentIndex(SCREEN_INPUT)
        )
        self._results_screen.decision_confirmed.connect(
            self._on_decision_confirmed
        )

        self._logs_screen.back_requested.connect(
            lambda: self._stack.setCurrentIndex(SCREEN_INPUT)
        )

    def _build_loading_overlay(self) -> QWidget:
        """
        Semi-transparent overlay shown during inference.
        """
        overlay = QWidget(self)
        overlay.setGeometry(0, 0, 1024, 600)
        overlay.setStyleSheet(f"""
            background-color: rgba(13, 17, 23, 210);
        """)

        layout = QVBoxLayout(overlay)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        spinner = QLabel("⟳")
        spinner.setStyleSheet(f"""
            color: {Colour.ACCENT_BLUE};
            font-size: 48px;
        """)
        spinner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(spinner)

        msg = QLabel("Analysing diversion options...")
        msg.setStyleSheet(f"""
            color: {Colour.TEXT_SECONDARY};
            font-size: {Font.SIZE_MEDIUM}px;
        """)
        msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(msg)

        # Animate spinner rotation
        self._spinner_label = spinner
        self._spinner_angle = 0
        self._spinner_timer = QTimer()
        self._spinner_timer.timeout.connect(self._rotate_spinner)

        return overlay

    def _rotate_spinner(self) -> None:
        chars = ["⟳", "↻", "⟳", "↺"]
        self._spinner_angle = (self._spinner_angle + 1) % len(chars)
        self._spinner_label.setText(chars[self._spinner_angle])

    # ── Signal handlers ────────────────────────────────────────────────────────

    def _on_run_requested(
        self,
        aircraft_type: str,
        lat: float,
        lon: float,
        fuel_state,
        emergency_type,
    ) -> None:
        """
        Called when pilot taps RUN on the input screen.
        Builds scenario and starts background inference.
        """
        try:
            scenario = build_scenario(
                aircraft_type=aircraft_type,
                aircraft_lat=lat,
                aircraft_lon=lon,
                fuel_state=fuel_state,
                emergency_type=emergency_type,
            )
        except ValueError as exc:
            # Show error on input screen (invalid aircraft type etc.)
            self._input_screen._flash_error(str(exc)[:60])
            return

        # Show loading overlay
        self._loading_overlay.show()
        self._loading_overlay.raise_()
        self._spinner_timer.start(300)

        # Start background inference worker
        self._worker = InferenceWorker(scenario)
        self._worker.finished.connect(self._on_inference_finished)
        self._worker.error.connect(self._on_inference_error)
        self._worker.start()

    def _on_inference_finished(self, report, explanations) -> None:
        """
        Called when inference completes successfully.
        """
        self._hide_loading()
        self._results_screen.load_results(report, explanations)
        self._stack.setCurrentIndex(SCREEN_RESULTS)

    def _on_inference_error(self, message: str) -> None:
        """
        Called when inference fails.
        """
        self._hide_loading()
        self._input_screen._flash_error(f"Error: {message[:50]}")

    def _on_decision_confirmed(self, icao: str, report) -> None:
        """
        Called when pilot confirms a diversion airport.
        Logs the compact decision and navigates back to input.
        """
        try:
            compact_log = decision_report_to_compact_dict(report)
            # Add confirmed airport marker
            compact_log["confirmed_airport"] = icao
            self._logs_screen.add_entry(compact_log)
        except Exception:
            pass  # Logging failure must not crash the system

        # Return to input screen for next scenario
        self._stack.setCurrentIndex(SCREEN_INPUT)

    def _hide_loading(self) -> None:
        self._spinner_timer.stop()
        self._loading_overlay.hide()