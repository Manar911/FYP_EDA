"""
app.py  —  EDA v3

Main application window.
Handles screen navigation, inference worker, and compact logging.
"""

from __future__ import annotations

from PySide6.QtWidgets import QMainWindow, QStackedWidget, QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt, QTimer

from eda.scenario_builder import build_scenario
from eda.operational_constraints import OperationalConstraints
from eda.logger import decision_report_to_compact_dict

from ui.theme import Colour, Font, Spacing, get_stylesheet
from ui.screens.input_screen import InputScreen
from ui.screens.results_screen import ResultsScreen
from ui.screens.logs_screen import LogsScreen
from ui.workers.inference_worker import InferenceWorker

SCREEN_INPUT   = 0
SCREEN_RESULTS = 1
SCREEN_LOGS    = 2


class EDAMainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("EDA — Emergency Diversion Assistant")
        self.setFixedSize(1024, 600)
        self.setStyleSheet(get_stylesheet())
        self._worker = None
        self._setup_ui()

    def _setup_ui(self):
        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        self._input_screen   = InputScreen()
        self._results_screen = ResultsScreen()
        self._logs_screen    = LogsScreen()

        self._stack.addWidget(self._input_screen)
        self._stack.addWidget(self._results_screen)
        self._stack.addWidget(self._logs_screen)

        self._loading = self._build_loading()
        self._loading.hide()

        self._input_screen.run_requested.connect(self._on_run)
        self._input_screen.logs_requested.connect(
            lambda: self._stack.setCurrentIndex(SCREEN_LOGS)
        )
        self._results_screen.back_requested.connect(
            lambda: self._stack.setCurrentIndex(SCREEN_INPUT)
        )
        self._results_screen.decision_confirmed.connect(self._on_confirmed)
        self._logs_screen.back_requested.connect(
            lambda: self._stack.setCurrentIndex(SCREEN_INPUT)
        )

    def _build_loading(self):
        overlay = QWidget(self)
        overlay.setGeometry(0, 0, 1024, 600)
        overlay.setStyleSheet(f"background-color: rgba(7, 17, 30, 220);")

        layout = QVBoxLayout(overlay)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(Spacing.MD)

        self._spin_lbl = QLabel("◈")
        self._spin_lbl.setStyleSheet(f"""
            color: {Colour.CYAN};
            font-size: 52px;
            background: transparent;
        """)
        self._spin_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._spin_lbl)

        status = QLabel("Analysing diversion options...")
        status.setStyleSheet(f"""
            color: {Colour.CYAN};
            font-size: {Font.SZ_BODY}px;
            font-weight: bold;
            background: transparent;
        """)
        status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(status)

        sub = QLabel("LightGBM inference in progress")
        sub.setStyleSheet(f"""
            color: {Colour.TEXT_SECONDARY};
            font-size: {Font.SZ_SM}px;
            background: transparent;
        """)
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(sub)

        self._spin_chars = ["◈", "◇", "◆", "◇"]
        self._spin_idx   = 0
        self._spin_timer = QTimer()
        self._spin_timer.timeout.connect(self._tick)

        return overlay

    def _tick(self):
        self._spin_idx = (self._spin_idx + 1) % len(self._spin_chars)
        self._spin_lbl.setText(self._spin_chars[self._spin_idx])

    def _on_run(self, aircraft_type, lat, lon, fuel_state, emergency_type,
                closed_airports, restricted_countries, unsafe_airports):
        try:
            scenario = build_scenario(
                aircraft_type=aircraft_type,
                aircraft_lat=lat,
                aircraft_lon=lon,
                fuel_state=fuel_state,
                emergency_type=emergency_type,
            )
        except ValueError as exc:
            self._input_screen._flash(str(exc)[:55])
            return

        constraints = OperationalConstraints(
            closed_airports=closed_airports,
            restricted_countries=restricted_countries,
            unsafe_airports=unsafe_airports,
        )

        self._loading.show()
        self._loading.raise_()
        self._spin_timer.start(250)

        self._worker = InferenceWorker(scenario, constraints)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_done(self, report, explanations):
        self._hide_loading()
        self._results_screen.load_results(report, explanations)
        self._stack.setCurrentIndex(SCREEN_RESULTS)

    def _on_error(self, message):
        self._hide_loading()
        self._input_screen._flash(f"Error: {message[:50]}")

    def _on_confirmed(self, icao, report):
        try:
            log = decision_report_to_compact_dict(report)
            log["confirmed_airport"] = icao
            self._logs_screen.add_entry(log)
        except Exception:
            pass
        self._stack.setCurrentIndex(SCREEN_INPUT)

    def _hide_loading(self):
        self._spin_timer.stop()
        self._loading.hide()