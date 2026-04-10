"""
app.py  —  EDA 

Main application window.
Handles screen navigation, inference worker, compact logging,
and startup integrity verification.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QMainWindow, QStackedWidget, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
)
from PySide6.QtCore import Qt, QTimer

from scenario_builder import build_scenario
from operational_constraints import OperationalConstraints
from logger import decision_report_to_compact_dict, save_decision_report_json
from integrity import check_integrity

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
        self._worker   = None
        self._use_ml   = True
        self._warnings = []
        self._setup_ui()

    def _setup_ui(self):
        result = check_integrity()
        self._use_ml   = True
        self._warnings = result.warnings

        model_failed   = any("lightgbm_model" in w for w in result.warnings)
        airport_failed = any("airports_csv"   in w for w in result.warnings)

        if model_failed:
            print("WARNING: ML model integrity check failed.")
            print("  LightGBM disabled — using deterministic ranking.")
            self._use_ml = False

        if airport_failed:
            print("WARNING: Airport database integrity check failed.")

        if not result.warnings:
            print("Integrity check passed — all artifacts verified.")

        central = QWidget()
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        self.setCentralWidget(central)

        self._banner = self._build_banner(model_failed, airport_failed)
        if self._banner:
            root_layout.addWidget(self._banner)

        self._stack = QStackedWidget()
        root_layout.addWidget(self._stack)

        self._input_screen   = InputScreen(
            use_ml=self._use_ml,
            integrity_ok=not result.warnings,
        )
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

    def _build_banner(self, model_failed, airport_failed):
        if not model_failed and not airport_failed:
            return None
        msg = (
            "⚠   ML model integrity check failed — running in deterministic mode"
            if model_failed
            else "⚠   Airport database integrity check failed — results may be unreliable"
        )
        banner = QWidget()
        banner.setFixedHeight(36)
        banner.setStyleSheet(f"background-color: {Colour.AMBER_BG}; border-bottom: 1px solid {Colour.AMBER_DIM};")
        lay = QHBoxLayout(banner)
        lay.setContentsMargins(Spacing.LG, 0, Spacing.LG, 0)
        lbl = QLabel(msg)
        lbl.setStyleSheet(f"color: {Colour.AMBER}; font-size: {Font.SZ_SM}px; font-weight: bold; background: transparent;")
        lay.addWidget(lbl, stretch=1)
        dismiss = QPushButton("✕")
        dismiss.setFixedSize(28, 28)
        dismiss.setStyleSheet(f"QPushButton {{ background: transparent; color: {Colour.AMBER}; border: none; font-size: {Font.SZ_MD}px; }} QPushButton:pressed {{ color: {Colour.TEXT_PRIMARY}; }}")
        dismiss.clicked.connect(banner.hide)
        lay.addWidget(dismiss)
        return banner

    def _build_loading(self):
        overlay = QWidget(self)
        overlay.setGeometry(0, 0, 1024, 600)
        overlay.setStyleSheet("background-color: rgba(7, 17, 30, 220);")
        layout = QVBoxLayout(overlay)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(Spacing.MD)
        self._spin_lbl = QLabel("◈")
        self._spin_lbl.setStyleSheet(f"color: {Colour.CYAN}; font-size: 52px; background: transparent;")
        self._spin_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._spin_lbl)
        status = QLabel("Analysing diversion options...")
        status.setStyleSheet(f"color: {Colour.CYAN}; font-size: {Font.SZ_BODY}px; font-weight: bold; background: transparent;")
        status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(status)
        ranker_msg = "Deterministic ranking active (ML disabled)" if not self._use_ml else "LightGBM inference in progress"
        sub = QLabel(ranker_msg)
        sub.setStyleSheet(f"color: {Colour.TEXT_SECONDARY}; font-size: {Font.SZ_SM}px; background: transparent;")
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

        self._worker = InferenceWorker(scenario, constraints, use_ml=self._use_ml)
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
            # Add to in-memory UI log
            log = decision_report_to_compact_dict(report)
            log["confirmed_airport"] = icao
            self._logs_screen.add_entry(log)

            # Save to disk in EDA_embedded/logs/
            save_decision_report_json(
                report,
                mode="compact",
                max_logs=200,
            )
        except Exception as exc:
            print(f"Logging error: {exc}")

        self._stack.setCurrentIndex(SCREEN_INPUT)

    def _hide_loading(self):
        self._spin_timer.stop()
        self._loading.hide()