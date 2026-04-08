"""
input_screen.py

Scenario input screen for the EDA embedded UI.

Allows the pilot/dispatcher to select:
- Aircraft type (horizontal scrollable picker)
- Latitude and longitude (numeric steppers)
- Fuel state (3 large touch buttons)
- Emergency type (7 large touch buttons)

Then tap RUN to trigger inference.
No keyboard input required except for lat/lon numeric entry.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QScrollArea, QLineEdit,
    QFrame, QSizePolicy,
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QDoubleValidator

from eda.scenario import EmergencyType, FuelState
from eda.scenario_builder import list_available_aircraft
from ui.theme import Colour, Font, Spacing


# ── Emergency type display labels ─────────────────────────────────────────────
EMERGENCY_LABELS = {
    EmergencyType.FUEL:                  "FUEL",
    EmergencyType.MEDICAL:               "MEDICAL",
    EmergencyType.MECHANICAL:            "MECHANICAL",
    EmergencyType.TECHNICAL:             "TECHNICAL",
    EmergencyType.WEATHER:               "WEATHER",
    EmergencyType.SECURITY:              "SECURITY",
    EmergencyType.OPERATIONAL_CONSTRAINTS: "OPERATIONAL",
}

FUEL_COLOURS = {
    FuelState.NORMAL:   Colour.FUEL_NORMAL,
    FuelState.LOW:      Colour.FUEL_LOW,
    FuelState.CRITICAL: Colour.FUEL_CRITICAL,
}


class InputScreen(QWidget):
    """
    Full input screen. Emits run_requested with the five user inputs
    when the pilot taps RUN.
    """

    run_requested = Signal(str, float, float, object, object)
    # args: aircraft_type, lat, lon, FuelState, EmergencyType

    logs_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._selected_aircraft:  str | None = None
        self._selected_fuel:      FuelState | None = None
        self._selected_emergency: EmergencyType | None = None
        self._aircraft_buttons:   dict[str, QPushButton] = {}
        self._fuel_buttons:       dict[FuelState, QPushButton] = {}
        self._emergency_buttons:  dict[EmergencyType, QPushButton] = {}
        self._setup_ui()

    # ── UI construction ────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_header())
        root.addWidget(self._build_body(), stretch=1)
        root.addWidget(self._build_run_bar())

    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setFixedHeight(Spacing.HEADER_HEIGHT)
        header.setStyleSheet(f"background-color: {Colour.BG_HEADER};")

        layout = QHBoxLayout(header)
        layout.setContentsMargins(Spacing.LG, 0, Spacing.LG, 0)

        title = QLabel("EDA  —  Emergency Diversion Assistant")
        title.setStyleSheet(f"""
            color: {Colour.TEXT_PRIMARY};
            font-size: {Font.SIZE_MEDIUM}px;
            font-weight: bold;
        """)
        layout.addWidget(title, stretch=1)

        logs_btn = QPushButton("LOGS")
        logs_btn.setFixedHeight(36)
        logs_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colour.BACK};
                color: {Colour.TEXT_SECONDARY};
                border: 1px solid {Colour.BORDER_DEFAULT};
                border-radius: 4px;
                font-size: {Font.SIZE_SMALL}px;
                padding: 0 14px;
            }}
            QPushButton:pressed {{
                background-color: {Colour.BACK_HOVER};
            }}
        """)
        logs_btn.clicked.connect(self.logs_requested.emit)
        layout.addWidget(logs_btn)

        return header

    def _build_body(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("border: none;")

        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(Spacing.LG, Spacing.LG, Spacing.LG, Spacing.LG)
        layout.setSpacing(Spacing.LG)

        layout.addWidget(self._build_aircraft_section())
        layout.addWidget(self._build_divider())
        layout.addWidget(self._build_position_fuel_row())
        layout.addWidget(self._build_divider())
        layout.addWidget(self._build_emergency_section())
        layout.addStretch()

        scroll.setWidget(body)
        return scroll

    def _build_aircraft_section(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Spacing.SM)

        label = self._section_label("AIRCRAFT TYPE")
        layout.addWidget(label)

        # Horizontal scrollable row of aircraft buttons
        scroll = QScrollArea()
        scroll.setFixedHeight(80)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setStyleSheet("border: none; background: transparent;")

        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(Spacing.SM)

        for aircraft in list_available_aircraft():
            btn = QPushButton(aircraft)
            btn.setFixedSize(100, 64)
            btn.setStyleSheet(self._picker_btn_style(False))
            btn.clicked.connect(lambda checked, a=aircraft: self._select_aircraft(a))
            self._aircraft_buttons[aircraft] = btn
            row_layout.addWidget(btn)

        row_layout.addStretch()
        row_widget.setLayout(row_layout)
        scroll.setWidget(row_widget)
        layout.addWidget(scroll)

        return container

    def _build_position_fuel_row(self) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Spacing.XL)

        layout.addWidget(self._build_position_section(), stretch=1)
        layout.addWidget(self._build_fuel_section())

        return container

    def _build_position_section(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Spacing.SM)

        layout.addWidget(self._section_label("POSITION"))

        validator = QDoubleValidator(-180.0, 180.0, 6)

        lat_row = QHBoxLayout()
        lat_lbl = QLabel("LAT")
        lat_lbl.setFixedWidth(36)
        lat_lbl.setStyleSheet(f"color: {Colour.TEXT_SECONDARY}; font-size: {Font.SIZE_SMALL}px;")
        self._lat_input = QLineEdit("26.2708")
        self._lat_input.setFixedHeight(48)
        self._lat_input.setValidator(validator)
        lat_row.addWidget(lat_lbl)
        lat_row.addWidget(self._lat_input)
        layout.addLayout(lat_row)

        lon_row = QHBoxLayout()
        lon_lbl = QLabel("LON")
        lon_lbl.setFixedWidth(36)
        lon_lbl.setStyleSheet(f"color: {Colour.TEXT_SECONDARY}; font-size: {Font.SIZE_SMALL}px;")
        self._lon_input = QLineEdit("50.6336")
        self._lon_input.setFixedHeight(48)
        self._lon_input.setValidator(validator)
        lon_row.addWidget(lon_lbl)
        lon_row.addWidget(self._lon_input)
        layout.addLayout(lon_row)

        return container

    def _build_fuel_section(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Spacing.SM)

        layout.addWidget(self._section_label("FUEL STATE"))

        btn_row = QHBoxLayout()
        btn_row.setSpacing(Spacing.SM)

        for state in [FuelState.NORMAL, FuelState.LOW, FuelState.CRITICAL]:
            btn = QPushButton(state.value.upper())
            btn.setFixedSize(100, 64)
            btn.setStyleSheet(self._fuel_btn_style(state, False))
            btn.clicked.connect(lambda checked, s=state: self._select_fuel(s))
            self._fuel_buttons[state] = btn
            btn_row.addWidget(btn)

        layout.addLayout(btn_row)
        return container

    def _build_emergency_section(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Spacing.SM)

        layout.addWidget(self._section_label("EMERGENCY TYPE"))

        grid = QGridLayout()
        grid.setSpacing(Spacing.SM)

        emergencies = list(EMERGENCY_LABELS.items())
        for i, (etype, label) in enumerate(emergencies):
            btn = QPushButton(label)
            btn.setFixedHeight(64)
            btn.setMinimumWidth(130)
            btn.setStyleSheet(self._emergency_btn_style(False))
            btn.clicked.connect(lambda checked, e=etype: self._select_emergency(e))
            self._emergency_buttons[etype] = btn
            grid.addWidget(btn, i // 4, i % 4)

        layout.addLayout(grid)
        return container

    def _build_run_bar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(88)
        bar.setStyleSheet(f"""
            background-color: {Colour.BG_HEADER};
            border-top: 1px solid {Colour.BORDER_DEFAULT};
        """)

        layout = QHBoxLayout(bar)
        layout.setContentsMargins(Spacing.LG, Spacing.MD, Spacing.LG, Spacing.MD)

        self._run_btn = QPushButton("▶   RUN DIVERSION ANALYSIS")
        self._run_btn.setFixedHeight(Spacing.BUTTON_HEIGHT)
        self._run_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colour.ACCENT_BLUE};
                color: {Colour.TEXT_PRIMARY};
                border: none;
                border-radius: {Spacing.BUTTON_RADIUS}px;
                font-size: {Font.SIZE_LARGE}px;
                font-weight: bold;
                letter-spacing: 1px;
            }}
            QPushButton:pressed {{
                background-color: {Colour.ACCENT_BLUE_DIM};
            }}
            QPushButton:disabled {{
                background-color: {Colour.ACCENT_BLUE_DIM};
                color: {Colour.TEXT_MUTED};
            }}
        """)
        self._run_btn.clicked.connect(self._on_run)
        layout.addWidget(self._run_btn)

        return bar

    # ── Selection handlers ─────────────────────────────────────────────────────

    def _select_aircraft(self, aircraft: str) -> None:
        self._selected_aircraft = aircraft
        for name, btn in self._aircraft_buttons.items():
            btn.setStyleSheet(self._picker_btn_style(name == aircraft))

    def _select_fuel(self, state: FuelState) -> None:
        self._selected_fuel = state
        for s, btn in self._fuel_buttons.items():
            btn.setStyleSheet(self._fuel_btn_style(s, s == state))

    def _select_emergency(self, etype: EmergencyType) -> None:
        self._selected_emergency = etype
        for e, btn in self._emergency_buttons.items():
            btn.setStyleSheet(self._emergency_btn_style(e == etype))

    def _on_run(self) -> None:
        if not self._selected_aircraft:
            self._flash_error("Please select an aircraft type.")
            return
        if not self._selected_fuel:
            self._flash_error("Please select a fuel state.")
            return
        if not self._selected_emergency:
            self._flash_error("Please select an emergency type.")
            return

        try:
            lat = float(self._lat_input.text())
            lon = float(self._lon_input.text())
        except ValueError:
            self._flash_error("Invalid position values.")
            return

        self.run_requested.emit(
            self._selected_aircraft,
            lat,
            lon,
            self._selected_fuel,
            self._selected_emergency,
        )

    def _flash_error(self, message: str) -> None:
        original = self._run_btn.text()
        self._run_btn.setText(f"  ⚠  {message}")
        self._run_btn.setEnabled(False)
        from PySide6.QtCore import QTimer
        QTimer.singleShot(2000, lambda: (
            self._run_btn.setText(original),
            self._run_btn.setEnabled(True),
        ))

    # ── Style helpers ──────────────────────────────────────────────────────────

    def _section_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(f"""
            color: {Colour.TEXT_SECONDARY};
            font-size: {Font.SIZE_SMALL}px;
            font-weight: bold;
            letter-spacing: 1px;
        """)
        return lbl

    def _build_divider(self) -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFixedHeight(1)
        line.setStyleSheet(f"background-color: {Colour.DIVIDER};")
        return line

    def _picker_btn_style(self, selected: bool) -> str:
        if selected:
            return f"""
                QPushButton {{
                    background-color: {Colour.ACCENT_BLUE};
                    color: {Colour.TEXT_PRIMARY};
                    border: 1px solid {Colour.ACCENT_BLUE};
                    border-radius: {Spacing.BUTTON_RADIUS}px;
                    font-size: {Font.SIZE_SMALL}px;
                    font-weight: bold;
                }}
            """
        return f"""
            QPushButton {{
                background-color: {Colour.BG_SECONDARY};
                color: {Colour.TEXT_SECONDARY};
                border: 1px solid {Colour.BORDER_DEFAULT};
                border-radius: {Spacing.BUTTON_RADIUS}px;
                font-size: {Font.SIZE_SMALL}px;
            }}
            QPushButton:pressed {{
                background-color: {Colour.BG_INPUT};
                color: {Colour.TEXT_PRIMARY};
            }}
        """

    def _fuel_btn_style(self, state: FuelState, selected: bool) -> str:
        colour = FUEL_COLOURS[state]
        if selected:
            return f"""
                QPushButton {{
                    background-color: {colour};
                    color: {Colour.BG_PRIMARY};
                    border: 2px solid {colour};
                    border-radius: {Spacing.BUTTON_RADIUS}px;
                    font-size: {Font.SIZE_SMALL}px;
                    font-weight: bold;
                }}
            """
        return f"""
            QPushButton {{
                background-color: {Colour.BG_SECONDARY};
                color: {colour};
                border: 1px solid {colour};
                border-radius: {Spacing.BUTTON_RADIUS}px;
                font-size: {Font.SIZE_SMALL}px;
                font-weight: bold;
            }}
            QPushButton:pressed {{
                background-color: {Colour.BG_INPUT};
            }}
        """

    def _emergency_btn_style(self, selected: bool) -> str:
        if selected:
            return f"""
                QPushButton {{
                    background-color: {Colour.EMERGENCY};
                    color: {Colour.TEXT_PRIMARY};
                    border: 2px solid {Colour.EMERGENCY};
                    border-radius: {Spacing.BUTTON_RADIUS}px;
                    font-size: {Font.SIZE_SMALL}px;
                    font-weight: bold;
                }}
            """
        return f"""
            QPushButton {{
                background-color: {Colour.BG_SECONDARY};
                color: {Colour.TEXT_SECONDARY};
                border: 1px solid {Colour.BORDER_DEFAULT};
                border-radius: {Spacing.BUTTON_RADIUS}px;
                font-size: {Font.SIZE_SMALL}px;
            }}
            QPushButton:pressed {{
                background-color: {Colour.BG_INPUT};
                color: {Colour.TEXT_PRIMARY};
            }}
        """