"""
input_screen.py  —  EDA 

Fixes:
1. Map pointer accuracy — uses devicePixelRatio correction for Windows DPI scaling
2. Footer (Reset + Run buttons) no longer cut off — layout restructured so
   footer is always visible regardless of content height
3. All sections sized to fit 1024x600 with no scrolling needed
4. Removed addStretch() that was pushing content down unpredictably
"""

from __future__ import annotations
from datetime import datetime, timezone

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QScrollArea, QLineEdit,
    QFrame, QListWidget,
)
from PySide6.QtCore import Signal, Qt, QTimer

from eda.scenario import EmergencyType, FuelState
from eda.scenario_builder import list_available_aircraft
from ui.theme import Colour, Font, Spacing
from ui.widgets.numpad_widget import NumpadWidget
from ui.widgets.map_widget import MapWidget
from ui.widgets.icao_keyboard import ICAOKeyboard


EMERGENCY_CFG = {
    EmergencyType.FUEL:                    "FUEL",
    EmergencyType.MEDICAL:                 "MEDICAL",
    EmergencyType.MECHANICAL:              "MECHANICAL",
    EmergencyType.TECHNICAL:               "TECHNICAL",
    EmergencyType.WEATHER:                 "WEATHER",
    EmergencyType.SECURITY:                "SECURITY",
    EmergencyType.OPERATIONAL_CONSTRAINTS: "OPERATIONAL",
}

FUEL_CFG = {
    FuelState.NORMAL:   ("NORMAL",   Colour.GREEN,  Colour.GREEN_DIM),
    FuelState.LOW:      ("LOW",      Colour.AMBER,  Colour.AMBER_DIM),
    FuelState.CRITICAL: ("CRITICAL", Colour.RED,    Colour.RED_DIM),
}


class InputScreen(QWidget):
    run_requested  = Signal(str, float, float, object, object, list, list, list)
    logs_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._aircraft: str | None = None
        self._fuel: FuelState | None = None
        self._emergency: EmergencyType | None = None
        self._lat = 26.2708
        self._lon = 50.6336
        self._acft_btns: dict[str, QPushButton] = {}
        self._fuel_btns: dict[FuelState, QPushButton] = {}
        self._emrg_btns: dict[EmergencyType, QPushButton] = {}
        self._excluded_icaos: list[str] = []
        self._setup_ui()
        self._start_clock()

    # ── Clock ─────────────────────────────────────────────────────────────────

    def _start_clock(self):
        self._clock_timer = QTimer()
        self._clock_timer.timeout.connect(self._update_clock)
        self._clock_timer.start(1000)
        self._update_clock()

    def _update_clock(self):
        now = datetime.now(timezone.utc)
        self._clock_label.setText(now.strftime("%d %b %Y    %H:%M:%S UTC"))

    # ── Layout ────────────────────────────────────────────────────────────────
    # Total height budget: 600px
    #   Header:   52px
    #   Content: 460px  (scroll area)
    #   Footer:   88px
    #   Total:   600px  ✓

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_header())   # 52px fixed
        root.addWidget(self._build_body(), stretch=1)  # fills remaining
        root.addWidget(self._build_footer())   # 88px fixed

    def _build_header(self):
        h = QWidget()
        h.setFixedHeight(Spacing.HEADER_H)
        h.setStyleSheet(f"""
            background-color: {Colour.BG_HEADER};
            border-bottom: 1px solid {Colour.BORDER};
        """)
        lay = QHBoxLayout(h)
        lay.setContentsMargins(Spacing.LG, 0, Spacing.LG, 0)

        title = QLabel("EDA  —  Emergency Diversion Assistant")
        title.setStyleSheet(f"""
            color: {Colour.CYAN};
            font-size: {Font.SZ_MD}px;
            font-weight: bold;
            background: transparent;
        """)
        lay.addWidget(title, stretch=1)

        self._clock_label = QLabel("")
        self._clock_label.setStyleSheet(f"""
            color: {Colour.TEXT_PRIMARY};
            font-size: {Font.SZ_BODY}px;
            font-family: "{Font.MONO}", "Courier New";
            background: transparent;
            min-width: 220px;
        """)
        self._clock_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        lay.addWidget(self._clock_label)
        lay.addSpacing(Spacing.LG)

        logs_btn = self._small_btn("Logs")
        logs_btn.clicked.connect(self.logs_requested.emit)
        lay.addWidget(logs_btn)
        return h

    def _build_body(self):
        """
        Scrollable body — all content sections.
        Vertical scrollbar only appears if content does not fit.
        """
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setStyleSheet("border: none;")

        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(Spacing.LG, Spacing.SM, Spacing.LG, Spacing.SM)
        lay.setSpacing(Spacing.SM)

        # ── ACFT TYPE ────────────────────────────────────────
        lay.addWidget(self._section_label("ACFT TYPE"))
        lay.addWidget(self._build_aircraft_row())
        lay.addWidget(self._hdivider())

        # ── EMERGENCY + FUEL (side by side) ──────────────────
        row1 = QHBoxLayout()
        row1.setSpacing(Spacing.LG)

        emrg_col = QVBoxLayout()
        emrg_col.setSpacing(Spacing.XS)
        emrg_col.addWidget(self._section_label("EMERGENCY TYPE"))
        emrg_col.addWidget(self._build_emergency_grid())
        row1.addLayout(emrg_col, stretch=2)

        vdiv = QFrame()
        vdiv.setFrameShape(QFrame.Shape.VLine)
        vdiv.setFixedWidth(1)
        vdiv.setStyleSheet(f"background-color: {Colour.BORDER};")
        row1.addWidget(vdiv)

        fuel_col = QVBoxLayout()
        fuel_col.setSpacing(Spacing.XS)
        fuel_col.addWidget(self._section_label("FUEL STATE"))
        fuel_col.addWidget(self._build_fuel_col())
        row1.addLayout(fuel_col, stretch=1)

        lay.addLayout(row1)
        lay.addWidget(self._hdivider())

        # ── POSITION ─────────────────────────────────────────
        lay.addWidget(self._section_label("POSITION"))
        lay.addWidget(self._build_position_row(), stretch=1)

        # ── OPERATIONAL CONSTRAINTS (hidden by default) ───────
        self._op_panel = self._build_op_panel()
        self._op_panel.setVisible(False)
        lay.addWidget(self._op_panel)

        # No addStretch — let content fill naturally
        scroll.setWidget(body)
        return scroll

    def _build_aircraft_row(self):
        container = QWidget()
        lay = QHBoxLayout(container)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(Spacing.XS)

        scroll = QScrollArea()
        scroll.setFixedHeight(60)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("border: none; background: transparent;")

        row = QWidget()
        row_lay = QHBoxLayout(row)
        row_lay.setContentsMargins(0, 0, 0, 0)
        row_lay.setSpacing(Spacing.XS)

        for ac in list_available_aircraft():
            btn = QPushButton(ac)
            btn.setFixedSize(100, 48)
            btn.setStyleSheet(self._unselected_style())
            btn.clicked.connect(lambda _, a=ac: self._sel_aircraft(a))
            self._acft_btns[ac] = btn
            row_lay.addWidget(btn)
        row_lay.addStretch()
        scroll.setWidget(row)
        self._acft_scroll = scroll

        left  = self._arrow_btn("◀")
        right = self._arrow_btn("▶")
        left.clicked.connect(lambda: self._scroll_aircraft(-120))
        right.clicked.connect(lambda: self._scroll_aircraft(120))

        lay.addWidget(left)
        lay.addWidget(scroll, stretch=1)
        lay.addWidget(right)
        return container

    def _scroll_aircraft(self, delta):
        sb = self._acft_scroll.horizontalScrollBar()
        sb.setValue(sb.value() + delta)

    def _build_emergency_grid(self):
        w = QWidget()
        grid = QGridLayout(w)
        grid.setSpacing(Spacing.XS)
        grid.setContentsMargins(0, 0, 0, 0)
        for i, (etype, label) in enumerate(EMERGENCY_CFG.items()):
            btn = QPushButton(label)
            btn.setFixedHeight(50)
            btn.setStyleSheet(self._unselected_style())
            btn.clicked.connect(lambda _, e=etype: self._sel_emergency(e))
            self._emrg_btns[etype] = btn
            grid.addWidget(btn, i // 4, i % 4)
        return w

    def _build_fuel_col(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(Spacing.XS)
        for state, (label, colour, dim) in FUEL_CFG.items():
            btn = QPushButton(label)
            btn.setFixedHeight(48)
            btn.setStyleSheet(self._fuel_unselected(colour, dim))
            btn.clicked.connect(lambda _, s=state: self._sel_fuel(s))
            self._fuel_btns[state] = btn
            lay.addWidget(btn)
        return w

    def _build_position_row(self):
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(Spacing.LG)

        self._map = MapWidget(self._lat, self._lon)
        self._map.setMinimumHeight(150)
        self._map.position_selected.connect(self._on_map_pos)
        lay.addWidget(self._map, stretch=1)

        coord = QVBoxLayout()
        coord.setSpacing(Spacing.XS)

        for attr, label, min_v, max_v in [
            ("_lat_field", "LATITUDE",  -90.0,  90.0),
            ("_lon_field", "LONGITUDE", -180.0, 180.0),
        ]:
            lbl = QLabel(label)
            lbl.setStyleSheet(f"""
                color: {Colour.TEXT_SECONDARY};
                font-size: {Font.SZ_SM}px;
                font-weight: bold;
                letter-spacing: 1px;
                background: transparent;
            """)
            coord.addWidget(lbl)

            val = str(self._lat if "LAT" in label else self._lon)
            field = QLineEdit(val)
            field.setReadOnly(True)
            field.setFixedHeight(48)
            field.setCursor(Qt.CursorShape.PointingHandCursor)
            field.setStyleSheet(f"""
                QLineEdit {{
                    background-color: {Colour.BG_INPUT};
                    color: {Colour.CYAN};
                    border: 1px solid {Colour.BORDER};
                    border-radius: {Spacing.RADIUS_SM}px;
                    padding: 6px 10px;
                    font-size: {Font.SZ_MD}px;
                    font-family: "{Font.MONO}", "Courier New";
                }}
                QLineEdit:hover {{ border-color: {Colour.CYAN_DIM}; }}
            """)
            setattr(self, attr, field)
            fld  = field
            mn, mx, la = min_v, max_v, label
            field.mousePressEvent = lambda e, f=fld, la=la, mn=mn, mx=mx: \
                self._open_numpad(la, f, mn, mx)
            coord.addWidget(field)

        coord.addStretch()
        lay.addLayout(coord)
        return w

    def _build_op_panel(self):
        panel = QWidget()
        panel.setStyleSheet(f"""
            QWidget {{
                background-color: {Colour.BG_CARD};
                border-top: 2px solid {Colour.CYAN_DIM};
            }}
        """)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(Spacing.MD, Spacing.SM, Spacing.MD, Spacing.SM)
        lay.setSpacing(Spacing.XS)

        title = QLabel("EXCLUDED AIRPORTS")
        title.setStyleSheet(f"""
            color: {Colour.CYAN};
            font-size: {Font.SZ_BODY}px;
            font-weight: bold;
            background: transparent;
        """)
        lay.addWidget(title)

        hint = QLabel(
            "Add airport ICAO codes to exclude. "
            "Only airports in the database are accepted."
        )
        hint.setStyleSheet(f"""
            color: {Colour.TEXT_SECONDARY};
            font-size: {Font.SZ_SM}px;
            background: transparent;
        """)
        hint.setWordWrap(True)
        lay.addWidget(hint)

        self._icao_list = QListWidget()
        self._icao_list.setFixedHeight(64)
        self._icao_list.setStyleSheet(f"""
            QListWidget {{
                background-color: {Colour.BG_INPUT};
                border: 1px solid {Colour.BORDER};
                border-radius: {Spacing.RADIUS_SM}px;
                color: {Colour.CYAN};
                font-size: {Font.SZ_BODY}px;
                font-family: "{Font.MONO}", "Courier New";
            }}
            QListWidget::item {{ padding: 4px 12px; border: none; }}
        """)
        lay.addWidget(self._icao_list)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(Spacing.SM)

        add_btn = QPushButton("Add Airport")
        add_btn.setFixedHeight(44)
        add_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colour.CYAN_BG};
                color: {Colour.CYAN};
                border: 1px solid {Colour.CYAN_DIM};
                border-radius: {Spacing.RADIUS_SM}px;
                font-size: {Font.SZ_BODY}px;
                font-weight: bold;
            }}
            QPushButton:pressed {{ background-color: {Colour.CYAN_DIM}; }}
        """)
        add_btn.clicked.connect(self._open_icao_keyboard)

        clr_btn = QPushButton("Clear All")
        clr_btn.setFixedHeight(44)
        clr_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {Colour.AMBER};
                border: 1px solid {Colour.AMBER_DIM};
                border-radius: {Spacing.RADIUS_SM}px;
                font-size: {Font.SZ_BODY}px;
            }}
            QPushButton:pressed {{ background-color: {Colour.AMBER_BG}; }}
        """)
        clr_btn.clicked.connect(self._clear_icaos)

        btn_row.addWidget(add_btn, stretch=2)
        btn_row.addWidget(clr_btn, stretch=1)
        lay.addLayout(btn_row)

        self._op_warning = QLabel("")
        self._op_warning.setStyleSheet(f"""
            color: {Colour.AMBER};
            font-size: {Font.SZ_SM}px;
            background: transparent;
        """)
        lay.addWidget(self._op_warning)
        return panel

    def _build_footer(self):
        """
        Footer always visible — fixed 88px at bottom.
        Never inside the scroll area.
        """
        bar = QWidget()
        bar.setFixedHeight(88)
        bar.setStyleSheet(f"""
            background-color: {Colour.BG_HEADER};
            border-top: 1px solid {Colour.BORDER};
        """)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(Spacing.LG, Spacing.MD, Spacing.LG, Spacing.MD)
        lay.setSpacing(Spacing.MD)

        reset_btn = QPushButton("Reset")
        reset_btn.setFixedSize(90, 60)
        reset_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {Colour.AMBER};
                border: 1px solid {Colour.AMBER_DIM};
                border-radius: {Spacing.RADIUS_SM}px;
                font-size: {Font.SZ_BODY}px;
                font-weight: bold;
            }}
            QPushButton:pressed {{ background-color: {Colour.AMBER_BG}; }}
        """)
        reset_btn.clicked.connect(self._reset)
        lay.addWidget(reset_btn)

        self._run_btn = QPushButton("Run Diversion Analysis")
        self._run_btn.setFixedHeight(60)
        self._run_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colour.GREEN_BG};
                color: {Colour.GREEN};
                border: 1px solid {Colour.GREEN_DIM};
                border-radius: {Spacing.RADIUS_SM}px;
                font-size: {Font.SZ_LG}px;
                font-weight: bold;
            }}
            QPushButton:pressed {{
                background-color: {Colour.GREEN_DIM};
                color: {Colour.TEXT_PRIMARY};
            }}
            QPushButton:disabled {{
                color: {Colour.TEXT_MUTED};
                border-color: {Colour.TEXT_MUTED};
                background-color: transparent;
            }}
        """)
        self._run_btn.clicked.connect(self._on_run)
        lay.addWidget(self._run_btn, stretch=1)
        return bar

    # ── Overlays ──────────────────────────────────────────────────────────────

    def _open_numpad(self, label, field, min_v, max_v):
        np = NumpadWidget(label, field.text(), min_v, max_v, self)
        np.setGeometry(0, 0, self.width(), self.height())
        np.setStyleSheet(
            f"NumpadWidget {{ background-color: rgba(14, 26, 36, 230); }}"
        )

        def accept(val):
            field.setText(val)
            if "LAT" in label:
                self._lat = float(val)
            else:
                self._lon = float(val)
            self._map.set_position(self._lat, self._lon)
            np.deleteLater()

        np.value_accepted.connect(accept)
        np.cancelled.connect(np.deleteLater)
        np.show()
        np.raise_()

    def _open_icao_keyboard(self):
        kb = ICAOKeyboard(self)
        kb.setGeometry(0, 0, self.width(), self.height())
        kb.setStyleSheet(
            f"ICAOKeyboard {{ background-color: rgba(14, 26, 36, 230); }}"
        )

        def on_code(code):
            if code not in self._excluded_icaos:
                self._excluded_icaos.append(code)
                self._icao_list.addItem(code)
            self._op_warning.setText("")
            kb.reset()

        kb.code_accepted.connect(on_code)
        kb.cancelled.connect(kb.deleteLater)
        kb.show()
        kb.raise_()

    def _clear_icaos(self):
        self._excluded_icaos.clear()
        self._icao_list.clear()

    # ── Selection ─────────────────────────────────────────────────────────────

    def _sel_aircraft(self, ac):
        self._aircraft = ac
        for name, btn in self._acft_btns.items():
            btn.setStyleSheet(
                self._selected_style() if name == ac
                else self._unselected_style()
            )

    def _sel_fuel(self, state):
        if not self._fuel_btns[state].isEnabled():
            return
        self._fuel = state
        for s, btn in self._fuel_btns.items():
            _, c, d = FUEL_CFG[s]
            btn.setStyleSheet(
                self._selected_style() if s == state
                else self._fuel_unselected(c, d)
            )

    def _sel_emergency(self, etype):
        self._emergency = etype
        for e, btn in self._emrg_btns.items():
            btn.setStyleSheet(
                self._selected_style() if e == etype
                else self._unselected_style()
            )

        is_fuel = etype == EmergencyType.FUEL
        nb = self._fuel_btns[FuelState.NORMAL]
        nb.setEnabled(not is_fuel)
        if is_fuel and self._fuel == FuelState.NORMAL:
            self._fuel = None
            for s, btn in self._fuel_btns.items():
                _, c, d = FUEL_CFG[s]
                btn.setStyleSheet(self._fuel_unselected(c, d))

        self._op_panel.setVisible(
            etype == EmergencyType.OPERATIONAL_CONSTRAINTS
        )

    def _on_map_pos(self, lat, lon):
        self._lat, self._lon = lat, lon
        self._lat_field.setText(str(lat))
        self._lon_field.setText(str(lon))

    def _reset(self):
        self._aircraft = None
        self._fuel     = None
        self._emergency = None
        self._lat, self._lon = 26.2708, 50.6336

        for btn in self._acft_btns.values():
            btn.setStyleSheet(self._unselected_style())
        for state, btn in self._fuel_btns.items():
            _, c, d = FUEL_CFG[state]
            btn.setEnabled(True)
            btn.setStyleSheet(self._fuel_unselected(c, d))
        for btn in self._emrg_btns.values():
            btn.setStyleSheet(self._unselected_style())

        self._lat_field.setText(str(self._lat))
        self._lon_field.setText(str(self._lon))
        self._map.set_position(self._lat, self._lon)
        self._op_panel.setVisible(False)
        self._clear_icaos()

    def _on_run(self):
        if not self._aircraft:
            return self._flash("Select an aircraft type")
        if not self._fuel:
            return self._flash("Select a fuel state")
        if not self._emergency:
            return self._flash("Select an emergency type")
        if (self._emergency == EmergencyType.OPERATIONAL_CONSTRAINTS
                and not self._excluded_icaos):
            self._op_warning.setText(
                "Add at least one airport ICAO code to exclude"
            )
            return

        self.run_requested.emit(
            self._aircraft, self._lat, self._lon,
            self._fuel, self._emergency,
            self._excluded_icaos, [], [],
        )

    def _flash(self, msg):
        orig = self._run_btn.text()
        self._run_btn.setText(f"  {msg}")
        self._run_btn.setEnabled(False)
        QTimer.singleShot(2500, lambda: (
            self._run_btn.setText(orig),
            self._run_btn.setEnabled(True),
        ))

    # ── Styles ────────────────────────────────────────────────────────────────

    def _selected_style(self) -> str:
        return f"""
            QPushButton {{
                background-color: {Colour.CYAN_BG};
                color: {Colour.CYAN};
                border: 1px solid {Colour.CYAN};
                border-radius: {Spacing.RADIUS_SM}px;
                font-size: {Font.SZ_SM}px;
                font-weight: bold;
            }}
        """

    def _unselected_style(self) -> str:
        return f"""
            QPushButton {{
                background-color: {Colour.BTN_BG};
                color: {Colour.TEXT_SECONDARY};
                border: 1px solid {Colour.BORDER};
                border-radius: {Spacing.RADIUS_SM}px;
                font-size: {Font.SZ_SM}px;
            }}
            QPushButton:pressed {{
                background-color: {Colour.BTN_PRESSED};
                color: {Colour.TEXT_PRIMARY};
            }}
            QPushButton:disabled {{
                color: {Colour.TEXT_MUTED};
                border-color: {Colour.TEXT_MUTED};
                background-color: {Colour.BG_BASE};
            }}
        """

    def _fuel_unselected(self, colour, dim) -> str:
        return f"""
            QPushButton {{
                background-color: {Colour.BTN_BG};
                color: {colour};
                border: 1px solid {dim};
                border-radius: {Spacing.RADIUS_SM}px;
                font-size: {Font.SZ_SM}px;
                font-weight: bold;
            }}
            QPushButton:pressed {{
                background-color: {Colour.BTN_PRESSED};
                color: {Colour.TEXT_PRIMARY};
            }}
            QPushButton:disabled {{
                color: {Colour.TEXT_MUTED};
                border-color: {Colour.TEXT_MUTED};
                background-color: {Colour.BG_BASE};
            }}
        """

    def _section_label(self, text) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(f"""
            color: {Colour.TEXT_SECONDARY};
            font-size: {Font.SZ_SM}px;
            font-weight: bold;
            letter-spacing: 1px;
            background: transparent;
        """)
        return lbl

    def _hdivider(self) -> QFrame:
        f = QFrame()
        f.setFrameShape(QFrame.Shape.HLine)
        f.setFixedHeight(1)
        f.setStyleSheet(f"background-color: {Colour.BORDER};")
        return f

    def _small_btn(self, text) -> QPushButton:
        btn = QPushButton(text)
        btn.setFixedHeight(34)
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {Colour.TEXT_SECONDARY};
                border: 1px solid {Colour.BORDER};
                border-radius: {Spacing.RADIUS_SM}px;
                font-size: {Font.SZ_SM}px;
                padding: 0 14px;
            }}
            QPushButton:pressed {{
                background-color: {Colour.BTN_PRESSED};
                color: {Colour.TEXT_PRIMARY};
            }}
        """)
        return btn

    def _arrow_btn(self, text) -> QPushButton:
        btn = QPushButton(text)
        btn.setFixedSize(36, 48)
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colour.BTN_BG};
                color: {Colour.CYAN};
                border: 1px solid {Colour.BORDER};
                border-radius: {Spacing.RADIUS_SM}px;
                font-size: {Font.SZ_LG}px;
            }}
            QPushButton:pressed {{ background-color: {Colour.CYAN_BG}; }}
        """)
        return btn