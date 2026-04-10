"""
results_screen.py  —  EDA 

"""

from __future__ import annotations
from typing import List
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea,
    QLabel, QPushButton, QDialog, QFrame,
)
from PySide6.QtCore import Signal, Qt
from core.models import DecisionReport
from core.explanation import Explanation
from ui.theme import Colour, Font, Spacing
from ui.widgets.result_card import ResultCard


class ConfirmDialog(QDialog):
    confirmed = Signal(str)

    def __init__(self, icao, name, distance_km, score, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Confirm Diversion")
        self.setModal(True)
        # Taller dialog to accommodate long airport names
        self.setFixedWidth(520)
        self.setMinimumHeight(300)
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {Colour.BG_CARD};
                border: 1px solid {Colour.BORDER_ACTIVE};
            }}
        """)
        self._icao = icao
        distance_nm = distance_km / 1.852

        layout = QVBoxLayout(self)
        layout.setContentsMargins(Spacing.XL, Spacing.XL, Spacing.XL, Spacing.XL)
        layout.setSpacing(Spacing.LG)

        # Header
        header = QLabel("CONFIRM DIVERSION SELECTION")
        header.setStyleSheet(f"""
            color: {Colour.CYAN};
            font-size: {Font.SZ_SM}px;
            font-weight: bold;
            letter-spacing: 2px;
            background: transparent;
        """)
        layout.addWidget(header)

        # Divider
        div = QFrame()
        div.setFrameShape(QFrame.Shape.HLine)
        div.setFixedHeight(1)
        div.setStyleSheet(f"background-color: {Colour.BORDER}; border: none;")
        layout.addWidget(div)

        # ICAO — large
        icao_lbl = QLabel(icao)
        icao_lbl.setStyleSheet(f"""
            color: {Colour.CYAN};
            font-size: {Font.SZ_XL}px;
            font-family: "{Font.MONO}", "Courier New";
            font-weight: bold;
            background: transparent;
        """)
        layout.addWidget(icao_lbl)

        # Airport name — word wrap so long names never get cut
        name_lbl = QLabel(name)
        name_lbl.setStyleSheet(f"""
            color: {Colour.TEXT_PRIMARY};
            font-size: {Font.SZ_MD}px;
            font-weight: bold;
            background: transparent;
        """)
        name_lbl.setWordWrap(True)
        layout.addWidget(name_lbl)

        # Distance and confidence
        details_lbl = QLabel(
            f"Distance:  {distance_nm:.1f} NM     "
            f"Confidence:  {int(score * 100)}%"
        )
        details_lbl.setStyleSheet(f"""
            color: {Colour.TEXT_SECONDARY};
            font-size: {Font.SZ_BODY}px;
            font-family: "{Font.MONO}", "Courier New";
            background: transparent;
        """)
        layout.addWidget(details_lbl)

        # Caution
        caution = QLabel("ADVISORY ONLY  —  FINAL DECISION: PILOT IN COMMAND")
        caution.setStyleSheet(f"""
            color: {Colour.AMBER};
            font-size: {Font.SZ_XS}px;
            font-weight: bold;
            letter-spacing: 1px;
            background: transparent;
        """)
        layout.addWidget(caution)

        layout.addStretch()

        # Button row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(Spacing.MD)

        # Cancel — RED
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFixedHeight(Spacing.TOUCH_MIN)
        cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colour.RED_BG};
                color: {Colour.RED};
                border: 1px solid {Colour.RED_DIM};
                border-radius: {Spacing.RADIUS_SM}px;
                font-size: {Font.SZ_BODY}px;
                font-weight: bold;
            }}
            QPushButton:pressed {{
                background-color: {Colour.RED_DIM};
                color: {Colour.TEXT_PRIMARY};
            }}
        """)
        cancel_btn.clicked.connect(self.reject)

        # Confirm — GREEN
        confirm_btn = QPushButton(f"Confirm  {icao}")
        confirm_btn.setFixedHeight(Spacing.TOUCH_MIN)
        confirm_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colour.GREEN_BG};
                color: {Colour.GREEN};
                border: 1px solid {Colour.GREEN_DIM};
                border-radius: {Spacing.RADIUS_SM}px;
                font-size: {Font.SZ_BODY}px;
                font-weight: bold;
            }}
            QPushButton:pressed {{
                background-color: {Colour.GREEN_DIM};
                color: {Colour.TEXT_PRIMARY};
            }}
        """)
        confirm_btn.clicked.connect(
            lambda: (self.confirmed.emit(self._icao), self.accept())
        )

        btn_row.addWidget(cancel_btn, stretch=1)
        btn_row.addWidget(confirm_btn, stretch=2)
        layout.addLayout(btn_row)


class ResultsScreen(QWidget):
    back_requested     = Signal()
    decision_confirmed = Signal(str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._report = None
        self._explanations = []
        self._setup_ui()

    def _setup_ui(self):
        self._root = QVBoxLayout(self)
        self._root.setContentsMargins(0, 0, 0, 0)
        self._root.setSpacing(0)

        self._header = self._build_header("", "", "")
        self._root.addWidget(self._header)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._scroll.setStyleSheet("border: none;")

        self._cards_w = QWidget()
        self._cards_l = QVBoxLayout(self._cards_w)
        self._cards_l.setContentsMargins(
            Spacing.LG, Spacing.LG, Spacing.LG, Spacing.LG
        )
        self._cards_l.setSpacing(Spacing.MD)
        self._cards_l.addStretch()

        self._scroll.setWidget(self._cards_w)
        self._root.addWidget(self._scroll, stretch=1)

    def _build_header(self, emergency, aircraft, fuel):
        h = QWidget()
        h.setFixedHeight(Spacing.HEADER_H)
        h.setStyleSheet(f"""
            background-color: {Colour.BG_HEADER};
            border-bottom: 1px solid {Colour.BORDER};
        """)
        lay = QHBoxLayout(h)
        lay.setContentsMargins(Spacing.LG, 0, Spacing.LG, 0)
        lay.setSpacing(Spacing.MD)

        back = QPushButton("Back")
        back.setFixedHeight(36)
        back.setStyleSheet(f"""
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
        back.clicked.connect(self.back_requested.emit)
        lay.addWidget(back)

        ctx = QLabel(
            f"{emergency.upper()}  ·  {aircraft}  ·  Fuel: {fuel.upper()}"
            if emergency else "Diversion Analysis"
        )
        ctx.setStyleSheet(f"""
            color: {Colour.CYAN};
            font-size: {Font.SZ_BODY}px;
            font-weight: bold;
            background: transparent;
        """)
        lay.addWidget(ctx, stretch=1)

        self._feasible_lbl = QLabel("")
        self._feasible_lbl.setStyleSheet(f"""
            color: {Colour.TEXT_SECONDARY};
            font-size: {Font.SZ_SM}px;
            background: transparent;
        """)
        lay.addWidget(self._feasible_lbl)
        return h

    def load_results(self, report, explanations):
        self._report = report
        self._explanations = explanations

        self._root.removeWidget(self._header)
        self._header.deleteLater()
        self._header = self._build_header(
            report.scenario.emergency_type.value,
            report.scenario.aircraft_type,
            report.scenario.fuel_state.value,
        )
        self._feasible_lbl.setText(
            f"{len(report.feasible)} airports evaluated"
        )
        self._root.insertWidget(0, self._header)

        while self._cards_l.count() > 1:
            item = self._cards_l.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not report.ranked_top:
            lbl = QLabel("No feasible diversion airports found for this scenario.")
            lbl.setStyleSheet(f"""
                color: {Colour.TEXT_SECONDARY};
                font-size: {Font.SZ_MD}px;
                padding: 40px;
                background: transparent;
            """)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setWordWrap(True)
            self._cards_l.insertWidget(0, lbl)
            return

        for i, (ranked, expl) in enumerate(
            zip(report.ranked_top, explanations), 1
        ):
            card = ResultCard(i, ranked, expl, self)
            card.confirmed.connect(self._on_confirm)
            self._cards_l.insertWidget(i - 1, card)

    def _on_confirm(self, icao):
        if not self._report:
            return
        ranked = next(
            (r for r in self._report.ranked_top if r.airport.icao == icao),
            None,
        )
        if not ranked:
            return
        dlg = ConfirmDialog(
            icao=icao,
            name=ranked.airport.name,
            distance_km=ranked.features.distance_km,
            score=ranked.score,
            parent=self,
        )
        dlg.confirmed.connect(
            lambda c: self.decision_confirmed.emit(c, self._report)
        )
        dlg.exec()