"""
results_screen.py

Results screen showing the top 3 ranked diversion airports
with XAI explanation text and confirm buttons.

Also handles the confirm dialog for pilot decision logging.
"""

from __future__ import annotations

from typing import List

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea,
    QLabel, QPushButton, QDialog, QFrame,
)
from PySide6.QtCore import Signal, Qt

from eda.models import DecisionReport
from eda.explanation import Explanation
from eda.scenario import EmergencyType, FuelState
from ui.theme import Colour, Font, Spacing
from ui.widgets.result_card import ResultCard


class ConfirmDialog(QDialog):
    """
    Confirmation dialog shown when pilot taps Confirm on a result card.
    """

    confirmed = Signal(str)  # airport ICAO

    def __init__(
        self,
        icao: str,
        airport_name: str,
        distance_km: float,
        score: float,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Confirm Diversion")
        self.setModal(True)
        self.setFixedSize(480, 260)
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {Colour.BG_SECONDARY};
                border: 1px solid {Colour.BORDER_DEFAULT};
                border-radius: {Spacing.CARD_RADIUS}px;
            }}
        """)
        self._icao = icao
        self._setup_ui(icao, airport_name, distance_km, score)

    def _setup_ui(
        self,
        icao: str,
        name: str,
        distance: float,
        score: float,
    ) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(Spacing.XL, Spacing.XL, Spacing.XL, Spacing.XL)
        layout.setSpacing(Spacing.LG)

        # Title
        title = QLabel("Confirm Diversion Selection")
        title.setStyleSheet(f"""
            color: {Colour.TEXT_PRIMARY};
            font-size: {Font.SIZE_LARGE}px;
            font-weight: bold;
        """)
        layout.addWidget(title)

        # Airport info
        info = QLabel(
            f"{icao}  —  {name}\n"
            f"Distance: {distance:.1f} km    "
            f"Confidence: {int(score * 100)}%"
        )
        info.setStyleSheet(f"""
            color: {Colour.TEXT_SECONDARY};
            font-size: {Font.SIZE_BODY}px;
            line-height: 1.6;
        """)
        layout.addWidget(info)

        # Caution
        caution = QLabel("Final diversion decision remains with the pilot.")
        caution.setStyleSheet(f"""
            color: {Colour.TEXT_CAUTION};
            font-size: {Font.SIZE_SMALL}px;
            font-style: italic;
        """)
        layout.addWidget(caution)

        layout.addStretch()

        # Buttons row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(Spacing.MD)

        cancel_btn = QPushButton("CANCEL")
        cancel_btn.setFixedHeight(Spacing.TOUCH_MIN)
        cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colour.BACK};
                color: {Colour.TEXT_SECONDARY};
                border: 1px solid {Colour.BORDER_DEFAULT};
                border-radius: {Spacing.BUTTON_RADIUS}px;
                font-size: {Font.SIZE_BODY}px;
                font-weight: bold;
            }}
            QPushButton:pressed {{
                background-color: {Colour.BACK_HOVER};
            }}
        """)
        cancel_btn.clicked.connect(self.reject)

        confirm_btn = QPushButton(f"✓  CONFIRM  {icao}")
        confirm_btn.setFixedHeight(Spacing.TOUCH_MIN)
        confirm_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colour.CONFIRM};
                color: {Colour.TEXT_PRIMARY};
                border: none;
                border-radius: {Spacing.BUTTON_RADIUS}px;
                font-size: {Font.SIZE_BODY}px;
                font-weight: bold;
            }}
            QPushButton:pressed {{
                background-color: {Colour.CONFIRM_HOVER};
            }}
        """)
        confirm_btn.clicked.connect(self._on_confirm)

        btn_row.addWidget(cancel_btn, stretch=1)
        btn_row.addWidget(confirm_btn, stretch=2)
        layout.addLayout(btn_row)

    def _on_confirm(self) -> None:
        self.confirmed.emit(self._icao)
        self.accept()


class ResultsScreen(QWidget):
    """
    Displays the top 3 ranked airports after inference completes.
    """

    back_requested    = Signal()
    decision_confirmed = Signal(str, object)
    # args: confirmed ICAO, DecisionReport

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._report: DecisionReport | None = None
        self._explanations: List[Explanation] = []
        self._setup_ui()

    def _setup_ui(self) -> None:
        self._root_layout = QVBoxLayout(self)
        self._root_layout.setContentsMargins(0, 0, 0, 0)
        self._root_layout.setSpacing(0)

        self._header = self._build_header("", "", "")
        self._root_layout.addWidget(self._header)

        # Scroll area for result cards
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._scroll.setStyleSheet("border: none;")

        self._cards_widget = QWidget()
        self._cards_layout = QVBoxLayout(self._cards_widget)
        self._cards_layout.setContentsMargins(
            Spacing.LG, Spacing.LG, Spacing.LG, Spacing.LG
        )
        self._cards_layout.setSpacing(Spacing.MD)
        self._cards_layout.addStretch()

        self._scroll.setWidget(self._cards_widget)
        self._root_layout.addWidget(self._scroll, stretch=1)

    def _build_header(
        self,
        emergency: str,
        aircraft: str,
        fuel: str,
    ) -> QWidget:
        header = QWidget()
        header.setFixedHeight(Spacing.HEADER_HEIGHT)
        header.setStyleSheet(f"""
            background-color: {Colour.BG_HEADER};
            border-bottom: 1px solid {Colour.BORDER_DEFAULT};
        """)

        layout = QHBoxLayout(header)
        layout.setContentsMargins(Spacing.LG, 0, Spacing.LG, 0)
        layout.setSpacing(Spacing.MD)

        back_btn = QPushButton("◀  BACK")
        back_btn.setFixedHeight(36)
        back_btn.setStyleSheet(f"""
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
        back_btn.clicked.connect(self.back_requested.emit)
        layout.addWidget(back_btn)

        context = QLabel(
            f"{emergency.upper()}  ·  {aircraft}  ·  {fuel.upper()} FUEL"
            if emergency else "RESULTS"
        )
        context.setStyleSheet(f"""
            color: {Colour.TEXT_SECONDARY};
            font-size: {Font.SIZE_SMALL}px;
            font-weight: bold;
            letter-spacing: 1px;
        """)
        layout.addWidget(context, stretch=1)

        feasible_label = QLabel("")
        feasible_label.setObjectName("feasible_label")
        feasible_label.setStyleSheet(f"""
            color: {Colour.TEXT_MUTED};
            font-size: {Font.SIZE_SMALL}px;
        """)
        layout.addWidget(feasible_label)

        return header

    def load_results(
        self,
        report: DecisionReport,
        explanations: List[Explanation],
    ) -> None:
        """
        Called after inference completes. Populates the screen with results.
        """
        self._report = report
        self._explanations = explanations

        # Rebuild header with context
        self._root_layout.removeWidget(self._header)
        self._header.deleteLater()
        self._header = self._build_header(
            report.scenario.emergency_type.value,
            report.scenario.aircraft_type,
            report.scenario.fuel_state.value,
        )
        # Update feasible count
        fl = self._header.findChild(QLabel, "feasible_label")
        if fl:
            fl.setText(f"{len(report.feasible)} airports evaluated")
        self._root_layout.insertWidget(0, self._header)

        # Clear existing cards
        while self._cards_layout.count() > 1:
            item = self._cards_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Add result cards
        if not report.ranked_top:
            no_result = QLabel("No feasible diversion airports found.")
            no_result.setStyleSheet(f"""
                color: {Colour.TEXT_SECONDARY};
                font-size: {Font.SIZE_MEDIUM}px;
                padding: 40px;
            """)
            no_result.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._cards_layout.insertWidget(0, no_result)
            return

        for i, (ranked, explanation) in enumerate(
            zip(report.ranked_top, explanations), start=1
        ):
            card = ResultCard(i, ranked, explanation, self)
            card.confirmed.connect(self._on_confirm_requested)
            self._cards_layout.insertWidget(i - 1, card)

    def _on_confirm_requested(self, icao: str) -> None:
        if not self._report:
            return

        # Find the ranked option for this ICAO
        ranked = next(
            (r for r in self._report.ranked_top if r.airport.icao == icao),
            None,
        )
        if not ranked:
            return

        dialog = ConfirmDialog(
            icao=icao,
            airport_name=ranked.airport.name,
            distance_km=ranked.features.distance_km,
            score=ranked.score,
            parent=self,
        )
        dialog.confirmed.connect(
            lambda confirmed_icao: self.decision_confirmed.emit(
                confirmed_icao, self._report
            )
        )
        dialog.exec()