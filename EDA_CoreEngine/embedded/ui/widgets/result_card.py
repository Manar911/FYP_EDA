"""
result_card.py

Widget displaying one ranked airport recommendation with
its XAI explanation text and a confirm button.

Displays:
- Rank number + ICAO code + airport name + distance
- ML confidence score
- Explanation bullet points (ranking_reasons from explanation.py)
- Summary sentence
- Confirm button for pilot decision
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame,
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QFont

from eda.ranking import RankedOption
from eda.explanation import Explanation
from ui.theme import Colour, Font, Spacing


class ResultCard(QWidget):
    """
    Displays one ranked airport with explanation and confirm button.
    """

    # Emitted when pilot taps Confirm
    confirmed = Signal(str)  # airport ICAO code

    def __init__(
        self,
        rank: int,
        ranked_option: RankedOption,
        explanation: Explanation,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._rank = rank
        self._ranked_option = ranked_option
        self._explanation = explanation
        self._setup_ui()

    def _rank_colour(self) -> str:
        if self._rank == 1:
            return Colour.RANK_1
        if self._rank == 2:
            return Colour.RANK_2
        return Colour.RANK_3

    def _card_bg(self) -> str:
        if self._rank == 1:
            return Colour.RANK_1_BG
        return Colour.RANK_2_BG

    def _border_colour(self) -> str:
        if self._rank == 1:
            return Colour.RANK_1
        return Colour.BORDER_DEFAULT

    def _setup_ui(self) -> None:
        airport  = self._ranked_option.airport
        features = self._ranked_option.features
        score    = self._ranked_option.score
        colour   = self._rank_colour()

        # ── Outer frame ──────────────────────────────────────────
        self.setStyleSheet(f"""
            ResultCard {{
                background-color: {self._card_bg()};
                border: 1px solid {self._border_colour()};
                border-radius: {Spacing.CARD_RADIUS}px;
            }}
        """)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(
            Spacing.LG, Spacing.MD,
            Spacing.LG, Spacing.MD
        )
        outer.setSpacing(Spacing.SM)

        # ── Row 1: rank badge + ICAO + name + distance + score ───
        top_row = QHBoxLayout()
        top_row.setSpacing(Spacing.MD)

        # Rank badge
        rank_label = QLabel(f"#{self._rank}")
        rank_label.setStyleSheet(f"""
            QLabel {{
                color: {colour};
                font-size: {Font.SIZE_LARGE}px;
                font-weight: bold;
                min-width: 32px;
            }}
        """)
        rank_label.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        top_row.addWidget(rank_label)

        # ICAO code — large and bold
        icao_label = QLabel(airport.icao)
        icao_label.setStyleSheet(f"""
            QLabel {{
                color: {colour};
                font-size: {Font.SIZE_XLARGE}px;
                font-weight: bold;
                min-width: 60px;
            }}
        """)
        icao_label.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        top_row.addWidget(icao_label)

        # Airport name + city
        name_layout = QVBoxLayout()
        name_layout.setSpacing(0)

        name_label = QLabel(airport.name)
        name_label.setStyleSheet(f"""
            QLabel {{
                color: {Colour.TEXT_PRIMARY};
                font-size: {Font.SIZE_MEDIUM}px;
                font-weight: bold;
            }}
        """)
        name_label.setWordWrap(True)
        name_layout.addWidget(name_label)

        city_label = QLabel(f"{airport.city}, {airport.country}")
        city_label.setStyleSheet(f"""
            QLabel {{
                color: {Colour.TEXT_SECONDARY};
                font-size: {Font.SIZE_SMALL}px;
            }}
        """)
        name_layout.addWidget(city_label)
        top_row.addLayout(name_layout, stretch=1)

        # Distance
        dist_label = QLabel(f"{features.distance_km:.1f} km")
        dist_label.setStyleSheet(f"""
            QLabel {{
                color: {Colour.TEXT_SECONDARY};
                font-size: {Font.SIZE_BODY}px;
                min-width: 70px;
            }}
        """)
        dist_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        top_row.addWidget(dist_label)

        # Score badge
        score_pct = int(score * 100)
        score_label = QLabel(f"{score_pct}%")
        score_label.setStyleSheet(f"""
            QLabel {{
                background-color: {colour};
                color: {Colour.BG_PRIMARY};
                font-size: {Font.SIZE_SMALL}px;
                font-weight: bold;
                border-radius: 4px;
                padding: 2px 8px;
                min-width: 42px;
            }}
        """)
        score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top_row.addWidget(score_label)

        outer.addLayout(top_row)

        # ── Divider ───────────────────────────────────────────────
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet(f"background-color: {Colour.DIVIDER}; max-height: 1px;")
        outer.addWidget(divider)

        # ── Explanation bullets ───────────────────────────────────
        reasons = self._explanation.ranking_reasons
        if reasons:
            for reason in reasons:
                reason_label = QLabel(f"  ·  {reason}")
                reason_label.setStyleSheet(f"""
                    QLabel {{
                        color: {Colour.TEXT_PRIMARY};
                        font-size: {Font.SIZE_BODY}px;
                    }}
                """)
                reason_label.setWordWrap(True)
                outer.addWidget(reason_label)

        # ── Summary sentence ──────────────────────────────────────
        if self._explanation.summary:
            summary_label = QLabel(f'"{self._explanation.summary}"')
            summary_label.setStyleSheet(f"""
                QLabel {{
                    color: {Colour.TEXT_SECONDARY};
                    font-size: {Font.SIZE_SMALL}px;
                    font-style: italic;
                    padding-top: 2px;
                }}
            """)
            summary_label.setWordWrap(True)
            outer.addWidget(summary_label)

        # ── Caution note ──────────────────────────────────────────
        caution_label = QLabel(self._explanation.caution)
        caution_label.setStyleSheet(f"""
            QLabel {{
                color: {Colour.TEXT_CAUTION};
                font-size: {Font.SIZE_TINY}px;
            }}
        """)
        outer.addWidget(caution_label)

        # ── Confirm button ────────────────────────────────────────
        confirm_btn = QPushButton(f"  ✓  Confirm  {airport.icao}  as diversion airport")
        confirm_btn.setMinimumHeight(Spacing.TOUCH_MIN)
        confirm_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colour.CONFIRM};
                color: {Colour.TEXT_PRIMARY};
                border: none;
                border-radius: {Spacing.BUTTON_RADIUS}px;
                font-size: {Font.SIZE_BODY}px;
                font-weight: bold;
                padding: 10px 20px;
            }}
            QPushButton:pressed {{
                background-color: {Colour.CONFIRM_HOVER};
            }}
        """)
        confirm_btn.clicked.connect(
            lambda: self.confirmed.emit(airport.icao)
        )
        outer.addWidget(confirm_btn)