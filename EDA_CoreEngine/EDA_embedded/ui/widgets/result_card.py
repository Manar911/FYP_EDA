"""
result_card.py  —  EDA 

One ranked airport recommendation card.
Distance displayed in NM (correctly converted from km ÷ 1.852).
ACFT TYPE abbreviation style maintained throughout.
Colours match map palette via theme.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame,
)
from PySide6.QtCore import Signal, Qt

from core.ranking import RankedOption
from core.explanation import Explanation
from ui.theme import Colour, Font, Spacing

KM_TO_NM = 1.852   # 1 NM = 1.852 km


class ResultCard(QWidget):
    confirmed = Signal(str)

    def __init__(self, rank: int, option: RankedOption, expl: Explanation, parent=None):
        super().__init__(parent)
        self._rank   = rank
        self._option = option
        self._expl   = expl
        self._build()

    def _colours(self):
        if self._rank == 1:
            return Colour.RANK_1_COLOUR, Colour.RANK_1_BG, Colour.CYAN_DIM
        if self._rank == 2:
            return Colour.RANK_2_COLOUR, Colour.RANK_2_BG, Colour.BORDER
        return Colour.RANK_3_COLOUR, Colour.RANK_3_BG, Colour.BORDER

    def _build(self):
        airport  = self._option.airport
        features = self._option.features
        score    = self._option.score
        colour, bg, border = self._colours()

        # Convert km to NM correctly
        distance_nm = features.distance_km / KM_TO_NM

        self.setStyleSheet(f"""
            ResultCard {{
                background-color: {bg};
                border: 1px solid {border};
                border-radius: {Spacing.RADIUS}px;
            }}
        """)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(Spacing.LG, Spacing.MD, Spacing.LG, Spacing.MD)
        outer.setSpacing(Spacing.SM)

        # ── Row 1: rank + ICAO + name + distance (NM) + confidence ──
        top = QHBoxLayout()
        top.setSpacing(Spacing.MD)

        rank_lbl = QLabel(f"#{self._rank}")
        rank_lbl.setFixedWidth(30)
        rank_lbl.setStyleSheet(f"""
            color: {colour};
            font-size: {Font.SZ_MD}px;
            font-weight: bold;
            background: transparent;
        """)
        rank_lbl.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        top.addWidget(rank_lbl)

        icao_lbl = QLabel(airport.icao)
        icao_lbl.setFixedWidth(72)
        icao_lbl.setStyleSheet(f"""
            color: {colour};
            font-size: {Font.SZ_XL}px;
            font-family: "{Font.MONO}", "Courier New";
            font-weight: bold;
            background: transparent;
        """)
        icao_lbl.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        top.addWidget(icao_lbl)

        name_col = QVBoxLayout()
        name_col.setSpacing(2)

        name_lbl = QLabel(airport.name)
        name_lbl.setStyleSheet(f"""
            color: {Colour.TEXT_PRIMARY};
            font-size: {Font.SZ_BODY}px;
            font-weight: bold;
            background: transparent;
        """)
        name_lbl.setWordWrap(True)
        name_col.addWidget(name_lbl)

        city_lbl = QLabel(f"{airport.city}, {airport.country}")
        city_lbl.setStyleSheet(f"""
            color: {Colour.TEXT_SECONDARY};
            font-size: {Font.SZ_SM}px;
            background: transparent;
        """)
        name_col.addWidget(city_lbl)
        top.addLayout(name_col, stretch=1)

        # Distance — correctly in NM
        dist_lbl = QLabel(f"{distance_nm:.1f} NM")
        dist_lbl.setStyleSheet(f"""
            color: {Colour.TEXT_SECONDARY};
            font-size: {Font.SZ_BODY}px;
            font-family: "{Font.MONO}", "Courier New";
            background: transparent;
            min-width: 80px;
        """)
        dist_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        top.addWidget(dist_lbl)

        # Confidence badge
        score_pct = int(score * 100)
        if score_pct >= 75:
            sc, sb = Colour.GREEN, Colour.GREEN_BG
        elif score_pct >= 40:
            sc, sb = Colour.AMBER, Colour.AMBER_BG
        else:
            sc, sb = Colour.TEXT_SECONDARY, Colour.BG_INPUT

        conf_lbl = QLabel(f"{score_pct}%")
        conf_lbl.setFixedSize(54, 30)
        conf_lbl.setStyleSheet(f"""
            color: {sc};
            background-color: {sb};
            border: 1px solid {sc};
            border-radius: 3px;
            font-size: {Font.SZ_SM}px;
            font-family: "{Font.MONO}", "Courier New";
            font-weight: bold;
        """)
        conf_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top.addWidget(conf_lbl)

        outer.addLayout(top)

        # Divider
        div = QFrame()
        div.setFrameShape(QFrame.Shape.HLine)
        div.setFixedHeight(1)
        div.setStyleSheet(f"background-color: {Colour.DIVIDER}; border: none;")
        outer.addWidget(div)

        # Explanation bullets
        for reason in self._expl.ranking_reasons:
            lbl = QLabel(f"    •  {reason}")
            lbl.setStyleSheet(f"""
                color: {Colour.TEXT_PRIMARY};
                font-size: {Font.SZ_BODY}px;
                background: transparent;
            """)
            lbl.setWordWrap(True)
            outer.addWidget(lbl)

        # Summary sentence
        if self._expl.summary:
            sum_lbl = QLabel(self._expl.summary)
            sum_lbl.setStyleSheet(f"""
                color: {Colour.TEXT_SECONDARY};
                font-size: {Font.SZ_SM}px;
                font-style: italic;
                background: transparent;
                padding-top: 2px;
            """)
            sum_lbl.setWordWrap(True)
            outer.addWidget(sum_lbl)

        # Advisory note
        adv = QLabel("ADVISORY ONLY  —  FINAL DECISION: PILOT IN COMMAND")
        adv.setStyleSheet(f"""
            color: {Colour.AMBER};
            font-size: {Font.SZ_XS}px;
            font-weight: bold;
            letter-spacing: 1px;
            background: transparent;
        """)
        outer.addWidget(adv)

        # Select button — no tick icon
        btn = QPushButton(f"Select {airport.icao} as Diversion Airport")
        btn.setMinimumHeight(Spacing.TOUCH_MIN)
        btn.setStyleSheet(f"""
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
        btn.clicked.connect(lambda: self.confirmed.emit(airport.icao))
        outer.addWidget(btn)