"""
numpad_widget.py  —  Numeric keypad for LAT / LON entry.
Large buttons, Roboto font, clean professional layout.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QFrame,
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QFont

from ui.theme import Colour, Font, Spacing


class NumpadWidget(QWidget):
    value_accepted = Signal(str)
    cancelled      = Signal()

    def __init__(
        self,
        label: str,
        initial_value: str = "",
        min_val: float = -180.0,
        max_val: float = 180.0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._label   = label
        self._min     = min_val
        self._max     = max_val
        self._value   = initial_value
        self._setup_ui()

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setAlignment(Qt.AlignmentFlag.AlignCenter)

        panel = QFrame()
        panel.setFixedSize(400, 520)
        panel.setStyleSheet(f"""
            QFrame {{
                background-color: {Colour.BG_CARD};
                border: 1px solid {Colour.BORDER_ACTIVE};
                border-radius: {Spacing.RADIUS}px;
            }}
        """)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(Spacing.LG, Spacing.LG, Spacing.LG, Spacing.LG)
        layout.setSpacing(Spacing.MD)

        # Label
        lbl = QLabel(self._label)
        lbl.setStyleSheet(f"""
            color: {Colour.CYAN};
            font-size: {Font.SZ_SM}px;
            font-weight: bold;
            letter-spacing: 2px;
            background: transparent;
        """)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl)

        # Display
        self._display = QLabel(self._value or "0")
        self._display.setStyleSheet(f"""
            color: {Colour.CYAN};
            font-size: {Font.SZ_3XL}px;
            font-family: "{Font.MONO}", "Courier New";
            font-weight: bold;
            background-color: {Colour.BG_INPUT};
            border: 1px solid {Colour.BORDER};
            border-radius: {Spacing.RADIUS_SM}px;
            padding: 6px 16px;
            min-height: 52px;
        """)
        self._display.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self._display)

        # Error
        self._err = QLabel("")
        self._err.setStyleSheet(f"color: {Colour.AMBER}; font-size: {Font.SZ_SM}px; background: transparent; min-height: 16px;")
        self._err.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._err)

        # Number grid
        grid = QGridLayout()
        grid.setSpacing(Spacing.SM)
        keys = [
            ("7",0,0),("8",0,1),("9",0,2),("⌫",0,3),
            ("4",1,0),("5",1,1),("6",1,2),("CLR",1,3),
            ("1",2,0),("2",2,1),("3",2,2),("±",2,3),
            ("0",3,0),(".",3,1),
        ]
        for text, row, col in keys:
            is_num = text.isdigit() or text == "."
            btn = QPushButton(text)
            btn.setFixedSize(76, 62)
            if is_num:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {Colour.BTN_BG};
                        color: {Colour.TEXT_PRIMARY};
                        border: 1px solid {Colour.BORDER};
                        border-radius: {Spacing.RADIUS_SM}px;
                        font-size: {Font.SZ_LG}px;
                        font-weight: bold;
                        font-family: "{Font.MONO}", "Courier New";
                    }}
                    QPushButton:pressed {{
                        background-color: {Colour.CYAN_BG};
                        color: {Colour.CYAN};
                        border-color: {Colour.CYAN};
                    }}
                """)
            else:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {Colour.AMBER_BG};
                        color: {Colour.AMBER};
                        border: 1px solid {Colour.AMBER_DIM};
                        border-radius: {Spacing.RADIUS_SM}px;
                        font-size: {Font.SZ_MD}px;
                        font-weight: bold;
                    }}
                    QPushButton:pressed {{
                        background-color: {Colour.AMBER_DIM};
                        color: {Colour.TEXT_PRIMARY};
                    }}
                """)
            btn.clicked.connect(lambda _, t=text: self._key(t))
            grid.addWidget(btn, row, col)
        layout.addLayout(grid)

        # Actions
        row = QHBoxLayout()
        row.setSpacing(Spacing.SM)
        for text, colour, bg, dim, slot in [
            ("CANCEL", Colour.TEXT_SECONDARY, "transparent", Colour.BORDER, self.cancelled.emit),
            ("DONE", Colour.GREEN, Colour.GREEN_BG, Colour.GREEN_DIM, self._done),
        ]:
            b = QPushButton(text)
            b.setFixedHeight(Spacing.TOUCH_MIN - 8)
            b.setStyleSheet(f"""
                QPushButton {{
                    background-color: {bg};
                    color: {colour};
                    border: 1px solid {dim};
                    border-radius: {Spacing.RADIUS_SM}px;
                    font-size: {Font.SZ_BODY}px;
                    font-weight: bold;
                }}
                QPushButton:pressed {{ background-color: {dim}; color: {Colour.TEXT_PRIMARY}; }}
            """)
            b.clicked.connect(slot)
            row.addWidget(b, stretch=1)
        layout.addLayout(row)
        root.addWidget(panel)

    def _key(self, k: str) -> None:
        self._err.setText("")
        if k == "CLR":
            self._value = ""
        elif k == "⌫":
            self._value = self._value[:-1]
        elif k == "±":
            self._value = self._value[1:] if self._value.startswith("-") else "-" + self._value
        elif k == ".":
            if "." not in self._value:
                self._value += "."
        else:
            digits = self._value.replace("-","").replace(".","")
            if len(digits) < 8:
                self._value += k
        self._display.setText(self._value or "0")

    def _done(self) -> None:
        raw = self._value.strip()
        if not raw or raw in ("-", "."):
            self._err.setText("Please enter a value")
            return
        try:
            val = float(raw)
        except ValueError:
            self._err.setText("Invalid number")
            return
        if not (self._min <= val <= self._max):
            self._err.setText(f"Must be between {self._min} and {self._max}")
            return
        self.value_accepted.emit(raw)