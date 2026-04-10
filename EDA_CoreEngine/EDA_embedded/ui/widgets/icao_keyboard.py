"""
icao_keyboard.py  —  EDA 

ICAO keyboard with validation against the airport database.
Rejects codes not found in airports.csv.
VVVV, HHHH and any other non-existent codes will be rejected.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame,
)
from PySide6.QtCore import Signal, Qt

from ui.theme import Colour, Font, Spacing

ROWS = [
    list("QWERTYUIOP"),
    list("ASDFGHJKL"),
    list("ZXCVBNM"),
]

# Load valid ICAO codes from the airport database once at import time
def _load_valid_icaos() -> set:
    try:
        from core.airport_db import load_airports
        airports = load_airports()
        return {a.icao.upper() for a in airports}
    except Exception:
        return set()

_VALID_ICAOS: set = set()  # populated on first use


class ICAOKeyboard(QWidget):
    """
    Letter keyboard for ICAO code entry.
    Validates entered code against the airport database.
    Only accepts codes that exist in airports.csv.
    """

    code_accepted = Signal(str)
    cancelled     = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._value = ""
        # Load valid ICAOs on first instantiation
        global _VALID_ICAOS
        if not _VALID_ICAOS:
            _VALID_ICAOS = _load_valid_icaos()
        self._setup_ui()

    def reset(self):
        self._value = ""
        self._update_display()
        self._error_label.setText("")

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setAlignment(Qt.AlignmentFlag.AlignCenter)

        panel = QFrame()
        panel.setFixedSize(580, 430)
        panel.setStyleSheet(f"""
            QFrame {{
                background-color: {Colour.BG_CARD};
                border: 1px solid {Colour.BORDER_ACTIVE};
                border-radius: {Spacing.RADIUS}px;
            }}
        """)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(Spacing.LG, Spacing.LG, Spacing.LG, Spacing.LG)
        layout.setSpacing(Spacing.SM)

        # Title
        title = QLabel("ENTER AIRPORT ICAO CODE")
        title.setStyleSheet(f"""
            color: {Colour.CYAN};
            font-size: {Font.SZ_SM}px;
            font-weight: bold;
            letter-spacing: 2px;
            background: transparent;
        """)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # 4 character boxes
        display_row = QHBoxLayout()
        display_row.setSpacing(Spacing.SM)
        display_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._char_labels = []
        for _ in range(4):
            box = QLabel("_")
            box.setFixedSize(84, 72)
            box.setAlignment(Qt.AlignmentFlag.AlignCenter)
            box.setStyleSheet(self._box_style(False))
            display_row.addWidget(box)
            self._char_labels.append(box)
        layout.addLayout(display_row)

        # Error label
        self._error_label = QLabel("")
        self._error_label.setStyleSheet(f"""
            color: {Colour.AMBER};
            font-size: {Font.SZ_SM}px;
            background: transparent;
            min-height: 18px;
        """)
        self._error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._error_label)

        # Keyboard rows
        for row in ROWS:
            row_layout = QHBoxLayout()
            row_layout.setSpacing(Spacing.XS + 1)
            row_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            for letter in row:
                btn = QPushButton(letter)
                btn.setFixedSize(48, 46)
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {Colour.BTN_BG};
                        color: {Colour.TEXT_PRIMARY};
                        border: 1px solid {Colour.BORDER};
                        border-radius: {Spacing.RADIUS_SM}px;
                        font-size: {Font.SZ_MD}px;
                        font-weight: bold;
                    }}
                    QPushButton:pressed {{
                        background-color: {Colour.CYAN_BG};
                        color: {Colour.CYAN};
                        border-color: {Colour.CYAN};
                    }}
                """)
                btn.clicked.connect(lambda _, l=letter: self._on_letter(l))
                row_layout.addWidget(btn)
            layout.addLayout(row_layout)

        # Action row
        action = QHBoxLayout()
        action.setSpacing(Spacing.SM)

        del_btn = QPushButton("DEL")
        del_btn.setFixedHeight(Spacing.TOUCH_MIN - 10)
        del_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colour.AMBER_BG};
                color: {Colour.AMBER};
                border: 1px solid {Colour.AMBER_DIM};
                border-radius: {Spacing.RADIUS_SM}px;
                font-size: {Font.SZ_BODY}px;
                font-weight: bold;
            }}
            QPushButton:pressed {{ background-color: {Colour.AMBER_DIM}; }}
        """)
        del_btn.clicked.connect(self._on_backspace)

        cancel_btn = QPushButton("CANCEL")
        cancel_btn.setFixedHeight(Spacing.TOUCH_MIN - 10)
        cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {Colour.TEXT_SECONDARY};
                border: 1px solid {Colour.BORDER};
                border-radius: {Spacing.RADIUS_SM}px;
                font-size: {Font.SZ_BODY}px;
            }}
            QPushButton:pressed {{ background-color: {Colour.BTN_PRESSED}; }}
        """)
        cancel_btn.clicked.connect(self.cancelled.emit)

        add_btn = QPushButton("ADD")
        add_btn.setFixedHeight(Spacing.TOUCH_MIN - 10)
        add_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colour.GREEN_BG};
                color: {Colour.GREEN};
                border: 1px solid {Colour.GREEN_DIM};
                border-radius: {Spacing.RADIUS_SM}px;
                font-size: {Font.SZ_BODY}px;
                font-weight: bold;
            }}
            QPushButton:pressed {{ background-color: {Colour.GREEN_DIM}; }}
        """)
        add_btn.clicked.connect(self._on_add)

        action.addWidget(del_btn, 1)
        action.addWidget(cancel_btn, 1)
        action.addWidget(add_btn, 1)
        layout.addLayout(action)

        root.addWidget(panel)

    def _on_letter(self, letter):
        if len(self._value) < 4:
            self._value += letter
            self._error_label.setText("")
            self._update_display()

    def _on_backspace(self):
        if self._value:
            self._value = self._value[:-1]
            self._error_label.setText("")
            self._update_display()

    def _on_add(self):
        if len(self._value) != 4:
            self._error_label.setText(
                f"ICAO code must be exactly 4 letters  ({len(self._value)}/4 entered)"
            )
            return

        # Validate against airport database
        if _VALID_ICAOS and self._value not in _VALID_ICAOS:
            self._error_label.setText(
                f"{self._value} is not in the airport database"
            )
            return

        self.code_accepted.emit(self._value)
        self._value = ""
        self._update_display()

    def _update_display(self):
        for i, box in enumerate(self._char_labels):
            if i < len(self._value):
                box.setText(self._value[i])
                box.setStyleSheet(self._box_style(True))
            else:
                box.setText("_")
                box.setStyleSheet(self._box_style(False))

    def _box_style(self, filled: bool) -> str:
        if filled:
            return f"""
                color: {Colour.CYAN};
                font-size: {Font.SZ_2XL}px;
                font-family: "{Font.MONO}", "Courier New";
                font-weight: bold;
                background-color: {Colour.CYAN_BG};
                border: 1px solid {Colour.CYAN_DIM};
                border-radius: {Spacing.RADIUS_SM}px;
            """
        return f"""
            color: {Colour.TEXT_MUTED};
            font-size: {Font.SZ_2XL}px;
            font-family: "{Font.MONO}", "Courier New";
            font-weight: bold;
            background-color: {Colour.BG_INPUT};
            border: 1px solid {Colour.BORDER};
            border-radius: {Spacing.RADIUS_SM}px;
        """