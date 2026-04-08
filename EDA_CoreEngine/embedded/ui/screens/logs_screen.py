"""
logs_screen.py

Runtime log viewer screen.

Displays the in-memory deque of compact decision log entries.
Max 200 entries — oldest auto-evicted (deque maxlen).
No file I/O at render time — pure in-memory display.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Any

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QListWidget,
    QListWidgetItem, QMessageBox,
)
from PySide6.QtCore import Signal, Qt

from ui.theme import Colour, Font, Spacing

MAX_LOGS = 200


class LogsScreen(QWidget):
    """
    Displays the runtime log buffer.
    Entries are compact dicts from decision_report_to_compact_dict().
    """

    back_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # In-memory buffer — deque handles retention automatically
        self._buffer: deque[dict[str, Any]] = deque(maxlen=MAX_LOGS)
        self._setup_ui()

    # ── Public API ─────────────────────────────────────────────────────────────

    def add_entry(self, compact_log: dict[str, Any]) -> None:
        """
        Add one compact log entry to the buffer and refresh the list.
        Called after each successful inference + pilot confirmation.
        """
        self._buffer.appendleft(compact_log)
        self._refresh_list()
        self._update_count()

    def clear(self) -> None:
        self._buffer.clear()
        self._list.clear()
        self._update_count()

    # ── UI construction ────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_header())

        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(
            Spacing.LG, Spacing.LG, Spacing.LG, Spacing.LG
        )
        body_layout.setSpacing(Spacing.MD)

        self._list = QListWidget()
        self._list.setAlternatingRowColors(True)
        self._list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self._list.setStyleSheet(f"""
            QListWidget {{
                background-color: {Colour.BG_SECONDARY};
                border: 1px solid {Colour.BORDER_DEFAULT};
                border-radius: {Spacing.CARD_RADIUS}px;
                color: {Colour.TEXT_PRIMARY};
                font-size: {Font.SIZE_BODY}px;
                outline: none;
            }}
            QListWidget::item {{
                padding: 10px 14px;
                border-bottom: 1px solid {Colour.DIVIDER};
            }}
            QListWidget::item:alternate {{
                background-color: {Colour.LOG_ROW_ALT};
            }}
        """)
        body_layout.addWidget(self._list, stretch=1)

        # Empty state label
        self._empty_label = QLabel("No decisions logged yet.")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet(f"""
            color: {Colour.TEXT_MUTED};
            font-size: {Font.SIZE_MEDIUM}px;
            padding: 40px;
        """)
        body_layout.addWidget(self._empty_label)

        # Clear button
        clear_btn = QPushButton("CLEAR LOG")
        clear_btn.setFixedHeight(Spacing.TOUCH_MIN)
        clear_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colour.BACK};
                color: {Colour.TEXT_SECONDARY};
                border: 1px solid {Colour.BORDER_DEFAULT};
                border-radius: {Spacing.BUTTON_RADIUS}px;
                font-size: {Font.SIZE_BODY}px;
            }}
            QPushButton:pressed {{
                background-color: {Colour.BACK_HOVER};
            }}
        """)
        clear_btn.clicked.connect(self._on_clear)
        body_layout.addWidget(clear_btn)

        root.addWidget(body, stretch=1)

        self._update_visibility()

    def _build_header(self) -> QWidget:
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

        title = QLabel("RUNTIME LOG")
        title.setStyleSheet(f"""
            color: {Colour.TEXT_PRIMARY};
            font-size: {Font.SIZE_MEDIUM}px;
            font-weight: bold;
            letter-spacing: 1px;
        """)
        layout.addWidget(title, stretch=1)

        self._count_label = QLabel(f"0 / {MAX_LOGS}")
        self._count_label.setStyleSheet(f"""
            color: {Colour.TEXT_MUTED};
            font-size: {Font.SIZE_SMALL}px;
        """)
        layout.addWidget(self._count_label)

        return header

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _refresh_list(self) -> None:
        self._list.clear()
        for entry in self._buffer:
            text = self._format_entry(entry)
            item = QListWidgetItem(text)
            item.setForeground(Qt.GlobalColor.white)
            self._list.addItem(item)
        self._update_visibility()

    def _format_entry(self, entry: dict[str, Any]) -> str:
        ts        = entry.get("timestamp", "")[:19].replace("T", "  ")
        scenario  = entry.get("scenario", {})
        emergency = scenario.get("emergency_type", "").upper()
        aircraft  = scenario.get("aircraft_type", "")
        fuel      = scenario.get("fuel_state", "").upper()

        selected  = entry.get("selected_option") or {}
        icao      = selected.get("airport_icao", "—")
        score_val = selected.get("score", 0.0)
        dist      = selected.get("distance_km", 0.0)
        score_pct = int(score_val * 100)

        feasible  = entry.get("feasible_airports_count", "?")

        return (
            f"{ts}   {emergency:<14} {aircraft:<12} "
            f"→  {icao}  ({score_pct}%)   "
            f"{dist:.1f} km   {fuel}   {feasible} feasible"
        )

    def _update_count(self) -> None:
        count = len(self._buffer)
        self._count_label.setText(f"{count} / {MAX_LOGS}")

    def _update_visibility(self) -> None:
        has_entries = len(self._buffer) > 0
        self._list.setVisible(has_entries)
        self._empty_label.setVisible(not has_entries)

    def _on_clear(self) -> None:
        if not self._buffer:
            return
        msg = QMessageBox(self)
        msg.setWindowTitle("Clear Log")
        msg.setText(f"Clear all {len(self._buffer)} log entries?")
        msg.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        msg.setDefaultButton(QMessageBox.StandardButton.No)
        msg.setStyleSheet(f"""
            QMessageBox {{
                background-color: {Colour.BG_SECONDARY};
                color: {Colour.TEXT_PRIMARY};
            }}
            QLabel {{
                color: {Colour.TEXT_PRIMARY};
                font-size: {Font.SIZE_BODY}px;
            }}
            QPushButton {{
                background-color: {Colour.BACK};
                color: {Colour.TEXT_PRIMARY};
                border: 1px solid {Colour.BORDER_DEFAULT};
                border-radius: 4px;
                padding: 6px 20px;
                font-size: {Font.SIZE_BODY}px;
            }}
        """)
        if msg.exec() == QMessageBox.StandardButton.Yes:
            self.clear()