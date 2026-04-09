from __future__ import annotations
from collections import deque
from typing import Any
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget, QListWidgetItem, QMessageBox
from PySide6.QtCore import Signal, Qt
from ui.theme import Colour, Font, Spacing
MAX_LOGS = 200
class LogsScreen(QWidget):
    back_requested = Signal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self._buffer: deque[dict[str, Any]] = deque(maxlen=MAX_LOGS)
        self._setup_ui()
    def add_entry(self, log):
        self._buffer.appendleft(log)
        self._refresh()
        self._count_lbl.setText(f"{len(self._buffer)} / {MAX_LOGS}")
    def clear(self):
        self._buffer.clear()
        self._list.clear()
        self._count_lbl.setText(f"0 / {MAX_LOGS}")
        self._update_vis()
    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0,0,0,0)
        root.setSpacing(0)
        root.addWidget(self._build_header())
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(Spacing.LG,Spacing.LG,Spacing.LG,Spacing.LG)
        lay.setSpacing(Spacing.MD)
        self._list = QListWidget()
        self._list.setAlternatingRowColors(True)
        self._list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        lay.addWidget(self._list, stretch=1)
        self._empty = QLabel("No decisions logged yet.")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty.setStyleSheet(f"color: {Colour.TEXT_MUTED}; font-size: {Font.SZ_MD}px; padding: 40px; background: transparent;")
        lay.addWidget(self._empty)
        clr = QPushButton("Clear Log")
        clr.setFixedHeight(Spacing.TOUCH_MIN)
        clr.setStyleSheet(f"QPushButton {{ background: transparent; color: {Colour.AMBER}; border: 1px solid {Colour.AMBER_DIM}; border-radius: {Spacing.RADIUS_SM}px; font-size: {Font.SZ_BODY}px; font-weight: bold; }} QPushButton:pressed {{ background: {Colour.AMBER_BG}; }}")
        clr.clicked.connect(self._on_clear)
        lay.addWidget(clr)
        root.addWidget(body, stretch=1)
        self._update_vis()
    def _build_header(self):
        h = QWidget()
        h.setFixedHeight(Spacing.HEADER_H)
        h.setStyleSheet(f"background-color: {Colour.BG_HEADER}; border-bottom: 1px solid {Colour.BORDER};")
        lay = QHBoxLayout(h)
        lay.setContentsMargins(Spacing.LG,0,Spacing.LG,0)
        lay.setSpacing(Spacing.MD)
        back = QPushButton("Back")
        back.setFixedHeight(36)
        back.setStyleSheet(f"QPushButton {{ background: transparent; color: {Colour.TEXT_SECONDARY}; border: 1px solid {Colour.BORDER}; border-radius: {Spacing.RADIUS_SM}px; font-size: {Font.SZ_SM}px; padding: 0 14px; }} QPushButton:pressed {{ background: {Colour.BTN_PRESSED}; }}")
        back.clicked.connect(self.back_requested.emit)
        lay.addWidget(back)
        t = QLabel("Runtime Log")
        t.setStyleSheet(f"color: {Colour.CYAN}; font-size: {Font.SZ_MD}px; font-weight: bold; background: transparent;")
        lay.addWidget(t, stretch=1)
        self._count_lbl = QLabel(f"0 / {MAX_LOGS}")
        self._count_lbl.setStyleSheet(f"color: {Colour.TEXT_MUTED}; font-size: {Font.SZ_SM}px; background: transparent;")
        lay.addWidget(self._count_lbl)
        return h
    def _refresh(self):
        self._list.clear()
        for e in self._buffer:
            self._list.addItem(QListWidgetItem(self._fmt(e)))
        self._update_vis()
    def _fmt(self, e):
        ts = e.get("timestamp","")[:19].replace("T","  ")
        sc = e.get("scenario",{})
        sel = e.get("selected_option") or {}
        return (f"{ts}   {sc.get('emergency_type','').upper():<13}  {sc.get('aircraft_type',''):<10}  "
                f"→  {sel.get('airport_icao','—')}  {int(sel.get('score',0)*100)}%  "
                f"{sel.get('distance_km',0):.0f} km   {sc.get('fuel_state','').upper()}")
    def _update_vis(self):
        has = len(self._buffer) > 0
        self._list.setVisible(has)
        self._empty.setVisible(not has)
    def _on_clear(self):
        if not self._buffer: return
        msg = QMessageBox(self)
        msg.setWindowTitle("Clear Log")
        msg.setText(f"Clear all {len(self._buffer)} log entries?")
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if msg.exec() == QMessageBox.StandardButton.Yes: self.clear()