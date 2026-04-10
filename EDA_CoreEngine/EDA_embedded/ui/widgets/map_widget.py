"""
map_widget.py  —  EDA 

Pointer fix — device pixel ratio correction applied inside the widget.

On Windows at 150% scaling, devicePixelRatio = 1.5
Mouse events arrive in logical pixels (already divided by DPR).
paintEvent also uses logical pixels.
So they should match — but Qt's mapToGlobal/mapFromGlobal can
introduce offset when the widget is inside a scroll area or nested layout.

Fix: use mapFromGlobal(QCursor.pos()) instead of e.pos() to get
the cursor position in widget-local logical coordinates directly.
This bypasses any coordinate transformation issues from nested layouts.
"""

from __future__ import annotations
from pathlib import Path

from PySide6.QtWidgets import QWidget, QSizePolicy
from PySide6.QtCore import Signal, Qt, QPoint
from PySide6.QtGui import (
    QPainter, QPixmap, QColor, QPen, QBrush,
    QFont, QMouseEvent, QCursor,
)

from ui.theme import Colour, Font

ASSETS   = Path(__file__).resolve().parent.parent.parent / "assets"
MAP_FILE = ASSETS / "world_map.png"


class MapWidget(QWidget):
    position_selected = Signal(float, float)

    def __init__(self, lat: float = 26.2708, lon: float = 50.6336, parent=None):
        super().__init__(parent)
        self._lat = lat
        self._lon = lon
        self._pixmap = None

        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.setMinimumSize(400, 150)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setStyleSheet(f"""
            MapWidget {{
                background-color: {Colour.BG_INPUT};
                border: 1px solid {Colour.BORDER};
                border-radius: 4px;
            }}
        """)

        if MAP_FILE.exists():
            px = QPixmap(str(MAP_FILE))
            if not px.isNull():
                self._pixmap = px

    # ── Coordinate conversion ─────────────────────────────────────────────────

    def _lon_to_x(self, lon: float) -> int:
        return int((lon + 180.0) / 360.0 * self.width())

    def _lat_to_y(self, lat: float) -> int:
        return int((90.0 - lat) / 180.0 * self.height())

    def _x_to_lon(self, x: int) -> float:
        return (x / self.width()) * 360.0 - 180.0

    def _y_to_lat(self, y: int) -> float:
        return 90.0 - (y / self.height()) * 180.0

    # ── Public API ────────────────────────────────────────────────────────────

    def set_position(self, lat: float, lon: float) -> None:
        self._lat = lat
        self._lon = lon
        self.update()

    def get_position(self):
        return self._lat, self._lon

    # ── Events ────────────────────────────────────────────────────────────────

    def mousePressEvent(self, e: QMouseEvent) -> None:
        if e.button() == Qt.MouseButton.LeftButton:
            # Map global cursor position to widget-local coordinates.
            # This is the most reliable method across all platforms and
            # DPI scaling settings — it bypasses nested layout offsets.
            global_pos = QCursor.pos()
            local_pos  = self.mapFromGlobal(global_pos)
            x = local_pos.x()
            y = local_pos.y()

            # Clamp to widget bounds
            x = max(0, min(x, self.width()  - 1))
            y = max(0, min(y, self.height() - 1))

            lat = round(self._y_to_lat(y), 4)
            lon = round(self._x_to_lon(x), 4)
            lat = max(-90.0,  min(90.0,  lat))
            lon = max(-180.0, min(180.0, lon))

            self._lat = lat
            self._lon = lon
            self.update()
            self.position_selected.emit(lat, lon)

    # ── Paint ─────────────────────────────────────────────────────────────────

    def paintEvent(self, _) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        p.fillRect(0, 0, w, h, QColor(Colour.BG_INPUT))

        if self._pixmap:
            scaled = self._pixmap.scaled(
                w, h,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            p.setOpacity(0.90)
            p.drawPixmap(0, 0, w, h, scaled)
            p.setOpacity(1.0)

        # Graticule
        pen = QPen(QColor(20, 40, 60))
        pen.setWidth(1)
        p.setPen(pen)
        for lat in range(-60, 91, 30):
            y = self._lat_to_y(lat)
            p.drawLine(0, y, w, y)
        for lon in range(-150, 181, 30):
            x = self._lon_to_x(lon)
            p.drawLine(x, 0, x, h)

        # Equator / prime meridian
        pen.setColor(QColor(0, 70, 100))
        p.setPen(pen)
        p.drawLine(0, self._lat_to_y(0), w, self._lat_to_y(0))
        p.drawLine(self._lon_to_x(0), 0, self._lon_to_x(0), h)

        # Aircraft marker — uses same conversion functions as mouse handler
        x = self._lon_to_x(self._lon)
        y = self._lat_to_y(self._lat)
        r = 9

        pen = QPen(QColor(Colour.CYAN))
        pen.setWidth(2)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(x - r - 3, y - r - 3, (r + 3) * 2, (r + 3) * 2)

        pen.setWidth(1)
        p.setPen(pen)
        p.drawLine(x - r - 10, y, x - r,      y)
        p.drawLine(x + r,      y, x + r + 10, y)
        p.drawLine(x, y - r - 10, x, y - r)
        p.drawLine(x, y + r,      x, y + r + 10)

        p.setBrush(QBrush(QColor(Colour.CYAN)))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(x - 4, y - 4, 8, 8)

        # Coordinate readout
        text = f"  {self._lat:+.4f}\u00b0   {self._lon:+.4f}\u00b0  "
        fnt = QFont("Roboto Mono", Font.SZ_SM)
        fnt.setBold(True)
        p.setFont(fnt)
        fm = p.fontMetrics()
        tw = fm.horizontalAdvance(text)
        th = fm.height() + 8
        p.fillRect(0, h - th, tw + 4, th, QColor(Colour.BG_CARD))
        p.setPen(QPen(QColor(Colour.CYAN)))
        p.drawText(4, h - 6, text)

        p.end()