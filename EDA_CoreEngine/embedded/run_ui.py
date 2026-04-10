"""
run_ui.py  —  EDA 

Target: Raspberry Pi 5 — 7-inch IPS 1024x600 capacitive touch display.

Scaling is forced to 1:1 always.
On laptop this means the window appears at true 1024x600 pixels.
On Pi 1024x600 native display it fills the screen perfectly.

Usage:
    python run_ui.py              # windowed (testing)
    python run_ui.py --fullscreen # fullscreen kiosk (Pi deployment)
"""

from __future__ import annotations
import sys
import os
from pathlib import Path

# ── Force 1:1 pixel mapping ───────────────────────────────────────────────────
# Must be set before QApplication is created.
# Ensures mouse events and paint coordinates use identical pixel space.
# Critical for map pointer accuracy on the Pi 1024x600 display.
os.environ["QT_SCALE_FACTOR"]            = "1"
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"
os.environ["QT_SCREEN_SCALE_FACTORS"]    = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH     = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont

from ui.app import EDAMainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("EDA Core Engine")
    app.setOrganizationName("EDA")
    app.setFont(QFont("Roboto", 14))

    window = EDAMainWindow()

    if "--fullscreen" in sys.argv:
        window.showFullScreen()
    else:
        window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()