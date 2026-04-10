"""
run_ui.py  —  EDA 

Entry point for the EDA embedded system.


Usage:
    python run_ui.py              # windowed
    python run_ui.py --fullscreen # Pi kiosk
"""

from __future__ import annotations
import sys
import os
from pathlib import Path

#  Force 1:1 pixel mapping for Pi display 
os.environ["QT_SCALE_FACTOR"]             = "1"
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"
os.environ["QT_SCREEN_SCALE_FACTORS"]     = "1"

#  Paths 
ROOT = Path(__file__).resolve().parent   # EDA_embedded/

# core/ is the eda package — import as 'eda'
sys.path.insert(0, str(ROOT / "core"))

# ui/ for UI imports
sys.path.insert(0, str(ROOT))

#  Launch 
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