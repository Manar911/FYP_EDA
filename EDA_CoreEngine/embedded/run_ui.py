"""
run_ui.py

Single entry point for the EDA embedded UI.

Project structure assumed:
    EDA_CoreEngine/
        src/eda/          ← core engine modules
        models/           ← trained model
        embedded/
            run_ui.py     ← this file
            ui/           ← UI modules
            logs/         ← runtime logs

Usage:
    Development (from embedded/ folder):
        python run_ui.py

    Pi kiosk deployment (fullscreen):
        python run_ui.py --fullscreen

    Or via start.sh which sets everything up automatically.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
# embedded/ is one level inside EDA_CoreEngine/
# src/ is at EDA_CoreEngine/src/
# models/ is at EDA_CoreEngine/models/

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH     = PROJECT_ROOT / "src"
MODELS_PATH  = PROJECT_ROOT / "models"

# Add src/ so that `from eda.xxx import yyy` works
sys.path.insert(0, str(SRC_PATH))

# Add embedded/ so that `from ui.xxx import yyy` works
sys.path.insert(0, str(Path(__file__).resolve().parent))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from ui.app import EDAMainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("EDA Core Engine")
    app.setOrganizationName("EDA")

    # Default font
    font = QFont("Arial", 14)
    app.setFont(font)

    # Disable high-DPI scaling — Pi display is native 1024x600
    app.setAttribute(Qt.ApplicationAttribute.AA_Use96Dpi)

    window = EDAMainWindow()

    # Fullscreen kiosk mode for Pi deployment
    if "--fullscreen" in sys.argv:
        window.showFullScreen()
    else:
        window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()