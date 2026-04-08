#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# start.sh
#
# Boot script for EDA embedded system on Raspberry Pi.
#
# Launches the EDA UI in fullscreen kiosk mode.
# Run automatically on boot via systemd or desktop autostart.
#
# Usage:
#   chmod +x start.sh
#   ./start.sh
#
# To run on boot, add to /etc/xdg/autostart/eda.desktop or systemd service.
# ─────────────────────────────────────────────────────────────────────────────

# Location of the embedded system on the Pi
EMBEDDED_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$EMBEDDED_DIR")"

# Python path — points to src/ in the project root
export PYTHONPATH="$PROJECT_ROOT/src"

# Disable screen saver and blanking for kiosk use
xset s off 2>/dev/null
xset -dpms 2>/dev/null
xset s noblank 2>/dev/null

# Launch the UI in fullscreen mode
cd "$EMBEDDED_DIR"
python run_ui.py --fullscreen