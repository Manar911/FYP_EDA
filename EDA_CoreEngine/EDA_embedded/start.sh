#!/bin/bash
# start.sh — EDA embedded system boot script
# Runs automatically on Pi startup via autostart

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Disable screen blanking
xset s off 2>/dev/null
xset -dpms 2>/dev/null
xset s noblank 2>/dev/null

cd "$SCRIPT_DIR"
python run_ui.py --fullscreen