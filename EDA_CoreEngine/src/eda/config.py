"""
config.py

Central configuration for the EDA Core Engine.

This file stores tunable parameters that control system behaviour.
Keeping these values separate from algorithm code makes the system
easier to tune, test, and maintain.
"""

# ----------------------------------------------------
# Ranking weights
# ----------------------------------------------------

DISTANCE_WEIGHT = 0.6
RUNWAY_WEIGHT = 0.4


# ----------------------------------------------------
# Capability bonuses
# ----------------------------------------------------

MEDICAL_BONUS = 0.5
RESCUE_BONUS = 0.2


# ----------------------------------------------------
# System parameters
# ----------------------------------------------------

DEFAULT_TOP_K = 3
DEFAULT_MAX_RANGE_KM = 800.0