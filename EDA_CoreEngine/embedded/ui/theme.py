"""
theme.py

Central theme configuration for the EDA embedded UI.

All colours, fonts, spacing, and stylesheet constants live here.
No hardcoded values anywhere else in the UI code.

Designed for:
- 7-inch IPS display at 1024x600
- Dark theme for low-light cockpit environments
- High contrast for readability under stress
- Touch-first sizing (minimum 70px touch targets)
"""

from __future__ import annotations


# ── Colours ───────────────────────────────────────────────────────────────────

class Colour:
    # Backgrounds
    BG_PRIMARY       = "#0D1117"   # main screen background
    BG_SECONDARY     = "#161B22"   # card / panel background
    BG_HEADER        = "#0D1117"   # header bar background
    BG_INPUT         = "#1C2128"   # input field background

    # Accent colours
    ACCENT_BLUE      = "#2563EB"   # primary action, selected state
    ACCENT_BLUE_DIM  = "#1E3A8A"   # inactive blue

    # Rank colours
    RANK_1           = "#F59E0B"   # gold   — rank 1
    RANK_2           = "#94A3B8"   # silver — rank 2
    RANK_3           = "#78716C"   # bronze — rank 3
    RANK_1_BG        = "#1C1A0A"   # rank 1 card background tint
    RANK_2_BG        = "#161B22"   # rank 2 card background
    RANK_3_BG        = "#161B22"   # rank 3 card background

    # Fuel state colours
    FUEL_NORMAL      = "#16A34A"   # green
    FUEL_LOW         = "#D97706"   # amber
    FUEL_CRITICAL    = "#DC2626"   # red

    # Emergency type colour
    EMERGENCY        = "#7C3AED"   # purple accent for emergency badges

    # Confirm button
    CONFIRM          = "#16A34A"   # green
    CONFIRM_HOVER    = "#15803D"

    # Text colours
    TEXT_PRIMARY     = "#F0F6FC"   # near-white
    TEXT_SECONDARY   = "#8B949E"   # muted grey
    TEXT_MUTED       = "#484F58"   # very muted
    TEXT_CAUTION     = "#F59E0B"   # amber for caution/warning text

    # Border colours
    BORDER_DEFAULT   = "#30363D"
    BORDER_ACTIVE    = "#2563EB"
    BORDER_RANK_1    = "#F59E0B"

    # Back button
    BACK             = "#30363D"
    BACK_HOVER       = "#484F58"

    # Log screen
    LOG_ROW_ALT      = "#1C2128"

    # Divider
    DIVIDER          = "#21262D"


# ── Fonts ─────────────────────────────────────────────────────────────────────

class Font:
    FAMILY           = "Arial"
    FAMILY_MONO      = "Courier New"

    SIZE_TINY        = 10
    SIZE_SMALL       = 12
    SIZE_BODY        = 14
    SIZE_MEDIUM      = 16
    SIZE_LARGE       = 20
    SIZE_XLARGE      = 24
    SIZE_TITLE       = 28
    SIZE_HERO        = 36


# ── Spacing ───────────────────────────────────────────────────────────────────

class Spacing:
    XS               = 4
    SM               = 8
    MD               = 12
    LG               = 16
    XL               = 24
    XXL              = 32

    HEADER_HEIGHT    = 48
    TOUCH_MIN        = 70    # minimum touch target size in pixels
    BUTTON_HEIGHT    = 76    # standard action button height
    CARD_RADIUS      = 8
    BUTTON_RADIUS    = 6


# ── Stylesheet ────────────────────────────────────────────────────────────────

def get_stylesheet() -> str:
    """
    Returns the global Qt stylesheet for the application.
    Applied once at QApplication level.
    """
    return f"""

    /* ── Global ── */
    QWidget {{
        background-color: {Colour.BG_PRIMARY};
        color: {Colour.TEXT_PRIMARY};
        font-family: {Font.FAMILY};
        font-size: {Font.SIZE_BODY}px;
    }}

    /* ── Scroll areas ── */
    QScrollArea {{
        border: none;
        background-color: transparent;
    }}

    QScrollBar:vertical {{
        background: {Colour.BG_SECONDARY};
        width: 6px;
        border-radius: 3px;
    }}

    QScrollBar::handle:vertical {{
        background: {Colour.BORDER_DEFAULT};
        border-radius: 3px;
        min-height: 30px;
    }}

    QScrollBar::add-line:vertical,
    QScrollBar::sub-line:vertical {{
        height: 0px;
    }}

    QScrollBar:horizontal {{
        background: {Colour.BG_SECONDARY};
        height: 6px;
        border-radius: 3px;
    }}

    QScrollBar::handle:horizontal {{
        background: {Colour.BORDER_DEFAULT};
        border-radius: 3px;
        min-width: 30px;
    }}

    QScrollBar::add-line:horizontal,
    QScrollBar::sub-line:horizontal {{
        width: 0px;
    }}

    /* ── Labels ── */
    QLabel {{
        background: transparent;
        color: {Colour.TEXT_PRIMARY};
    }}

    /* ── Push buttons (base) ── */
    QPushButton {{
        background-color: {Colour.BG_SECONDARY};
        color: {Colour.TEXT_PRIMARY};
        border: 1px solid {Colour.BORDER_DEFAULT};
        border-radius: {Spacing.BUTTON_RADIUS}px;
        padding: 8px 16px;
        font-size: {Font.SIZE_BODY}px;
        font-family: {Font.FAMILY};
    }}

    QPushButton:pressed {{
        background-color: {Colour.BG_INPUT};
    }}

    /* ── Line edits ── */
    QLineEdit {{
        background-color: {Colour.BG_INPUT};
        color: {Colour.TEXT_PRIMARY};
        border: 1px solid {Colour.BORDER_DEFAULT};
        border-radius: {Spacing.BUTTON_RADIUS}px;
        padding: 8px 12px;
        font-size: {Font.SIZE_MEDIUM}px;
        font-family: {Font.FAMILY_MONO};
    }}

    QLineEdit:focus {{
        border: 1px solid {Colour.BORDER_ACTIVE};
    }}

    /* ── Frames ── */
    QFrame {{
        background: transparent;
    }}

    /* ── List widgets (logs screen) ── */
    QListWidget {{
        background-color: {Colour.BG_SECONDARY};
        border: 1px solid {Colour.BORDER_DEFAULT};
        border-radius: {Spacing.CARD_RADIUS}px;
        color: {Colour.TEXT_PRIMARY};
        font-size: {Font.SIZE_BODY}px;
        outline: none;
    }}

    QListWidget::item {{
        padding: 10px 12px;
        border-bottom: 1px solid {Colour.DIVIDER};
    }}

    QListWidget::item:selected {{
        background-color: {Colour.BG_INPUT};
        color: {Colour.TEXT_PRIMARY};
    }}

    QListWidget::item:alternate {{
        background-color: {Colour.LOG_ROW_ALT};
    }}

    /* ── Stacked widget ── */
    QStackedWidget {{
        background-color: {Colour.BG_PRIMARY};
    }}

    /* ── Dialog ── */
    QDialog {{
        background-color: {Colour.BG_SECONDARY};
        border: 1px solid {Colour.BORDER_DEFAULT};
        border-radius: {Spacing.CARD_RADIUS}px;
    }}

    """