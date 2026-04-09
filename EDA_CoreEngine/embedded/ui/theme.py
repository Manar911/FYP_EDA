"""
theme.py  —  EDA 

Updated palette:
    BG_BASE    #0E1A24  — main background
    BG_CARD    #1C2E3E  — side panels, airport info cards
    AMBER      #FFBF00  — caution
    GREEN      #00FF66  — confirmed / safe
    RED        #FF3B30  — danger
    CYAN       #00C8E8  — active / advisory (unchanged)
"""

from __future__ import annotations


class Colour:
    # ── Backgrounds ───────────────────────────────────────────
    BG_BASE        = "#0E1A24"
    BG_CARD        = "#1C2E3E"
    BG_HEADER      = "#091420"
    BG_INPUT       = "#152232"
    BG_OVERLAY     = "#0E1A24"

    # ── EFIS cyan ─────────────────────────────────────────────
    CYAN           = "#00C8E8"
    CYAN_DIM       = "#006070"
    CYAN_BG        = "#001E28"

    # ── Caution amber ─────────────────────────────────────────
    AMBER          = "#FFBF00"
    AMBER_DIM      = "#7A5A00"
    AMBER_BG       = "#1A1200"

    # ── Danger red ────────────────────────────────────────────
    RED            = "#FF3B30"
    RED_DIM        = "#7A1A14"
    RED_BG         = "#1A0806"

    # ── Confirmed green ───────────────────────────────────────
    GREEN          = "#00FF66"
    GREEN_DIM      = "#007A30"
    GREEN_BG       = "#001A0C"

    # ── Text ──────────────────────────────────────────────────
    TEXT_PRIMARY   = "#D8EAF5"
    TEXT_SECONDARY = "#5A7A90"
    TEXT_MUTED     = "#1E3040"
    TEXT_CAUTION   = AMBER

    # ── Borders ───────────────────────────────────────────────
    BORDER         = "#1E3040"
    BORDER_ACTIVE  = CYAN
    BORDER_CAUTION = AMBER
    BORDER_DANGER  = RED

    # ── Buttons ───────────────────────────────────────────────
    BTN_BG         = "#111E2A"
    BTN_PRESSED    = "#1A2E3E"

    # ── Fuel states ───────────────────────────────────────────
    FUEL_NORMAL    = GREEN
    FUEL_LOW       = AMBER
    FUEL_CRITICAL  = RED

    # ── Rank ──────────────────────────────────────────────────
    RANK_1_COLOUR  = CYAN
    RANK_1_BG      = "#001C28"
    RANK_2_COLOUR  = TEXT_PRIMARY
    RANK_2_BG      = BG_CARD
    RANK_3_COLOUR  = TEXT_SECONDARY
    RANK_3_BG      = BG_CARD

    # ── Divider ───────────────────────────────────────────────
    DIVIDER        = "#152030"

    # ── Unified selected button style ─────────────────────────
    # All selected buttons use these — no per-button colour variation
    SEL_BG         = CYAN_BG
    SEL_BORDER     = CYAN
    SEL_TEXT       = CYAN


class Font:
    PRIMARY  = "Roboto"
    MONO     = "Roboto Mono"
    FALLBACK = "Arial"

    SZ_XS    = 11
    SZ_SM    = 13
    SZ_BODY  = 15
    SZ_MD    = 17
    SZ_LG    = 20
    SZ_XL    = 24
    SZ_2XL   = 30
    SZ_3XL   = 38


class Spacing:
    XS   = 4
    SM   = 8
    MD   = 12
    LG   = 16
    XL   = 24

    HEADER_H   = 52
    TOUCH_MIN  = 72
    BTN_H      = 78
    RADIUS     = 4
    RADIUS_SM  = 3


def get_stylesheet() -> str:
    return f"""
    QWidget {{
        background-color: {Colour.BG_BASE};
        color: {Colour.TEXT_PRIMARY};
        font-family: "{Font.PRIMARY}", "{Font.FALLBACK}";
        font-size: {Font.SZ_BODY}px;
        border: none;
        outline: none;
    }}
    QScrollArea {{ border: none; background: transparent; }}
    QScrollBar:vertical {{
        background: {Colour.BG_CARD};
        width: 8px;
        border-radius: 4px;
    }}
    QScrollBar::handle:vertical {{
        background: {Colour.CYAN_DIM};
        border-radius: 4px;
        min-height: 32px;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
    QScrollBar:horizontal {{
        background: {Colour.BG_CARD};
        height: 8px;
        border-radius: 4px;
    }}
    QScrollBar::handle:horizontal {{
        background: {Colour.CYAN_DIM};
        border-radius: 4px;
        min-width: 32px;
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
    QLabel {{ background: transparent; color: {Colour.TEXT_PRIMARY}; }}
    QPushButton {{
        background-color: {Colour.BTN_BG};
        color: {Colour.TEXT_SECONDARY};
        border: 1px solid {Colour.BORDER};
        border-radius: {Spacing.RADIUS_SM}px;
        padding: 8px 14px;
        font-size: {Font.SZ_BODY}px;
        font-family: "{Font.PRIMARY}", "{Font.FALLBACK}";
    }}
    QPushButton:pressed {{
        background-color: {Colour.BTN_PRESSED};
        color: {Colour.TEXT_PRIMARY};
    }}
    QPushButton:disabled {{
        color: {Colour.TEXT_MUTED};
        border-color: {Colour.TEXT_MUTED};
        background-color: {Colour.BG_BASE};
    }}
    QLineEdit {{
        background-color: {Colour.BG_INPUT};
        color: {Colour.CYAN};
        border: 1px solid {Colour.BORDER};
        border-radius: {Spacing.RADIUS_SM}px;
        padding: 8px 12px;
        font-size: {Font.SZ_MD}px;
        font-family: "{Font.MONO}", "Courier New";
    }}
    QLineEdit:focus {{ border-color: {Colour.CYAN}; }}
    QFrame {{ background: transparent; }}
    QDialog {{
        background-color: {Colour.BG_CARD};
        border: 1px solid {Colour.BORDER_ACTIVE};
    }}
    QListWidget {{
        background-color: {Colour.BG_CARD};
        border: 1px solid {Colour.BORDER};
        border-radius: {Spacing.RADIUS}px;
        color: {Colour.TEXT_PRIMARY};
        font-size: {Font.SZ_SM}px;
        font-family: "{Font.MONO}", "Courier New";
        outline: none;
    }}
    QListWidget::item {{
        padding: 10px 14px;
        border-bottom: 1px solid {Colour.DIVIDER};
        color: {Colour.TEXT_SECONDARY};
    }}
    QListWidget::item:alternate {{ background-color: {Colour.BG_INPUT}; }}
    QStackedWidget {{ background-color: {Colour.BG_BASE}; }}
    QMessageBox {{
        background-color: {Colour.BG_CARD};
        color: {Colour.TEXT_PRIMARY};
    }}
    QMessageBox QLabel {{
        color: {Colour.TEXT_PRIMARY};
        font-size: {Font.SZ_BODY}px;
    }}
    QMessageBox QPushButton {{ min-width: 80px; padding: 8px 18px; }}
    """