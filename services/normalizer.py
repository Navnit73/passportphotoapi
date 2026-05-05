"""
Unit normalization helpers.

Converts all measurements (mm, inch, pixel) to percentage
of image height — the internal canonical unit.
"""


def mm_to_pct(value_mm: float, photo_height_mm: float) -> float:
    """Convert millimeters to percentage of photo height."""
    if photo_height_mm <= 0:
        raise ValueError(f"photo_height_mm must be positive, got {photo_height_mm}")
    return (value_mm / photo_height_mm) * 100.0


def inch_to_pct(value_in: float, photo_height_in: float) -> float:
    """Convert inches to percentage of photo height."""
    if photo_height_in <= 0:
        raise ValueError(f"photo_height_in must be positive, got {photo_height_in}")
    return (value_in / photo_height_in) * 100.0


def px_to_pct(value_px: int, photo_height_px: int) -> float:
    """Convert pixels to percentage of photo height."""
    if photo_height_px <= 0:
        raise ValueError(f"photo_height_px must be positive, got {photo_height_px}")
    return (value_px / photo_height_px) * 100.0


def pct_to_px(pct: float, photo_height_px: int) -> float:
    """Convert percentage of photo height to pixels."""
    return (pct / 100.0) * photo_height_px


def mm_to_px(value_mm: float, dpi: int = 300) -> float:
    """Convert millimeters to pixels at given DPI."""
    return (value_mm / 25.4) * dpi


def inch_to_px(value_in: float, dpi: int = 300) -> float:
    """Convert inches to pixels at given DPI."""
    return value_in * dpi


def compute_top_padding_pct(
    top_padding_mm: float,
    photo_height_mm: float,
) -> float:
    """
    Normalization formula from requirements:
    top_padding_pct = (top_padding_mm / photo_height_mm) * 100
    """
    return mm_to_pct(top_padding_mm, photo_height_mm)
