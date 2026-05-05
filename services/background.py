"""
Background validation and correction pipeline.

Validation:
  1. Sample edge pixels (corners + border midpoints)
  2. Compute average RGB
  3. Compare with target color
  4. Check RGB deviation, brightness, variance

Correction (if invalid):
  Replace background using rembg with u2netp model.
"""

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ─── Singleton rembg session ───
_rembg_session = None

def _get_rembg_session():
    """Get or create the rembg session (singleton)."""
    global _rembg_session
    if _rembg_session is None:
        logger.info("Initializing rembg session with u2net_human_seg model...")
        from rembg import new_session
        _rembg_session = new_session("u2net_human_seg")
    return _rembg_session

# ─── Background color map ───
BG_COLORS = {
    "white":      (255, 255, 255),
    "plain white": (255, 255, 255),
    "light-gray": (243, 244, 246),
    "light-blue": (239, 246, 255),
    "blue":       (0, 71, 171),
    "red":        (200, 30, 30),  # Standard passport red
}


@dataclass
class BackgroundValidationResult:
    """Result of background validation."""
    is_valid: bool
    avg_rgb: tuple[float, float, float]
    brightness: float
    rgb_deviation: float
    variance: float
    message: str


def _parse_bg_color(color_str: str) -> tuple[int, int, int]:
    """Convert background color string to RGB tuple."""
    color_str = color_str.lower().strip()

    # 1. Smart mapping for descriptive strings
    # Use spaces or exact matches to avoid accidental matches like "colored" -> "red"
    words = color_str.split()
    
    if "red" in words:
        return BG_COLORS["red"]
    
    if "gray" in color_str or "grey" in color_str:
        return BG_COLORS["light-gray"]
    
    if "blue" in color_str:
        if "light" in color_str:
            return BG_COLORS["light-blue"]
        return BG_COLORS["blue"]
    
    if "white" in color_str or "off-white" in color_str or "cream" in color_str or "light colored" in color_str:
        return BG_COLORS["white"]

    # 2. Check direct mapping
    if color_str in BG_COLORS:
        return BG_COLORS[color_str]

    # Check hex color
    if color_str.startswith("#") and len(color_str) == 7:
        r = int(color_str[1:3], 16)
        g = int(color_str[3:5], 16)
        b = int(color_str[5:7], 16)
        return (r, g, b)

    # Default to white
    logger.warning(f"Unknown background color '{color_str}', defaulting to white")
    return (255, 255, 255)


def _sample_edge_pixels(image: np.ndarray) -> np.ndarray:
    """
    Sample pixels from image edges for background analysis.

    Samples from:
    - 4 corners (3x3 blocks)
    - 4 border midpoints (3x3 blocks)
    - Additional border strip samples

    Returns an Nx3 array of RGB values.
    """
    h, w = image.shape[:2]
    samples = []

    # Corner regions (top-left, top-right, bottom-left, bottom-right)
    margin = min(10, h // 10, w // 10)
    corners = [
        (0, 0),                    # top-left
        (0, w - margin),           # top-right
        (h - margin, 0),           # bottom-left
        (h - margin, w - margin),  # bottom-right
    ]

    for cy, cx in corners:
        block = image[cy:cy + margin, cx:cx + margin]
        samples.append(block.reshape(-1, 3))

    # Border midpoints
    mid_h, mid_w = h // 2, w // 2
    midpoints = [
        (0, mid_w - margin // 2),              # top-center
        (h - margin, mid_w - margin // 2),     # bottom-center
        (mid_h - margin // 2, 0),              # left-center
        (mid_h - margin // 2, w - margin),     # right-center
    ]

    for cy, cx in midpoints:
        cy = max(0, min(cy, h - margin))
        cx = max(0, min(cx, w - margin))
        block = image[cy:cy + margin, cx:cx + margin]
        samples.append(block.reshape(-1, 3))

    # Top strip (first 5 rows)
    top_strip = image[0:min(5, h), :]
    samples.append(top_strip.reshape(-1, 3))

    return np.vstack(samples)


def validate_background(
    image: np.ndarray,
    target_color: str = "plain white",
    rgb_tolerance: int = 10,
    brightness_min: int = 240,
    max_variance: float = 15.0,
) -> BackgroundValidationResult:
    """
    Validate if the image background matches the target color.

    Args:
        image: BGR numpy array
        target_color: target background color string
        rgb_tolerance: max allowed RGB deviation per channel
        brightness_min: minimum average brightness of edge pixels
        max_variance: max allowed variance across samples (texture check)

    Returns:
        BackgroundValidationResult with is_valid flag and diagnostics
    """
    # Convert BGR to RGB for analysis
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Sample edge pixels
    samples = _sample_edge_pixels(rgb_image)

    # Compute average RGB
    avg_rgb = samples.mean(axis=0)
    avg_r, avg_g, avg_b = float(avg_rgb[0]), float(avg_rgb[1]), float(avg_rgb[2])

    # Compute brightness (average of RGB channels)
    brightness = (avg_r + avg_g + avg_b) / 3.0

    # Compute variance (texture detection)
    variance = float(samples.std())

    # Parse target color
    target_rgb = _parse_bg_color(target_color)

    # Compute per-channel deviation from target
    deviations = [
        abs(avg_r - target_rgb[0]),
        abs(avg_g - target_rgb[1]),
        abs(avg_b - target_rgb[2]),
    ]
    max_deviation = max(deviations)

    # ─── Validate ───
    issues = []

    if max_deviation > rgb_tolerance:
        issues.append(
            f"RGB deviation {max_deviation:.1f} exceeds tolerance {rgb_tolerance}"
        )

    if brightness < brightness_min:
        issues.append(
            f"Brightness {brightness:.0f} below minimum {brightness_min}"
        )

    if variance > max_variance:
        issues.append(
            f"Variance {variance:.1f} exceeds max {max_variance} (possible texture)"
        )

    is_valid = len(issues) == 0
    message = "Background valid" if is_valid else "; ".join(issues)

    logger.info(
        f"Background validation: valid={is_valid}, "
        f"avg_rgb=({avg_r:.0f},{avg_g:.0f},{avg_b:.0f}), "
        f"brightness={brightness:.0f}, deviation={max_deviation:.1f}, "
        f"variance={variance:.1f}"
    )

    return BackgroundValidationResult(
        is_valid=is_valid,
        avg_rgb=(avg_r, avg_g, avg_b),
        brightness=brightness,
        rgb_deviation=max_deviation,
        variance=variance,
        message=message,
    )


def correct_background_rembg(
    image: np.ndarray,
    target_color: str = "plain white",
) -> np.ndarray:
    """
    Replace background using rembg with u2netp model.

    Args:
        image: BGR numpy array
        target_color: target background color string

    Returns:
        BGR numpy array with background replaced
    """
    logger.info("Using rembg for background removal")

    try:
        from rembg import remove

        # Use shared singleton session to avoid reloading model weights
        session = _get_rembg_session()

        # rembg expects RGB input and returns RGBA
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        from PIL import Image
        pil_image = Image.fromarray(rgb_image)

        # Remove background → RGBA
        # Use alpha matting and post-processing for sharper hair details and cleaner edges
        result_rgba = remove(
            pil_image, 
            session=session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=10,
            post_process_mask=True
        )

        # Create target background
        target_rgb = _parse_bg_color(target_color)
        bg = Image.new("RGBA", result_rgba.size, (*target_rgb, 255))

        # Composite person over background
        bg.paste(result_rgba, mask=result_rgba.split()[3])

        # Convert back to BGR numpy
        result_rgb = np.array(bg.convert("RGB"))
        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

        logger.info(f"Background replaced with {target_color} via rembg (u2netp)")
        return result_bgr

    except ImportError:
        logger.error("rembg not installed — cannot use fallback")
        raise ValueError("Background correction failed: rembg not available")
    except Exception as e:
        logger.error(f"rembg fallback failed: {e}")
        raise ValueError(f"Background correction failed: {e}")


def correct_background(
    image: np.ndarray,
    target_color: str = "plain white",
    rgb_tolerance: int = 10,
    brightness_min: int = 240,
) -> tuple[np.ndarray, bool]:
    """
    Background correction pipeline using rembg.

    Args:
        image: BGR numpy array
        target_color: target background color
        rgb_tolerance: validation tolerance (unused but kept for compatibility)
        brightness_min: minimum brightness (unused but kept for compatibility)

    Returns:
        (corrected_image, success) tuple
    """
    try:
        corrected = correct_background_rembg(image, target_color)
        logger.info("Background correction succeeded (rembg)")
        return corrected, True
    except Exception as e:
        logger.error(f"Background correction failed: {e}")
        return image, False
