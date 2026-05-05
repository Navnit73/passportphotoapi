"""
JPEG compression with binary-search to meet file size limits.

Uses Pillow's JPEG encoder with subsampling=0 (4:4:4) for
maximum quality. Strips EXIF, sets sRGB color profile,
embeds DPI metadata.
"""

import io
import logging

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def compress_to_jpeg(
    image: np.ndarray,
    max_size_kb: int = 240,
    target_dpi: int = 300,
    initial_quality: int = 95,
) -> bytes:
    """
    Compress an image to JPEG meeting a file size constraint.

    Uses binary search to find the highest quality that fits
    within max_size_kb.

    Args:
        image: BGR numpy array
        max_size_kb: maximum file size in KB (e.g. 240 for DS-160)
        target_dpi: DPI to embed in metadata
        initial_quality: starting JPEG quality

    Returns:
        JPEG bytes

    Raises:
        ValueError: if image cannot be compressed to meet the limit
    """
    # Use 90% of the allowed limit as our internal target to ensure we stay safely below it
    # while maintaining high quality. (e.g., if limit is 250KB, we target ~225KB)
    target_max_kb = int(max_size_kb * 0.9)
    max_size_bytes = target_max_kb * 1024

    # Convert BGR to RGB for Pillow
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)

    # First try at maximum quality (100)
    buf = io.BytesIO()
    pil_image.save(
        buf,
        format="JPEG",
        quality=100,
        subsampling=0,  # 4:4:4 — no chroma subsampling
        dpi=(target_dpi, target_dpi),
        icc_profile=None,
        exif=b"",
    )

    if buf.tell() <= max_size_bytes:
        logger.info(
            f"JPEG at quality=100: "
            f"{buf.tell() // 1024}KB (safely within {max_size_kb}KB limit)"
        )
        return buf.getvalue()

    # Binary search for optimal quality if q=100 is too large
    logger.info(
        f"JPEG at q=100 is {buf.tell() // 1024}KB, "
        f"exceeds target {target_max_kb}KB — starting binary search"
    )

    low, high = 1, 100
    best_bytes = None
    best_quality = -1

    while low <= high:
        mid = (low + high) // 2

        buf = io.BytesIO()
        pil_image.save(
            buf,
            format="JPEG",
            quality=mid,
            subsampling=0,
            dpi=(target_dpi, target_dpi),
            icc_profile=None,
            exif=b"",
        )

        size = buf.tell()

        if size <= max_size_bytes:
            best_bytes = buf.getvalue()
            best_quality = mid
            low = mid + 1  # try higher quality
        else:
            high = mid - 1  # try lower quality

    if best_bytes is None:
        raise ValueError(
            f"Image cannot be compressed to {max_size_kb}KB. "
            f"Even at quality=1 the file is too large."
        )

    logger.info(
        f"Compressed to quality={best_quality}: "
        f"{len(best_bytes) // 1024}KB"
    )
    return best_bytes


def encode_jpeg(
    image: np.ndarray,
    quality: int = 95,
    target_dpi: int = 300,
) -> bytes:
    """
    Simple JPEG encode without size constraint.

    Args:
        image: BGR numpy array
        quality: JPEG quality (1-100)
        target_dpi: DPI to embed

    Returns:
        JPEG bytes
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)

    buf = io.BytesIO()
    pil_image.save(
        buf,
        format="JPEG",
        quality=quality,
        subsampling=0,
        dpi=(target_dpi, target_dpi),
        icc_profile=None,
        exif=b"",
    )

    return buf.getvalue()


def create_print_sheet(
    photo_bytes: bytes,
    target_w: int,
    target_h: int,
    paper_size: str = "4x6_in",
    photos_per_sheet: int = 6,
    spacing_mm: float = 5,
    dpi: int = 300,
) -> bytes:
    """
    Create a print sheet with multiple copies of the photo.

    Args:
        photo_bytes: JPEG bytes of the processed photo
        target_w: photo width in pixels
        target_h: photo height in pixels
        paper_size: paper size string
        photos_per_sheet: target number of photos
        spacing_mm: gap between photos in mm
        dpi: print DPI

    Returns:
        JPEG bytes of the print sheet
    """
    # Paper dimensions in pixels at given DPI
    paper_sizes = {
        "4x6_in": (int(4 * dpi), int(6 * dpi)),      # 1200x1800
        "A4": (int(210 / 25.4 * dpi), int(297 / 25.4 * dpi)),  # 2480x3508
        "Letter": (int(8.5 * dpi), int(11 * dpi)),    # 2550x3300
    }

    sheet_w, sheet_h = paper_sizes.get(paper_size, paper_sizes["4x6_in"])
    spacing_px = int((spacing_mm / 25.4) * dpi)

    # Load the photo
    photo = Image.open(io.BytesIO(photo_bytes))
    from PIL import ImageDraw

    # Calculate grid dynamically
    cols = (sheet_w - 40) // (target_w + spacing_px)
    rows = (sheet_h - 40) // (target_h + spacing_px)
    margin_x = (sheet_w - (cols * target_w + (cols - 1) * spacing_px)) // 2
    margin_y = (sheet_h - (rows * target_h + (rows - 1) * spacing_px)) // 2

    # Create white sheet
    sheet = Image.new("RGB", (sheet_w, sheet_h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)

    def draw_dashed_rect(x0, y0, x1, y1):
        dash_len = 20
        space_len = 15
        color = "#bbbbbb"
        width = 2
        # top
        for x in range(int(x0), int(x1), dash_len + space_len):
            draw.line([(x, y0), (min(x + dash_len, x1), y0)], fill=color, width=width)
        # bottom
        for x in range(int(x0), int(x1), dash_len + space_len):
            draw.line([(x, y1), (min(x + dash_len, x1), y1)], fill=color, width=width)
        # left
        for y in range(int(y0), int(y1), dash_len + space_len):
            draw.line([(x0, y), (x0, min(y + dash_len, y1))], fill=color, width=width)
        # right
        for y in range(int(y0), int(y1), dash_len + space_len):
            draw.line([(x1, y), (x1, min(y + dash_len, y1))], fill=color, width=width)

    # Paste photos in grid and add dashed cutting guides
    for r in range(rows):
        for c in range(cols):
            x = margin_x + c * (target_w + spacing_px)
            y = margin_y + r * (target_h + spacing_px)
            
            # Paste the photo without any black border
            sheet.paste(photo, (x, y))
            
            # Draw a dashed cutting line right around the photo's edge
            draw_dashed_rect(x, y, x + target_w, y + target_h)

    # Encode
    buf = io.BytesIO()
    sheet.save(buf, format="JPEG", quality=95, dpi=(dpi, dpi))
    return buf.getvalue()
