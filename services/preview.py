"""
Generate a preview image with dynamic measurement lines overlaid.
"""

import io
import logging

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config.countries.schema import CountryConfig

logger = logging.getLogger(__name__)


def create_preview_image(
    image: np.ndarray,
    config: CountryConfig,
    head_height_pct: float,
    eye_from_bottom_pct: float,
    top_margin_pct: float,
) -> bytes:
    """
    Generate a preview image with visual measurement lines.
    
    Args:
        image: BGR numpy array (the cropped output image)
        config: the country configuration
        head_height_pct: actual head height %
        eye_from_bottom_pct: actual eye from bottom %
        top_margin_pct: actual top margin %
        
    Returns:
        JPEG bytes of the preview image
    """
    h, w = image.shape[:2]

    # Calculate Y coordinates in the photo (0 is top)
    top_of_head_y = int(h * (top_margin_pct / 100))
    chin_y = int(h * ((top_margin_pct + head_height_pct) / 100))
    eye_y = int(h * (1.0 - (eye_from_bottom_pct / 100)))

    # Calculate physical values
    total_height_in = config.photo_spec.height_in
    total_width_in = config.photo_spec.width_in

    # Target constraints
    min_head_pct = config.face_constraints.head_height_pct.min
    max_head_pct = config.face_constraints.head_height_pct.max
    min_eye_pct = config.face_constraints.eye_position.from_bottom_pct.min
    max_eye_pct = config.face_constraints.eye_position.from_bottom_pct.max

    if total_height_in:
        # Head height text
        min_head_in = total_height_in * (min_head_pct / 100)
        max_head_in = total_height_in * (max_head_pct / 100)
        val_head_in = total_height_in * (head_height_pct / 100)
        head_text = f"{val_head_in:.2f}in ({min_head_in:.2f}-{max_head_in:.2f}in)"
        
        # Eye from bottom text
        min_eye_in = total_height_in * (min_eye_pct / 100)
        max_eye_in = total_height_in * (max_eye_pct / 100)
        val_eye_in = total_height_in * (eye_from_bottom_pct / 100)
        eye_text = f"{val_eye_in:.2f}in ({min_eye_in:.2f}-{max_eye_in:.2f}in)"
        
        total_height_text = f"{total_height_in}in"
        total_width_text = f"{total_width_in}in"
    else:
        # Fallback to mm
        total_height_mm = config.photo_spec.height_mm
        total_width_mm = config.photo_spec.width_mm
        
        min_head_mm = total_height_mm * (min_head_pct / 100)
        max_head_mm = total_height_mm * (max_head_pct / 100)
        val_head_mm = total_height_mm * (head_height_pct / 100)
        head_text = f"{val_head_mm:.1f}mm ({min_head_mm:.1f}-{max_head_mm:.1f}mm)"
        
        min_eye_mm = total_height_mm * (min_eye_pct / 100)
        max_eye_mm = total_height_mm * (max_eye_pct / 100)
        val_eye_mm = total_height_mm * (eye_from_bottom_pct / 100)
        eye_text = f"{val_eye_mm:.1f}mm ({min_eye_mm:.1f}-{max_eye_mm:.1f}mm)"

        total_height_text = f"{total_height_mm}mm"
        total_width_text = f"{total_width_mm}mm"

    # Create padded image
    pad_l, pad_r = 60, 250
    pad_t, pad_b = 60, 60

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    # Create white canvas
    new_w = w + pad_l + pad_r
    new_h = h + pad_t + pad_b
    canvas = Image.new("RGB", (new_w, new_h), "white")
    canvas.paste(pil_img, (pad_l, pad_t))

    draw = ImageDraw.Draw(canvas)
    
    try:
        # Try to use a standard sans-serif font
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        try:
            font = ImageFont.truetype("Helvetica", 20)
        except IOError:
            font = ImageFont.load_default()

    color = "#388E3C"  # Dark green
    line_w = 2

    # Draw photo boundary
    draw.rectangle([pad_l, pad_t, pad_l + w, pad_t + h], outline="#888888", width=1)

    def draw_dimension_vertical(x, y1, y2, text):
        draw.line([(x, y1), (x, y2)], fill=color, width=line_w)
        # arrows
        draw.polygon([(x, y1), (x - 5, y1 + 10), (x + 5, y1 + 10)], fill=color)
        draw.polygon([(x, y2), (x - 5, y2 - 10), (x + 5, y2 - 10)], fill=color)
        
        # Draw rotated text
        txt_img = Image.new('RGBA', (300, 30), (255, 255, 255, 0))
        txt_draw = ImageDraw.Draw(txt_img)
        txt_draw.text((0, 0), text, fill=color, font=font)
        txt_img = txt_img.rotate(90, expand=1)
        
        # Center vertically between y1 and y2
        paste_y = (y1 + y2) // 2 - txt_img.height // 2
        canvas.paste(txt_img, (x + 5, paste_y), txt_img)

    def draw_dimension_horizontal(y, x1, x2, text):
        draw.line([(x1, y), (x2, y)], fill=color, width=line_w)
        # arrows
        draw.polygon([(x1, y), (x1 + 10, y - 5), (x1 + 10, y + 5)], fill=color)
        draw.polygon([(x2, y), (x2 - 10, y - 5), (x2 - 10, y + 5)], fill=color)
        
        try:
            bbox = font.getbbox(text)
            tw = bbox[2] - bbox[0]
        except AttributeError:
            tw = 100
            
        draw.text(((x1 + x2) // 2 - tw // 2, y + 5), text, fill=color, font=font)

    def draw_feature_line(photo_y, right_ext, x_end=None):
        """
        Draw a horizontal feature line.
        - x_end: if provided, line stops at this absolute canvas X coordinate.
                 Otherwise falls back to pad_l + w + right_ext.
        """
        y = pad_t + photo_y
        x_stop = x_end if x_end is not None else (pad_l + w + right_ext)
        draw.line([(pad_l, y), (x_stop, y)], fill=color, width=1)

    # Midpoint of the photo horizontally (matches the red reference line position)
    half_x = pad_l + (w // 2)

    # 1. Total Height (Left side)
    draw_dimension_vertical(pad_l - 20, pad_t, pad_t + h, total_height_text)
    
    # 2. Total Width (Bottom side)
    draw_dimension_horizontal(pad_t + h + 20, pad_l, pad_l + w, total_width_text)

    # 3. Feature mapping lines:
    #    - Top of head: full width (needed so dimension bracket on right can anchor to it)
    #    - Chin: left edge → center only (half line)
    #    - Eye:  left edge → center only (half line)
    draw_feature_line(top_of_head_y, 40)                          # full width + right extension
    draw_feature_line(chin_y, 0, x_end=half_x)                   # left → center only
    draw_feature_line(eye_y,  0, x_end=half_x)                   # left → center only

    # 4. Head Height Dimension (Top of head to chin) — right side bracket
    draw_dimension_vertical(pad_l + w + 20, pad_t + top_of_head_y, pad_t + chin_y, head_text)

    # 5. Eye from Bottom Dimension (Bottom of photo to eyes) — right side bracket
    draw_dimension_vertical(pad_l + w + 60, pad_t + eye_y, pad_t + h, eye_text)

    # Convert canvas to JPEG bytes
    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=90)
    return buf.getvalue()