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
    """
    h, w = image.shape[:2]

    # Calculate Y coordinates in the photo (0 is top)
    top_of_head_y = int(h * (top_margin_pct / 100))
    chin_y = int(h * ((top_margin_pct + head_height_pct) / 100))
    eye_y = int(h * (1.0 - (eye_from_bottom_pct / 100)))

    # Calculate physical values
    total_height_in = config.photo_spec.height_in
    total_width_in = config.photo_spec.width_in

    min_head_pct = config.face_constraints.head_height_pct.min
    max_head_pct = config.face_constraints.head_height_pct.max
    min_eye_pct = config.face_constraints.eye_position.from_bottom_pct.min
    max_eye_pct = config.face_constraints.eye_position.from_bottom_pct.max

    if total_height_in:
        min_head_in = total_height_in * (min_head_pct / 100)
        max_head_in = total_height_in * (max_head_pct / 100)
        val_head_in = total_height_in * (head_height_pct / 100)
        head_text = f"{val_head_in:.2f}in ({min_head_in:.2f}-{max_head_in:.2f}in)"

        min_eye_in = total_height_in * (min_eye_pct / 100)
        max_eye_in = total_height_in * (max_eye_pct / 100)
        val_eye_in = total_height_in * (eye_from_bottom_pct / 100)
        eye_text = f"{val_eye_in:.2f}in ({min_eye_in:.2f}-{max_eye_in:.2f}in)"

        total_height_text = f"{total_height_in}in"
        total_width_text = f"{total_width_in}in"
    else:
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

    # --- Padding ---
    pad_l = 80    # left: space for vertical height bracket
    pad_r = 280   # right: space for head-height + eye brackets + labels
    pad_t = 60
    pad_b = 60

    # --- Canvas size ---
    new_w = w + pad_l + pad_r
    new_h = h + pad_t + pad_b

    # --- Photo offset: centered horizontally within the full canvas ---
    photo_x = (new_w - w) // 2   # <-- this centers the photo
    photo_y = pad_t               # top padding stays fixed

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    canvas = Image.new("RGB", (new_w, new_h), "white")
    canvas.paste(pil_img, (photo_x, photo_y))  # <-- centered paste

    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        try:
            font = ImageFont.truetype("Helvetica", 20)
        except IOError:
            font = ImageFont.load_default()

    color = "#388E3C"
    line_w = 2

    # Photo boundary box (now uses photo_x instead of pad_l)
    draw.rectangle(
        [photo_x, photo_y, photo_x + w, photo_y + h],
        outline="#888888", width=1
    )

    def draw_dimension_vertical(x, y1, y2, text):
        draw.line([(x, y1), (x, y2)], fill=color, width=line_w)
        draw.polygon([(x, y1), (x - 5, y1 + 10), (x + 5, y1 + 10)], fill=color)
        draw.polygon([(x, y2), (x - 5, y2 - 10), (x + 5, y2 - 10)], fill=color)

        txt_img = Image.new('RGBA', (300, 30), (255, 255, 255, 0))
        txt_draw = ImageDraw.Draw(txt_img)
        txt_draw.text((0, 0), text, fill=color, font=font)
        txt_img = txt_img.rotate(90, expand=True)

        paste_y = (y1 + y2) // 2 - txt_img.height // 2
        canvas.paste(txt_img, (x + 5, paste_y), txt_img)

    def draw_dimension_horizontal(y, x1, x2, text):
        draw.line([(x1, y), (x2, y)], fill=color, width=line_w)
        draw.polygon([(x1, y), (x1 + 10, y - 5), (x1 + 10, y + 5)], fill=color)
        draw.polygon([(x2, y), (x2 - 10, y - 5), (x2 - 10, y + 5)], fill=color)

        try:
            bbox = font.getbbox(text)
            tw = bbox[2] - bbox[0]
        except AttributeError:
            tw = 100

        draw.text(((x1 + x2) // 2 - tw // 2, y + 5), text, fill=color, font=font)

    def draw_feature_line(photo_rel_y, right_ext, x_end=None):
        """Draw a horizontal feature line relative to photo_y."""
        y = photo_y + photo_rel_y
        x_stop = x_end if x_end is not None else (photo_x + w + right_ext)
        draw.line([(photo_x, y), (x_stop, y)], fill=color, width=1)

    # Center x of the photo (for half-width lines)
    half_x = photo_x + (w // 2)

    # 1. Total Height — left side bracket
    draw_dimension_vertical(photo_x - 20, photo_y, photo_y + h, total_height_text)

    # 2. Total Width — bottom bracket
    draw_dimension_horizontal(photo_y + h + 20, photo_x, photo_x + w, total_width_text)

    # 3. Feature lines
    draw_feature_line(top_of_head_y, 40)               # full width + right ext
    draw_feature_line(chin_y, 0, x_end=half_x)         # left → center only
    draw_feature_line(eye_y,  0, x_end=half_x)         # left → center only

    # 4. Head Height bracket — right side
    draw_dimension_vertical(
        photo_x + w + 20,
        photo_y + top_of_head_y,
        photo_y + chin_y,
        head_text
    )

    # 5. Eye from Bottom bracket — right side
    draw_dimension_vertical(
        photo_x + w + 60,
        photo_y + eye_y,
        photo_y + h,
        eye_text
    )

    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=90)
    return buf.getvalue()