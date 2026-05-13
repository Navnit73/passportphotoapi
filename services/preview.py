"""
Generate a preview image with dynamic measurement lines overlaid.
Improved UI: more left padding, cleaner arrows, pill backgrounds on labels.
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
    chin_y        = int(h * ((top_margin_pct + head_height_pct) / 100))
    eye_y         = int(h * (1.0 - (eye_from_bottom_pct / 100)))

    # ── Physical measurement text ────────────────────────────────────────────
    total_height_in = config.photo_spec.height_in
    total_width_in  = config.photo_spec.width_in

    min_head_pct = config.face_constraints.head_height_pct.min
    max_head_pct = config.face_constraints.head_height_pct.max
    min_eye_pct  = config.face_constraints.eye_position.from_bottom_pct.min
    max_eye_pct  = config.face_constraints.eye_position.from_bottom_pct.max

    if total_height_in:
        min_head_in = total_height_in * (min_head_pct / 100)
        max_head_in = total_height_in * (max_head_pct / 100)
        val_head_in = total_height_in * (head_height_pct / 100)
        head_text   = f"{val_head_in:.2f}in ({min_head_in:.2f}-{max_head_in:.2f}in)"

        min_eye_in = total_height_in * (min_eye_pct / 100)
        max_eye_in = total_height_in * (max_eye_pct / 100)
        val_eye_in = total_height_in * (eye_from_bottom_pct / 100)
        eye_text   = f"{val_eye_in:.2f}in ({min_eye_in:.2f}-{max_eye_in:.2f}in)"

        total_height_text = f"{total_height_in}in"
        total_width_text  = f"{total_width_in}in"
    else:
        total_height_mm = config.photo_spec.height_mm or 0.0
        total_width_mm  = config.photo_spec.width_mm  or 0.0

        min_head_mm = total_height_mm * (min_head_pct / 100)
        max_head_mm = total_height_mm * (max_head_pct / 100)
        val_head_mm = total_height_mm * (head_height_pct / 100)
        head_text   = f"{val_head_mm:.1f}mm ({min_head_mm:.1f}-{max_head_mm:.1f}mm)"

        min_eye_mm = total_height_mm * (min_eye_pct / 100)
        max_eye_mm = total_height_mm * (max_eye_pct / 100)
        val_eye_mm = total_height_mm * (eye_from_bottom_pct / 100)
        eye_text   = f"{val_eye_mm:.1f}mm ({min_eye_mm:.1f}-{max_eye_mm:.1f}mm)"

        total_height_text = f"{total_height_mm}mm"
        total_width_text  = f"{total_width_mm}mm"

    # ── Canvas padding ───────────────────────────────────────────────────────
    # pad_l increased so the left bracket + rotated label never touch the photo
    pad_l = 130   # ← was 80; extra room for bracket line + text + gap
    pad_r = 300   # right: two brackets + labels
    pad_t = 70
    pad_b = 70

    new_w   = w + pad_l + pad_r
    new_h   = h + pad_t + pad_b
    photo_x = (new_w - w) // 2   # horizontally centred
    photo_y = pad_t

    # ── Build canvas ─────────────────────────────────────────────────────────
    rgb     = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    canvas  = Image.new("RGB", (new_w, new_h), "#F8F9FA")   # off-white bg
    canvas.paste(pil_img, (photo_x, photo_y))
    draw = ImageDraw.Draw(canvas)

    # ── Fonts ────────────────────────────────────────────────────────────────
    font_size = 22
    try:
        font      = ImageFont.truetype("arial.ttf",   font_size)
        font_bold = ImageFont.truetype("arialbd.ttf", font_size)
    except IOError:
        try:
            font      = ImageFont.truetype("Helvetica", font_size)
            font_bold = font
        except IOError:
            font      = ImageFont.load_default()
            font_bold = font

    # ── Design tokens ────────────────────────────────────────────────────────
    COLOR_LINE    = "#2E7D32"   # dark green
    COLOR_FEAT    = "#43A047"   # mid green for feature lines
    COLOR_ARROW   = "#2E7D32"
    COLOR_TEXT    = "#1B5E20"   # very dark green text
    COLOR_PILL_BG = "#E8F5E9"   # light green pill
    COLOR_PILL_BD = "#A5D6A7"   # pill border
    LINE_W        = 1
    ARROW_SIZE    = 8           # half-width of arrow head
    ARROW_LEN     = 14          # length of arrow head

    # ── Photo boundary ───────────────────────────────────────────────────────
    # Thin rounded-corner feel via 1px outline + subtle shadow row
    draw.rectangle(
        [photo_x - 1, photo_y - 1, photo_x + w, photo_y + h],
        outline="#BDBDBD", width=1
    )
    draw.rectangle(
        [photo_x, photo_y, photo_x + w - 1, photo_y + h - 1],
        outline="#E0E0E0", width=1
    )

    # ── Helper: text size ────────────────────────────────────────────────────
    def text_w(text, f=None):
        f = f or font
        try:
            bb = f.getbbox(text)
            return bb[2] - bb[0]
        except AttributeError:
            return len(text) * 12

    def text_h(text, f=None):
        f = f or font
        try:
            bb = f.getbbox(text)
            return bb[3] - bb[1]
        except AttributeError:
            return font_size

    # ── Helper: draw arrow tip ───────────────────────────────────────────────
    def arrow_tip_up(draw, cx, y):
        draw.polygon(
            [(cx, y), (cx - ARROW_SIZE, y + ARROW_LEN), (cx + ARROW_SIZE, y + ARROW_LEN)],
            fill=COLOR_ARROW
        )

    def arrow_tip_down(draw, cx, y):
        draw.polygon(
            [(cx, y), (cx - ARROW_SIZE, y - ARROW_LEN), (cx + ARROW_SIZE, y - ARROW_LEN)],
            fill=COLOR_ARROW
        )

    def arrow_tip_left(draw, x, cy):
        draw.polygon(
            [(x, cy), (x + ARROW_LEN, cy - ARROW_SIZE), (x + ARROW_LEN, cy + ARROW_SIZE)],
            fill=COLOR_ARROW
        )

    def arrow_tip_right(draw, x, cy):
        draw.polygon(
            [(x, cy), (x - ARROW_LEN, cy - ARROW_SIZE), (x - ARROW_LEN, cy + ARROW_SIZE)],
            fill=COLOR_ARROW
        )

    # ── Helper: pill label (horizontal) ─────────────────────────────────────
    def draw_pill_label(cx, cy, text, f=None):
        """Draw text centred at (cx, cy) on a rounded pill background."""
        f   = f or font
        tw  = text_w(text, f)
        th  = text_h(text, f)
        pad = 8
        rx0, ry0 = cx - tw // 2 - pad, cy - th // 2 - pad // 2
        rx1, ry1 = cx + tw // 2 + pad, cy + th // 2 + pad // 2
        draw.rounded_rectangle([rx0, ry0, rx1, ry1],
                                radius=6, fill=COLOR_PILL_BG, outline=COLOR_PILL_BD, width=1)
        draw.text((cx - tw // 2, cy - th // 2), text, fill=COLOR_TEXT, font=f)

    # ── Helper: vertical dimension bracket ──────────────────────────────────
    def draw_dimension_vertical(x, y1, y2, text):
        """
        Vertical double-headed arrow at x, between y1 and y2.
        Label is rotated 90° and placed on a pill to the RIGHT of the line.
        """
        # Main line (leave room for arrow heads)
        draw.line([(x, y1 + ARROW_LEN), (x, y2 - ARROW_LEN)],
                  fill=COLOR_LINE, width=LINE_W)
        # Arrow heads
        arrow_tip_up(draw, x, y1)
        arrow_tip_down(draw, x, y2)

        # Tick marks (small horizontal serifs at y1 and y2)
        tick = 6
        draw.line([(x - tick, y1), (x + tick, y1)], fill=COLOR_LINE, width=LINE_W)
        draw.line([(x - tick, y2), (x + tick, y2)], fill=COLOR_LINE, width=LINE_W)

        # Rotated label on pill background
        tw = text_w(text)
        th = text_h(text)
        label_w = int(tw + 16)
        label_h = int(th + 10)

        txt_img  = Image.new("RGBA", (label_w, label_h), (255, 255, 255, 0))
        txt_draw = ImageDraw.Draw(txt_img)
        # pill bg on the rotated canvas
        txt_draw.rounded_rectangle(
            [0, 0, label_w - 1, label_h - 1],
            radius=6, fill=COLOR_PILL_BG + "EE", outline=COLOR_PILL_BD
        )
        txt_draw.text((8, 5), text, fill=COLOR_TEXT, font=font)
        txt_img  = txt_img.rotate(90, expand=True)

        mid_y   = (y1 + y2) // 2
        paste_x = x + 6                         # 6 px gap from the line
        paste_y = mid_y - txt_img.height // 2
        canvas.paste(txt_img, (paste_x, paste_y), txt_img)

    # ── Helper: horizontal dimension bracket ────────────────────────────────
    def draw_dimension_horizontal(y, x1, x2, text):
        draw.line([(x1 + ARROW_LEN, y), (x2 - ARROW_LEN, y)],
                  fill=COLOR_LINE, width=LINE_W)
        arrow_tip_left(draw, x1, y)
        arrow_tip_right(draw, x2, y)

        tick = 6
        draw.line([(x1, y - tick), (x1, y + tick)], fill=COLOR_LINE, width=LINE_W)
        draw.line([(x2, y - tick), (x2, y + tick)], fill=COLOR_LINE, width=LINE_W)

        cx = (x1 + x2) // 2
        draw_pill_label(cx, y + 18, text)

    # ── Helper: feature line ─────────────────────────────────────────────────
    def draw_feature_line(photo_rel_y, right_ext, x_end=None):
        y      = photo_y + photo_rel_y
        x_stop = x_end if x_end is not None else (photo_x + w + right_ext)
        # Dashed feel: draw as a solid thin line with lighter colour
        draw.line([(photo_x, y), (x_stop, y)],
                  fill=COLOR_FEAT, width=1)
        # Small triangle marker at left photo edge
        draw.polygon(
            [(photo_x,     y),
             (photo_x + 7, y - 4),
             (photo_x + 7, y + 4)],
            fill=COLOR_FEAT
        )

    # ────────────────────────────────────────────────────────────────────────
    # 1. Total Height — left-side bracket
    #    x position = photo_x - 30  →  label goes further left, away from photo
    # ────────────────────────────────────────────────────────────────────────
    left_bracket_x = photo_x - 35   # plenty of gap from photo edge
    draw_dimension_vertical(
        left_bracket_x,
        photo_y,
        photo_y + h,
        total_height_text
    )

    # 2. Total Width — bottom bracket
    draw_dimension_horizontal(
        photo_y + h + 30,
        photo_x,
        photo_x + w,
        total_width_text
    )

    # 3. Feature lines (head top, chin, eyes)
    draw_feature_line(top_of_head_y, 50)
    draw_feature_line(chin_y,        50)
    draw_feature_line(eye_y,         90)

    # 4. Head Height bracket — first right bracket
    draw_dimension_vertical(
        photo_x + w + 25,
        photo_y + top_of_head_y,
        photo_y + chin_y,
        head_text
    )

    # 5. Eye from Bottom bracket — second right bracket
    draw_dimension_vertical(
        photo_x + w + 70,
        photo_y + eye_y,
        photo_y + h,
        eye_text
    )

    # ── Output ───────────────────────────────────────────────────────────────
    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=92)
    return buf.getvalue()