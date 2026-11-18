"""
Preview image: the final photo on a white canvas with official-style
dimension lines (width, height, head size, eye height) and measured guides.

`render_measurement_preview` is generic (any photo size / unit) and is used by
both the main /process endpoint (via `create_preview_image`) and the US app.
"""

import io
import logging
from fractions import Fraction

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from config.countries.schema import CountryConfig

logger = logging.getLogger(__name__)

INK = (28, 28, 36)
PURPLE = (92, 76, 220)
SLATE = (132, 140, 168)
TEAL = (22, 150, 130)
RED = (200, 60, 60)
MUTED = (110, 114, 130)

_FONT_CANDIDATES = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "DejaVuSans.ttf",
]


def _font(size: int) -> ImageFont.FreeTypeFont:
    for p in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def _pill(text: str, size: int, color, rotate: bool = False) -> Image.Image:
    f = _font(size)
    l, t, r, b = f.getbbox(text)
    padx, pady = int(size * 0.62), int(size * 0.42)
    w, h = (r - l) + 2 * padx, (b - t) + 2 * pady
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((1, 1, w - 2, h - 2), radius=int(h * 0.3), fill=(255, 255, 255, 255),
                        outline=color, width=max(2, size // 14))
    d.text((padx - l, pady - t), text, font=f, fill=INK)
    return im.rotate(90, expand=True) if rotate else im


# ─────────────────────────── label formatting ───────────────────────────

def fmt_inches(v: float) -> str:
    """1.375 -> '1 3/8' (nearest 1/8 inch)."""
    fr = Fraction(round(v * 8), 8)
    whole, rem = divmod(fr.numerator, fr.denominator)
    if rem == 0:
        return str(whole)
    return f"{whole} {Fraction(rem, fr.denominator)}" if whole else str(Fraction(rem, fr.denominator))


def fmt_mm(v: float) -> str:
    return f"{v:.1f}".rstrip("0").rstrip(".")


def _range_label(lo: float, hi: float, unit: str) -> str:
    if unit == "in":
        return f"{fmt_inches(lo)}–{fmt_inches(hi)} in"
    return f"{fmt_mm(lo)}–{fmt_mm(hi)} mm"


# ─────────────────────────── generic renderer ───────────────────────────

def render_measurement_preview(
    image: np.ndarray,
    crown_frac: float,
    chin_frac: float,
    eye_frac: float,
    width_label: str,
    height_label: str,
    head_label: str,
    eye_label: str,
    caption: str,
    compliant: bool = True,
) -> bytes:
    """
    Args:
        image: final BGR photo.
        crown_frac / chin_frac / eye_frac: positions as fractions of the image
            height measured from the TOP (0 = top edge, 1 = bottom edge).
    """
    K = 2                                        # supersample for crisp lines
    ih, iw = image.shape[:2]
    long_side = 600 * K
    sc = long_side / max(ih, iw)
    pw, ph = int(round(iw * sc)), int(round(ih * sc))

    left, right, top, bottom = 250 * K, 310 * K, 250 * K, 190 * K
    W, H = pw + left + right, ph + top + bottom
    x0, y0 = left, top
    x1, y1 = x0 + pw, y0 + ph

    canvas = Image.new("RGB", (W, H), (255, 255, 255))

    # ── soft shadow + rounded photo card ──
    rad = 22 * K
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle((x0, y0 + 14 * K, x1, y1 + 14 * K), rad, fill=(40, 50, 80, 70))
    shadow = shadow.filter(ImageFilter.GaussianBlur(22 * K))
    canvas.paste(shadow, (0, 0), shadow)
    photo = Image.fromarray(cv2.cvtColor(cv2.resize(image, (pw, ph), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB))
    mask = Image.new("L", (pw, ph), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, pw - 1, ph - 1), rad, fill=255)
    canvas.paste(photo, (x0, y0), mask)
    d = ImageDraw.Draw(canvas)
    d.rounded_rectangle((x0, y0, x1, y1), rad, outline=(226, 229, 238), width=K)
    lw = 2 * K

    def tick_v(x, ya, yb, col, t=9 * K):
        d.line((x, ya, x, yb), fill=col, width=lw)
        for y in (ya, yb):
            d.line((x - t, y, x + t, y), fill=col, width=lw)

    def tick_h(y, xa, xb, col, t=9 * K):
        d.line((xa, y, xb, y), fill=col, width=lw)
        for x in (xa, xb):
            d.line((x, y - t, x, y + t), fill=col, width=lw)

    # ── top: photo width ──
    ty = y0 - 62 * K
    tick_h(ty, x0, x1, SLATE)
    pill = _pill(width_label, 30 * K, PURPLE)
    cx = (x0 + x1) // 2
    d.rectangle((cx - pill.width // 2 - 14 * K, ty - 6 * K, cx + pill.width // 2 + 14 * K, ty + 6 * K), fill=(255, 255, 255))
    canvas.paste(pill, (cx - pill.width // 2, ty - pill.height // 2 - 30 * K), pill)

    # ── far right: photo height ──
    rx = x1 + 120 * K
    tick_v(rx, y0, y1, SLATE)
    pill = _pill(height_label, 30 * K, PURPLE)
    canvas.paste(pill, (rx + 26 * K, (y0 + y1) // 2 - pill.height // 2), pill)

    # ── left: head height bracket + guides ──
    ycrown, ychin, yeye = y0 + crown_frac * ph, y0 + chin_frac * ph, y0 + eye_frac * ph
    lx = x0 - 62 * K
    tick_v(lx, int(ycrown), int(ychin), SLATE)
    for y in (ycrown, ychin):
        d.line((lx, y, x0 + int(pw * 0.80), y), fill=PURPLE, width=lw)
    pill = _pill(head_label, 28 * K, PURPLE, rotate=True)
    canvas.paste(pill, (lx - pill.width - 22 * K, int((ycrown + ychin) / 2 - pill.height / 2)), pill)

    # ── right: eye height bracket (eye line -> bottom edge) ──
    ex = x1 + 52 * K
    tick_v(ex, int(yeye), int(y1), TEAL)
    d.line((x0 + int(pw * 0.74), yeye, ex, yeye), fill=TEAL, width=lw)
    pill = _pill(eye_label, 26 * K, TEAL, rotate=True)
    canvas.paste(pill, (ex + 14 * K, int((yeye + y1) / 2 - pill.height / 2)), pill)

    # ── caption + badge ──
    cf = _font(21 * K)
    tw = d.textlength(caption, font=cf)
    d.text(((W - tw) / 2, y1 + 56 * K), caption, font=cf, fill=MUTED)
    badge = _pill("Meets requirements" if compliant else "Needs attention", 22 * K, TEAL if compliant else RED)
    canvas.paste(badge, ((W - badge.width) // 2, y1 + 104 * K), badge)

    out = canvas.resize((W // K, H // K), Image.LANCZOS)
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=92, subsampling=0)
    return buf.getvalue()


# ─────────────────────────── main /process entry point ───────────────────────────

def create_preview_image(
    image: np.ndarray,
    config: CountryConfig,
    head_height_pct: float,
    eye_from_bottom_pct: float,
    top_margin_pct: float,
) -> bytes:
    """Preview for the main pipeline; labels come from the country config."""
    ps = config.photo_spec
    if ps.height_in:
        unit, f = "in", 1.0
        hv = ps.height_in
        wv = ps.width_in or hv * image.shape[1] / image.shape[0]
        wl, hl = f"{fmt_inches(wv)} in", f"{fmt_inches(hv)} in"
    else:
        unit = "mm"
        hv = ps.height_mm or 0.0
        wv = ps.width_mm or hv * image.shape[1] / max(image.shape[0], 1)
        wl, hl = f"{fmt_mm(wv)} mm", f"{fmt_mm(hv)} mm"

    fc = config.face_constraints
    hmin, hmax = fc.head_height_pct.min, fc.head_height_pct.max
    emin, emax = fc.eye_position.from_bottom_pct.min, fc.eye_position.from_bottom_pct.max
    head_label = _range_label(hv * hmin / 100, hv * hmax / 100, unit)
    eye_label = _range_label(hv * emin / 100, hv * emax / 100, unit)

    compliant = hmin <= head_height_pct <= hmax and emin <= eye_from_bottom_pct <= emax
    caption = (f"Head {head_height_pct:.1f}%  ·  Eyes {eye_from_bottom_pct:.1f}% from bottom"
               f"  ·  Top margin {top_margin_pct:.1f}%")
    return render_measurement_preview(
        image,
        crown_frac=top_margin_pct / 100,
        chin_frac=(top_margin_pct + head_height_pct) / 100,
        eye_frac=1.0 - eye_from_bottom_pct / 100,
        width_label=wl, height_label=hl,
        head_label=head_label, eye_label=eye_label,
        caption=caption, compliant=compliant,
    )
