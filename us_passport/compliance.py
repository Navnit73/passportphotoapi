"""
Compliance checks against the US Department of State photo rules.
Used for BOTH /validate (as-uploaded file) and the final verification of
every /process result, so a processed photo is always judged by the same rules.
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from . import spec
from .measure import Measurement


@dataclass
class Check:
    key: str
    label: str
    value: str
    target: str
    status: str            # success | warning | error
    details: Optional[str] = None


@dataclass
class Metrics:
    head_pct: float        # head height / image height * 100
    eye_pct: float         # eye line from bottom * 100
    top_margin_pct: float
    chin_margin_pct: float
    center_offset_pct: float   # + = face right of centre
    width: int
    height: int


def metrics_for(m: Measurement) -> Metrics:
    f, H, W = m.face, m.face.h, m.face.w
    return Metrics(
        head_pct=100 * (f.chin_y - m.crown_y) / H,
        eye_pct=100 * (H - f.eye_y) / H,
        top_margin_pct=100 * m.crown_y / H,
        chin_margin_pct=100 * (H - f.chin_y) / H,
        center_offset_pct=100 * (f.mid_x - W / 2) / W,
        width=W, height=H,
    )


def _range(v, lo, hi):
    return "success" if lo <= v <= hi else "error"


def face_lighting(img: np.ndarray, m: Measurement) -> tuple[float, float, float]:
    """(mean luminance, left/right imbalance 0..1, sharpness) of the face area."""
    f = m.face
    y0, y1 = int(max(0, m.crown_y + 0.35 * (f.chin_y - m.crown_y))), int(min(f.h, f.chin_y))
    x0, x1 = int(max(0, f.mid_x - 0.45 * f.face_w)), int(min(f.w, f.mid_x + 0.45 * f.face_w))
    roi = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    if roi.size < 100:
        return 128.0, 0.0, 0.0
    mid = roi.shape[1] // 2
    l, r = roi[:, :mid].mean(), roi[:, mid:].mean()
    imbalance = abs(l - r) / max(l + r, 1.0) * 2
    sharp = float(cv2.Laplacian(roi, cv2.CV_32F).var())
    return float(roi.mean()), float(imbalance), sharp


def evaluate(
    img: np.ndarray,
    m: Measurement,
    file_bytes: Optional[bytes] = None,
    file_format: Optional[str] = None,
    dpi: Optional[tuple] = None,
    size: Optional[tuple] = None,
) -> tuple[list[Check], Metrics]:
    """Run every check. `file_*` args enable the file-level checks."""
    f = m.face
    mt = metrics_for(m)
    c: list[Check] = []

    # ── file level ──
    if file_format is not None:
        c.append(Check("format", "File format", file_format, "JPEG",
                       "success" if file_format.upper() in ("JPEG", "JPG") else "error"))
    if file_bytes is not None:
        kb = len(file_bytes) / 1024
        c.append(Check("size", "File size", f"{kb:.0f} KB", f"≤ {spec.MAX_FILE_KB} KB",
                       "success" if kb <= spec.MAX_FILE_KB else "error"))
    rw, rh = size if size else (mt.width, mt.height)      # true file size, not the analysed copy
    sq = rw == rh
    in_range = spec.MIN_PX <= rw <= spec.MAX_PX
    c.append(Check("resolution", "Resolution", f"{rw}x{rh}px",
                   f"square, {spec.MIN_PX}–{spec.MAX_PX}px",
                   "success" if (sq and in_range) else "error",
                   None if sq else "Photo must be square (1:1)."))
    if dpi is not None:
        d = dpi[0] if dpi else 0
        c.append(Check("dpi", "DPI", str(int(d)) if d else "none", "≥ 300 recommended",
                       "success" if d >= 300 else "warning"))

    # ── geometry ──
    c.append(Check("head", "Head size", f"{mt.head_pct:.1f}% of height",
                   f"{spec.HEAD_MIN*100:.0f}–{spec.HEAD_MAX*100:.0f}% (1\"–1⅜\")",
                   _range(mt.head_pct, spec.HEAD_MIN * 100, spec.HEAD_MAX * 100),
                   "Top of hair to bottom of chin." + (" (hair top estimated)" if m.crown_estimated else "")))
    c.append(Check("eye", "Eye height", f"{mt.eye_pct:.1f}% from bottom",
                   f"{spec.EYE_MIN*100:.0f}–{spec.EYE_MAX*100:.0f}% (1⅛\"–1⅜\")",
                   _range(mt.eye_pct, spec.EYE_MIN * 100, spec.EYE_MAX * 100)))
    top_ok = mt.top_margin_pct >= 0.5 and mt.chin_margin_pct > 0
    c.append(Check("margins", "Full head visible",
                   f"top {mt.top_margin_pct:.1f}% / chin-to-bottom {mt.chin_margin_pct:.1f}%",
                   "whole head inside frame", "success" if top_ok else "error"))
    off = abs(mt.center_offset_pct) / 100
    c.append(Check("centered", "Head centered", f"{mt.center_offset_pct:+.1f}% off centre",
                   f"within ±{spec.CENTER_TOL_PASS*100:.0f}%",
                   "success" if off <= spec.CENTER_TOL_PASS else
                   "warning" if off <= spec.CENTER_TOL_FAIL else "error"))

    # ── pose / expression ──
    def tri(v, p, fl):
        v = abs(v)
        return "success" if v <= p else "warning" if v <= fl else "error"

    c.append(Check("tilt", "Head tilt", f"{f.roll:+.1f}°", f"≤ {spec.ROLL_PASS:.0f}°", tri(f.roll, spec.ROLL_PASS, spec.ROLL_FAIL)))
    c.append(Check("orientation", "Facing camera", f"yaw {f.yaw:+.1f}°, pitch {f.pitch:+.1f}°",
                   "straight at camera",
                   "error" if "error" in (tri(f.yaw, spec.YAW_PASS, spec.YAW_FAIL), tri(f.pitch, spec.PITCH_PASS, spec.PITCH_FAIL))
                   else "warning" if "warning" in (tri(f.yaw, spec.YAW_PASS, spec.YAW_FAIL), tri(f.pitch, spec.PITCH_PASS, spec.PITCH_FAIL))
                   else "success"))
    c.append(Check("eyes", "Eyes open", f"blink score {f.blink:.2f}", "both eyes open",
                   "success" if f.blink < 0.45 else "error"))
    c.append(Check("expression", "Neutral expression", f"jaw {f.jaw_open:.2f}, smile {f.smile:.2f}",
                   "mouth closed, neutral or natural smile",
                   "error" if f.jaw_open > 0.30 else "warning" if (f.jaw_open > 0.15 or f.smile > 0.6) else "success"))

    # ── image quality ──
    lum, imb, sharp = face_lighting(img, m)
    c.append(Check("lighting", "Lighting", f"brightness {lum:.0f}, imbalance {imb*100:.0f}%",
                   "even, no shadows on face",
                   "error" if (lum < 55 or lum > 235 or imb > 0.35) else
                   "warning" if (lum < 80 or lum > 215 or imb > 0.2) else "success"))
    c.append(Check("sharpness", "Sharpness", f"{sharp:.0f}", "in focus, not blurry",
                   "success" if sharp >= 25 else "warning" if sharp >= 10 else "error"))

    # ── background ──
    r, g, b = m.bg_mean
    if m.bg_pixels < 200:
        c.append(Check("background", "Background", "not measurable", "plain white/off-white", "warning"))
    else:
        mn = min(r, g, b)
        ok = mn >= spec.BG_MIN_CHANNEL and m.bg_std <= spec.BG_MAX_STD
        c.append(Check("background", "Background", f"rgb({r:.0f},{g:.0f},{b:.0f}), σ={m.bg_std:.1f}",
                       "plain white/off-white, no shadows",
                       "success" if ok else "error"))
    return c, mt


def overall(checks: list[Check]) -> str:
    return "FAIL" if any(x.status == "error" for x in checks) else "PASS"


def score(checks: list[Check]) -> int:
    pts = {"success": 1.0, "warning": 0.6, "error": 0.0}
    return round(100 * sum(pts[x.status] for x in checks) / max(len(checks), 1))
