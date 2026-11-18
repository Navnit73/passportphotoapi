"""
US passport photo engine: align -> matte/background -> solve crop -> render ->
JPEG -> independent re-measurement (closed loop).
"""

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from services.compressor import compress_to_jpeg  # generic JPEG size binary-search

from . import spec
from .compliance import Check, Metrics, evaluate, overall
from .measure import Measurement, USPhotoError, compute_alpha, detect_face, measure

logger = logging.getLogger(__name__)

MAX_PASSES = 3
GEOMETRY_KEYS = {"head", "eye", "margins", "centered"}


@dataclass
class CropPlan:
    side: float            # source px that map to the output square
    x0: float
    y0: float
    head_frac: float
    eye_frac: float
    top_frac: float


@dataclass
class USResult:
    jpeg: bytes
    image: np.ndarray               # decoded final (BGR)
    checks: list[Check]
    metrics: Metrics
    measurement: Measurement
    background_corrected: bool
    rotated_deg: float
    passes: int
    warnings: list[str] = field(default_factory=list)

    @property
    def compliant(self) -> bool:
        return overall(self.checks) == "PASS"


# ─────────────────────────── alignment ───────────────────────────

def auto_orient(img: np.ndarray) -> np.ndarray:
    """Fix sideways / upside-down photos (missing EXIF) by finding the upright orientation."""
    def upright(im):
        try:
            f = detect_face(im)
            return abs(f.roll) < 45
        except USPhotoError as e:
            if e.code == "multiple_faces":
                raise
            return False
    if upright(img):
        return img
    for code in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180):
        cand = cv2.rotate(img, code)
        if upright(cand):
            logger.info("Auto-rotated photo to upright orientation")
            return cand
    return img


def straighten(img: np.ndarray):
    """Level the eye line. Returns (image, face, rotation_deg)."""
    img = auto_orient(img)
    face = detect_face(img)
    if abs(face.roll) > spec.ROLL_CORRECT_MAX:
        raise USPhotoError(
            f"Head is tilted {face.roll:.0f}°. Please retake with your head upright.", "head_tilt")
    if abs(face.roll) < 0.4:
        return img, face, 0.0
    cx = (face.iris_l[0] + face.iris_r[0]) / 2
    cy = face.eye_y
    # positive roll == image-right eye lower -> rotate counter-clockwise (cv2: +angle)
    M = cv2.getRotationMatrix2D((cx, cy), face.roll, 1.0)
    out = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                         flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return out, detect_face(out), float(face.roll)


def check_pose(face):
    if abs(face.yaw) > spec.YAW_FAIL:
        raise USPhotoError(
            f"Face is turned {abs(face.yaw):.0f}° sideways. Look straight at the camera.", "face_turned")
    if abs(face.pitch) > spec.PITCH_FAIL:
        raise USPhotoError("Head is tilted up/down. Keep your chin level and look straight ahead.", "face_pitch")
    if face.blink >= 0.6:
        raise USPhotoError("Eyes appear closed. Both eyes must be open.", "eyes_closed")
    if face.jaw_open > 0.4:
        raise USPhotoError("Mouth is open. Use a neutral expression with mouth closed.", "mouth_open")


# ─────────────────────────── background ───────────────────────────

def bg_is_acceptable(m: Measurement) -> bool:
    return (m.bg_pixels >= 0.02 * m.face.w * m.face.h
            and min(m.bg_mean) >= spec.BG_MIN_CHANNEL
            and m.bg_std <= spec.BG_MAX_STD - 2)


def composite_white(img: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = np.clip((alpha - 0.04) / 0.90, 0, 1)[..., None]
    out = img.astype(np.float32) * a + 255.0 * (1 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


# ─────────────────────────── crop solver ───────────────────────────

def solve_crop(crown_y, eye_y, chin_y, mid_x) -> CropPlan:
    """
    Pick head height H and eye-from-bottom E (fractions of the square) that sit
    closest to the centre of the legal window while keeping the whole head in frame.
    """
    head = chin_y - crown_y
    if head <= 10:
        raise USPhotoError("Could not measure head size.", "measure_failed")
    a = (eye_y - crown_y) / head          # eye position inside the head (0 top .. 1 chin)
    best = None
    for H in np.arange(0.53, 0.671, 0.0025):
        for E in np.arange(0.575, 0.676, 0.0025):
            t = 1 - E - a * H             # space above the hair
            if t < spec.MIN_TOP_MARGIN or (1 - t - H) < 0.05:
                continue
            cost = ((H - spec.TARGET_HEAD) / 0.04) ** 2 + ((E - spec.TARGET_EYE) / 0.04) ** 2
            if t < spec.PREFERRED_TOP_MARGIN:
                cost += ((spec.PREFERRED_TOP_MARGIN - t) / 0.02) ** 2
            if best is None or cost < best[0]:
                best = (cost, H, E, t)
    if best is None:
        raise USPhotoError(
            "Head proportions cannot fit the US template (hair/headwear too tall for the frame).",
            "cannot_fit")
    _, H, E, t = best
    side = head / H
    return CropPlan(side=side, x0=mid_x - side / 2, y0=crown_y - t * side,
                    head_frac=float(H), eye_frac=float(E), top_frac=float(t))


def render(img: np.ndarray, plan: CropPlan, pad_bgr) -> np.ndarray:
    """Sub-pixel crop + resample to OUT_PX x OUT_PX (padding outside the photo)."""
    s = spec.OUT_PX / plan.side
    if s < 1:   # downscale: INTER_AREA first (no aliasing), then translate
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        M = np.array([[1, 0, -plan.x0 * s], [0, 1, -plan.y0 * s]], dtype=np.float64)
    else:
        M = np.array([[s, 0, -plan.x0 * s], [0, s, -plan.y0 * s]], dtype=np.float64)
    return cv2.warpAffine(img, M, (spec.OUT_PX, spec.OUT_PX), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=pad_bgr)


# ─────────────────────────── pipeline ───────────────────────────

def is_inverted(m: Measurement) -> bool:
    """Upside-down photo: far more person mass above the eye line than below it.
    (Landmarks cannot tell, the face mesh is not reliable on inverted faces.)"""
    row = m.alpha.sum(axis=1)
    e = int(np.clip(m.face.eye_y, 0, len(row) - 1))
    return row[:e].sum() > 1.8 * max(row[e:].sum(), 1.0)


def process(img: np.ndarray, max_kb: int = spec.MAX_FILE_KB) -> USResult:
    if img is None or img.size == 0:
        raise USPhotoError("Invalid image.", "bad_image")

    # 1. straighten
    work, face, rot = straighten(img)
    check_pose(face)

    # 2. matte, crown, background
    m = measure(work, face=face)
    if is_inverted(m):
        logger.info("Photo is upside-down; rotating 180°")
        work, face, rot = straighten(cv2.rotate(work, cv2.ROTATE_180))
        check_pose(face)
        m = measure(work, face=face)
    corrected = False
    if bg_is_acceptable(m):
        pad = np.median(work[(m.alpha < 0.03)], axis=0) if m.bg_pixels else np.array([255, 255, 255])
        pad = tuple(float(v) for v in np.clip(pad, 0, 255))
        base = work
    else:
        base = composite_white(work, m.alpha)
        pad = (255.0, 255.0, 255.0)
        corrected = True

    crown, eye, chin, mid_x = m.crown_y, face.eye_y, face.chin_y, face.mid_x
    warnings = list(m.warnings)
    final = None

    # 3. closed loop: plan -> render -> encode -> re-measure -> correct bias
    for n in range(1, MAX_PASSES + 1):
        plan = solve_crop(crown, eye, chin, mid_x)
        if plan.side < spec.MIN_SOURCE_CROP_PX:
            raise USPhotoError(
                f"Photo resolution is too low (face crop only {plan.side:.0f}px). Use a higher-resolution photo.",
                "low_resolution")
        out = render(base, plan, pad)
        jpeg = compress_to_jpeg(out, max_size_kb=max_kb, target_dpi=spec.OUT_DPI)
        dec = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        try:
            fm = measure(dec)
        except USPhotoError as e:
            raise USPhotoError(f"Result verification failed: {e}", "verify_failed")
        # Hair top is hardest to segment on the white result (light hair on white).
        # If it disagrees with where the source-side measurement puts it, trust the source.
        expected_crown = (crown - plan.y0) * (spec.OUT_PX / plan.side)
        if abs(fm.crown_y - expected_crown) > 0.012 * spec.OUT_PX and not m.crown_estimated:
            logger.info(f"Final hair-top {fm.crown_y:.0f}px vs expected {expected_crown:.0f}px; using expected")
            fm.crown_y = float(max(0.0, expected_crown))
            warnings.append("Hair top is low-contrast against the white background; measured from the original.")
        checks, mt = evaluate(dec, fm, file_bytes=jpeg, file_format="JPEG", dpi=(spec.OUT_DPI,))
        final = (jpeg, dec, checks, mt, fm, plan)
        if overall(checks) == "PASS":
            logger.info(f"US photo compliant after {n} pass(es): head={mt.head_pct:.1f}% eye={mt.eye_pct:.1f}%")
            break
        if not any(c.status == "error" and c.key in GEOMETRY_KEYS for c in checks):
            break   # re-planning cannot fix blur / lighting / background / expression
        # Feed measured positions back (mapped to source coords) and re-plan.
        s = spec.OUT_PX / plan.side
        crown = plan.y0 + fm.crown_y / s
        chin = plan.y0 + fm.face.chin_y / s
        eye = plan.y0 + fm.face.eye_y / s
        mid_x = plan.x0 + fm.face.mid_x / s
        logger.info(f"Pass {n} not compliant ({[c.key for c in checks if c.status=='error']}); re-planning")

    jpeg, dec, checks, mt, fm, plan = final
    if plan.side < spec.WARN_SOURCE_CROP_PX:
        warnings.append(f"Low source resolution (head crop {plan.side:.0f}px, upscaled to {spec.OUT_PX}px); result may look soft.")
    if plan.y0 + plan.side > base.shape[0]:
        warnings.append("Photo ended near the shoulders; the bottom was padded with background colour.")
    warnings += [w for w in fm.warnings if w not in warnings]
    return USResult(jpeg, dec, checks, mt, fm, corrected, rot, n, warnings)
