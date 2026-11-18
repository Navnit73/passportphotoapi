"""
Independent measurement of a face photo against US requirements.

* Face landmarks / pose / expression : own MediaPipe FaceLandmarker instance
* Top of hair                         : alpha matte (MODNet -> rembg, shared
                                        background-removal service)
* Chin / eyes                         : landmarks 152 and iris centres
"""

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from services.background import _remove_background  # shared matting (MODNet -> rembg)

logger = logging.getLogger(__name__)

MODEL_PATH = str(Path(__file__).parent.parent / "models" / "face_landmarker.task")

L_IRIS, R_IRIS = 468, 473
CHIN, FOREHEAD, NOSE_TIP = 152, 10, 1
FACE_L, FACE_R = 234, 454            # widest face-contour points
_L_EYE = [33, 160, 158, 133, 153, 144]
_R_EYE = [362, 385, 387, 263, 373, 380]

_landmarker = None
_lock = threading.Lock()


class USPhotoError(Exception):
    """A photo problem the user must fix (maps to HTTP 400)."""

    def __init__(self, message: str, code: str = "invalid_photo"):
        super().__init__(message)
        self.code = code


@dataclass
class Face:
    iris_l: tuple[float, float]        # image-left iris (x, y) px
    iris_r: tuple[float, float]
    eye_y: float
    mid_x: float                       # face mid-line x
    chin_y: float
    forehead_y: float
    face_w: float                      # cheek-to-cheek width px
    roll: float                        # deg, + = image-right eye lower
    yaw: float
    pitch: float
    blink: float                       # 0 open .. 1 closed (worse eye)
    jaw_open: float
    smile: float
    w: int
    h: int


@dataclass
class Measurement:
    face: Face
    alpha: np.ndarray                  # float32 0..1 matte
    crown_y: float
    crown_estimated: bool
    bg_mean: tuple[float, float, float]
    bg_std: float
    bg_pixels: int
    warnings: list[str] = field(default_factory=list)

    @property
    def head_h(self) -> float:
        return self.face.chin_y - self.crown_y


def _get_landmarker():
    global _landmarker
    if _landmarker is None:
        opts = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=4,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        _landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(opts)
    return _landmarker


def _euler(matrix: np.ndarray) -> tuple[float, float, float]:
    r = np.array(matrix)[:3, :3].astype(np.float64)
    r = r / np.maximum(np.linalg.norm(r, axis=0, keepdims=True), 1e-9)
    yaw = np.degrees(np.arcsin(np.clip(r[0, 2], -1, 1)))
    pitch = np.degrees(np.arctan2(-r[1, 2], r[2, 2]))
    roll = np.degrees(np.arctan2(-r[0, 1], r[1, 1]))
    return float(yaw), float(pitch), float(roll)


def detect_face(img: np.ndarray) -> Face:
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with _lock:
        res = _get_landmarker().detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        )
    if not res.face_landmarks:
        raise USPhotoError("No face detected. Use a clear, front-facing photo.", "no_face")
    if len(res.face_landmarks) > 1:
        raise USPhotoError(
            f"{len(res.face_landmarks)} faces detected. Only one person may appear in the photo.",
            "multiple_faces",
        )
    lm = res.face_landmarks[0]

    def pt(i):
        return lm[i].x * w, lm[i].y * h

    # iris centres (pupil level); fall back to eye-contour means
    try:
        a, b = pt(L_IRIS), pt(R_IRIS)
    except IndexError:
        a = tuple(np.mean([pt(i) for i in _L_EYE], axis=0))
        b = tuple(np.mean([pt(i) for i in _R_EYE], axis=0))
    il, ir = (a, b) if a[0] <= b[0] else (b, a)

    roll = float(np.degrees(np.arctan2(ir[1] - il[1], ir[0] - il[0])))
    eye_y = (il[1] + ir[1]) / 2
    cheek_l, cheek_r = pt(FACE_L), pt(FACE_R)
    mid_x = float(np.mean([(il[0] + ir[0]) / 2, pt(CHIN)[0], pt(FOREHEAD)[0],
                           (cheek_l[0] + cheek_r[0]) / 2]))

    yaw = pitch = 0.0
    if res.facial_transformation_matrixes is not None and len(res.facial_transformation_matrixes):
        yaw, pitch, _ = _euler(res.facial_transformation_matrixes[0])

    bs = {c.category_name: c.score for c in (res.face_blendshapes[0] if res.face_blendshapes else [])}
    return Face(
        iris_l=il, iris_r=ir, eye_y=float(eye_y), mid_x=mid_x,
        chin_y=float(pt(CHIN)[1]), forehead_y=float(pt(FOREHEAD)[1]),
        face_w=float(abs(cheek_r[0] - cheek_l[0])),
        roll=roll, yaw=yaw, pitch=pitch,
        blink=float(max(bs.get("eyeBlinkLeft", 0), bs.get("eyeBlinkRight", 0))),
        jaw_open=float(bs.get("jawOpen", 0)),
        smile=float(max(bs.get("mouthSmileLeft", 0), bs.get("mouthSmileRight", 0))),
        w=w, h=h,
    )


def compute_alpha(img: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgba = np.array(_remove_background(rgb).convert("RGBA"))
    return rgba[..., 3].astype(np.float32) / 255.0


def find_crown(alpha: np.ndarray, face: Face) -> tuple[float, bool]:
    """Topmost hair pixel of the connected person blob that contains the face."""
    mask = (alpha > 0.5).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    fallback = face.chin_y - 1.30 * (face.chin_y - face.forehead_y)

    # person blob = first labelled pixel on the face mid-line between eyes and chin
    px = int(np.clip(face.mid_x, 0, face.w - 1))
    lab = 0
    for y in np.linspace(face.eye_y, face.chin_y, 8):
        lab = labels[int(np.clip(y, 0, face.h - 1)), px]
        if lab:
            break
    if lab == 0:
        return max(0.0, fallback), True
    ys, xs = np.where(labels == lab)
    band = np.abs(xs - face.mid_x) <= 0.85 * face.face_w
    if not band.any():
        return max(0.0, fallback), True
    crown = float(ys[band].min())
    # sanity: hair top must sit above the forehead landmark
    if crown >= face.forehead_y - 0.02 * (face.chin_y - face.forehead_y):
        return max(0.0, fallback), True
    return crown, False


def background_stats(img: np.ndarray, alpha: np.ndarray):
    bg = (alpha < 0.03).astype(np.uint8)
    bg = cv2.erode(bg, np.ones((9, 9), np.uint8))
    px = img[bg > 0]
    if px.shape[0] < 200:
        return (0.0, 0.0, 0.0), 99.0, int(px.shape[0])
    rgb = px[:, ::-1].astype(np.float64)
    return tuple(float(v) for v in rgb.mean(0)), float(rgb.std(0).mean()), int(px.shape[0])


def measure(img: np.ndarray, face: Face | None = None, alpha: np.ndarray | None = None) -> Measurement:
    face = face or detect_face(img)
    alpha = alpha if alpha is not None else compute_alpha(img)
    crown, est = find_crown(alpha, face)
    mean, std, n = background_stats(img, alpha)
    warns = []
    if est:
        warns.append("Top of hair could not be segmented; head height was estimated.")
    ratio = (face.chin_y - crown) / max(face.chin_y - face.forehead_y, 1)
    if not est and ratio > 1.9:
        warns.append("Unusually tall hair/headwear above the forehead; check head size.")
    return Measurement(face, alpha, crown, est, mean, std, n, warns)
