"""
Server-side face detection using MediaPipe FaceLandmarker (tasks API).

Detects facial landmarks (478-point mesh), extracts key biometric
points (eyes, chin, forehead), and validates face presence.

This runs entirely server-side — no client-side detection needed.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)

# ─── Model path ───
MODEL_PATH = str(Path(__file__).parent.parent / "models" / "face_landmarker.task")

# ─── MediaPipe Landmark Indices (478-point mesh) ───
# Same indices as the existing TypeScript mediapipe.ts

# Eye contours
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# Iris centers
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473

# Key points
CHIN_TIP = 152
NOSE_TIP = 1
FOREHEAD_TOP = 10

# Full jaw outline — 17 points matching the TS JAW_OUTLINE_INDICES
JAW_OUTLINE_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454,  # right side
    152,                                            # chin center
    234, 127, 162, 21, 54, 103, 67,                # left side
]


@dataclass
class Point:
    x: float
    y: float


@dataclass
class FaceBox:
    x: float
    y: float
    width: float
    height: float


@dataclass
class FaceDetectionResult:
    """Result of face detection on an image."""
    eye_center: Point
    left_eye_center: Point
    right_eye_center: Point
    chin_y: float
    forehead_y: float
    top_of_head_y: float  # estimated crown position
    face_box: FaceBox
    tilt_angle: float
    image_width: int
    image_height: int
    face_count: int


# ─── Singleton face landmarker instance ───
_face_landmarker = None


def _get_face_landmarker():
    """Get or create the MediaPipe FaceLandmarker instance (singleton)."""
    global _face_landmarker
    if _face_landmarker is None:
        logger.info(f"Initializing MediaPipe FaceLandmarker from {MODEL_PATH}...")

        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=5,  # detect up to 5 for multi-face rejection
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )

        _face_landmarker = FaceLandmarker.create_from_options(options)
        logger.info("MediaPipe FaceLandmarker initialized successfully")

    return _face_landmarker


def _center_of_points(points: list[Point]) -> Point:
    """Compute the average center of a set of points."""
    if not points:
        raise ValueError("Cannot compute center of empty points list")
    avg_x = sum(p.x for p in points) / len(points)
    avg_y = sum(p.y for p in points) / len(points)
    return Point(avg_x, avg_y)


def _compute_tilt_angle(left_eye: Point, right_eye: Point) -> float:
    """Compute head tilt angle in degrees using eye centers."""
    dy = left_eye.y - right_eye.y
    dx = left_eye.x - right_eye.x
    angle_rad = np.arctan2(dy, dx)
    return float(np.degrees(angle_rad))


# ─── Head top multiplier (matches TS HEAD_TOP_MULTIPLIER) ───
HEAD_TOP_MULTIPLIER = 1.34


def detect_face(image: np.ndarray) -> FaceDetectionResult:
    """
    Detect face in an image using MediaPipe FaceLandmarker.

    Args:
        image: BGR numpy array (from cv2.imread or similar)

    Returns:
        FaceDetectionResult with all landmarks in pixel coordinates

    Raises:
        ValueError: if no face detected or multiple faces
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid or empty image")

    h, w = image.shape[:2]

    # Convert BGR to RGB for MediaPipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    # Detect
    landmarker = _get_face_landmarker()
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        raise ValueError("No face detected in image")

    face_count = len(result.face_landmarks)
    logger.info(f"Detected {face_count} face(s)")

    if face_count > 1:
        raise ValueError(
            f"Multiple faces detected ({face_count}). "
            "Passport photos must contain exactly one face."
        )

    # Use the first (and only) face
    landmarks = result.face_landmarks[0]

    # ─── Convert normalized landmarks to pixel coordinates ───
    def to_px(idx: int) -> Point:
        lm = landmarks[idx]
        return Point(x=lm.x * w, y=lm.y * h)

    def points_for(indices: list[int]) -> list[Point]:
        return [to_px(i) for i in indices]

    # Eye centers
    left_eye_points = points_for(LEFT_EYE_INDICES)
    right_eye_points = points_for(RIGHT_EYE_INDICES)
    left_eye_center = _center_of_points(left_eye_points)
    right_eye_center = _center_of_points(right_eye_points)
    eye_center = _center_of_points([left_eye_center, right_eye_center])

    # Chin and forehead
    chin = to_px(CHIN_TIP)
    forehead = to_px(FOREHEAD_TOP)

    # ─── Face bounding box from all landmarks ───
    all_x = [lm.x * w for lm in landmarks]
    all_y = [lm.y * h for lm in landmarks]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    face_box = FaceBox(
        x=min_x, y=min_y,
        width=max_x - min_x,
        height=max_y - min_y,
    )

    # ─── Estimate top of head (crown) ───
    face_height = chin.y - forehead.y
    estimated_head_height = face_height * HEAD_TOP_MULTIPLIER
    top_of_head_y = chin.y - estimated_head_height
    top_of_head_y = max(0.0, top_of_head_y)

    # ─── Tilt angle ───
    tilt_angle = _compute_tilt_angle(left_eye_center, right_eye_center)

    logger.info(
        f"Face detected: eye_center=({eye_center.x:.0f}, {eye_center.y:.0f}), "
        f"chin_y={chin.y:.0f}, top_of_head_y={top_of_head_y:.0f}, "
        f"tilt={tilt_angle:.1f}°"
    )

    return FaceDetectionResult(
        eye_center=eye_center,
        left_eye_center=left_eye_center,
        right_eye_center=right_eye_center,
        chin_y=chin.y,
        forehead_y=forehead.y,
        top_of_head_y=top_of_head_y,
        face_box=face_box,
        tilt_angle=tilt_angle,
        image_width=w,
        image_height=h,
        face_count=face_count,
    )
