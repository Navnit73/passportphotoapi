"""
Hair crown detection using MediaPipe ImageSegmenter (tasks API).

Extracts the person mask and finds the topmost non-background
pixel — the hair crown. This MUST run on the ORIGINAL image
BEFORE any background replacement.
"""

import logging
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)

# ─── Model path ───
MODEL_PATH = str(Path(__file__).parent.parent / "models" / "image_segmenter.tflite")

# ─── Singleton segmenter instance ───
_segmenter = None


def _get_segmenter():
    """Get or create the MediaPipe ImageSegmenter instance."""
    global _segmenter
    if _segmenter is None:
        logger.info(f"Initializing MediaPipe ImageSegmenter from {MODEL_PATH}...")

        BaseOptions = mp.tasks.BaseOptions
        ImageSegmenter = mp.tasks.vision.ImageSegmenter
        ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.IMAGE,
            output_category_mask=False,
            output_confidence_masks=True,
        )

        _segmenter = ImageSegmenter.create_from_options(options)
        logger.info("MediaPipe ImageSegmenter initialized")

    return _segmenter


def detect_hair_crown(image: np.ndarray) -> float:
    """
    Detect the hair crown (topmost point of person) using segmentation.

    IMPORTANT: This must be called on the ORIGINAL image before
    any background replacement.

    Args:
        image: BGR numpy array (original, unmodified image)

    Returns:
        crown_y: Y-coordinate (pixels) of the topmost person pixel.

    Raises:
        ValueError: if segmentation fails or no person found
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid or empty image")

    h, w = image.shape[:2]

    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    segmenter = _get_segmenter()
    result = segmenter.segment(mp_image)

    if not result.confidence_masks or len(result.confidence_masks) == 0:
        raise ValueError("Segmentation failed — no mask produced")

    # The selfie segmenter returns a single confidence mask
    # where higher values = person
    mask = result.confidence_masks[0].numpy_view()

    # Threshold the mask to get a binary person mask
    binary_mask = (mask > 0.5).astype(np.uint8)

    # Find the topmost row with any person pixel
    person_rows = np.where(binary_mask.any(axis=1))[0]

    if len(person_rows) == 0:
        raise ValueError("No person detected in segmentation mask")

    crown_y = float(person_rows[0])

    logger.info(
        f"Hair crown detected at y={crown_y:.0f} "
        f"(image height={h}, crown at {(crown_y / h) * 100:.1f}% from top)"
    )

    return crown_y


def get_person_mask(image: np.ndarray) -> np.ndarray:
    """
    Get the full segmentation mask for background operations.

    Returns a float32 mask where 1.0 = person, 0.0 = background.

    Args:
        image: BGR numpy array

    Returns:
        mask: float32 array (H, W) with values 0.0-1.0
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid or empty image")

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    segmenter = _get_segmenter()
    result = segmenter.segment(mp_image)

    if not result.confidence_masks or len(result.confidence_masks) == 0:
        raise ValueError("Segmentation failed — no mask produced")

    mask = result.confidence_masks[0].numpy_view().copy()
    return mask.astype(np.float32)
