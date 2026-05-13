"""
Background removal and color replacement.

Primary:  MODNet ONNX model (portrait-optimized matting)
Fallback: rembg with u2net model
"""

import logging
import os
import shutil
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

logger = logging.getLogger(__name__)

# ─── Config ───
MODNET_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "modnet.onnx")

BG_COLORS = {
    "white":       (255, 255, 255),
    "plain white": (255, 255, 255),
    "light-gray":  (243, 244, 246),
    "light-blue":  (239, 246, 255),
    "blue":        (0, 71, 171),
    "red":         (200, 30, 30),
}

# ─── Singleton sessions ───
_modnet_session = None
_rembg_session = None


# ═══════════════════════════════════════════════════════════════
#  Model Loading
# ═══════════════════════════════════════════════════════════════

def _download_modnet_model():
    """Download MODNet ONNX from Hugging Face if missing."""
    if os.path.exists(MODNET_MODEL_PATH):
        return True
    logger.info("Downloading MODNet ONNX model...")
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id="gradio/Modnet", filename="modnet.onnx")
        shutil.copy(path, MODNET_MODEL_PATH)
        logger.info("MODNet model downloaded.")
        return True
    except Exception as e:
        logger.error(f"MODNet download failed: {e}")
        return False


def _get_modnet_session():
    """Get or create MODNet ONNX session (singleton)."""
    global _modnet_session
    if _modnet_session is None:
        if not _download_modnet_model():
            _modnet_session = False
            return _modnet_session
        try:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            _modnet_session = ort.InferenceSession(
                MODNET_MODEL_PATH, opts, providers=["CPUExecutionProvider"]
            )
            logger.info("MODNet model loaded.")
        except Exception as e:
            logger.error(f"MODNet load error: {e}")
            _modnet_session = False
    return _modnet_session


def _get_rembg_session():
    """Get or create rembg u2net session (singleton fallback)."""
    global _rembg_session
    if _rembg_session is None:
        try:
            from rembg import new_session
            _rembg_session = new_session("u2net")
            logger.info("rembg u2net model loaded.")
        except Exception as e:
            logger.error(f"rembg load error: {e}")
            _rembg_session = False
    return _rembg_session


# ═══════════════════════════════════════════════════════════════
#  Background Removal (MODNet → u2net fallback)
# ═══════════════════════════════════════════════════════════════

def _run_modnet(image_rgb: np.ndarray) -> np.ndarray | None:
    """Run MODNet on an RGB array. Returns alpha matte (0-255) or None."""
    session = _get_modnet_session()
    if not session:
        return None
    try:
        h, w, _ = image_rgb.shape

        # MODNet expects fixed input size - keep original aspect ratio within max dimension
        max_dim = 1024  # or 2048, whatever the model was trained on
        scale = max_dim / max(h, w)
        if scale < 1.0:
            new_h, new_w = int(h * scale), int(w * scale)
            im = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            new_h, new_w = h, w
            im = image_rgb

        # Pad to make dimensions multiples of 32 (common requirement for ONNX models)
        pad_h = (32 - new_h % 32) % 32
        pad_w = (32 - new_w % 32) % 32
        im = cv2.copyMakeBorder(im, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        
        # Normalize
        im = (im.astype(np.float32) - 127.5) / 127.5
        im = np.transpose(np.expand_dims(im, 0), (0, 3, 1, 2))

        # Inference
        name = session.get_inputs()[0].name
        outputs = session.run(None, {name: im})
        
        # outputs[0] is a numpy array. Use np.asarray to satisfy type checkers 
        # (like Pyright) that complain about indexing into list[Unknown] or SparseTensor.
        matte_array = np.asarray(outputs[0])
        matte = matte_array[0, 0]

        # Remove padding from output
        matte = matte[:new_h, :new_w]
        
        # Scale to 0-255 and resize back to original
        matte = np.clip(matte * 255.0, 0, 255).astype(np.uint8)
        
        if new_h != h or new_w != w:
            matte = cv2.resize(matte, (w, h), interpolation=cv2.INTER_LINEAR)
            
        return matte
    except Exception as e:
        logger.error(f"MODNet inference failed: {e}")
        return None

def _remove_background(image_rgb: np.ndarray) -> Image.Image:
    """
    Remove background from an RGB numpy array.
    Returns RGBA PIL Image with transparent background.

    Pipeline: MODNet first → rembg u2net fallback.
    """
    # 1. Try MODNet
    matte = _run_modnet(image_rgb)

    if matte is not None:
        fg_ratio = np.count_nonzero(matte > 128) / matte.size
        logger.info(f"MODNet foreground ratio: {fg_ratio:.4f}")
        if fg_ratio >= 0.05:
            logger.info("Using MODNet result.")
            return Image.fromarray(np.dstack((image_rgb, matte)), "RGBA")

    # 2. Fallback to rembg u2net
    logger.info("Falling back to rembg (u2net).")
    try:
        session = _get_rembg_session()
        if session is False:
            logger.error("Skipping rembg fallback because session failed to load earlier.")
            return Image.fromarray(image_rgb).convert("RGBA")
            
        from rembg import remove
        pil_img = Image.fromarray(image_rgb)
        result = remove(pil_img, session=session)
        return result if isinstance(result, Image.Image) else Image.fromarray(image_rgb).convert("RGBA")
    except Exception as e:
        logger.error(f"rembg fallback failed: {e}")
        return Image.fromarray(image_rgb).convert("RGBA")


# ═══════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════

def _parse_bg_color(color_str: str) -> tuple[int, int, int]:
    """Convert color string/hex to RGB tuple."""
    s = color_str.lower().strip()
    words = s.split()

    if "red" in words:
        return BG_COLORS["red"]
    if "gray" in s or "grey" in s:
        return BG_COLORS["light-gray"]
    if "blue" in s:
        return BG_COLORS["light-blue"] if "light" in s else BG_COLORS["blue"]
    if any(w in s for w in ("white", "off-white", "cream", "light colored")):
        return BG_COLORS["white"]
    if s in BG_COLORS:
        return BG_COLORS[s]
    if s.startswith("#") and len(s) == 7:
        return (int(s[1:3], 16), int(s[3:5], 16), int(s[5:7], 16))

    logger.warning(f"Unknown bg color '{color_str}', defaulting to white")
    return (255, 255, 255)


def _sample_edge_pixels(image: np.ndarray) -> np.ndarray:
    """Sample pixels from image edges (corners + midpoints + top strip)."""
    h, w = image.shape[:2]
    margin = min(10, h // 10, w // 10)
    samples = []

    # 4 corners
    for cy, cx in [(0, 0), (0, w - margin), (h - margin, 0), (h - margin, w - margin)]:
        samples.append(image[cy:cy + margin, cx:cx + margin].reshape(-1, 3))

    # 4 border midpoints
    mh, mw = h // 2, w // 2
    for cy, cx in [(0, mw - margin // 2), (h - margin, mw - margin // 2),
                   (mh - margin // 2, 0), (mh - margin // 2, w - margin)]:
        cy, cx = max(0, min(cy, h - margin)), max(0, min(cx, w - margin))
        samples.append(image[cy:cy + margin, cx:cx + margin].reshape(-1, 3))

    # Top strip
    samples.append(image[0:min(5, h), :].reshape(-1, 3))
    return np.vstack(samples)


# ═══════════════════════════════════════════════════════════════
#  Public API  (used by routes.py and main.py)
# ═══════════════════════════════════════════════════════════════

@dataclass
class BackgroundValidationResult:
    """Result of background validation."""
    is_valid: bool
    avg_rgb: tuple[float, float, float]
    brightness: float
    rgb_deviation: float
    variance: float
    message: str


def validate_background(
    image: np.ndarray,
    target_color: str = "plain white",
    rgb_tolerance: int = 10,
    max_variance: float = 15.0,
) -> BackgroundValidationResult:
    """Check if the image background matches the target color."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    samples = _sample_edge_pixels(rgb)

    avg = samples.mean(axis=0)
    avg_r, avg_g, avg_b = float(avg[0]), float(avg[1]), float(avg[2])
    brightness = (avg_r + avg_g + avg_b) / 3.0
    variance = float(samples.std())
    target = _parse_bg_color(target_color)
    dev = max(abs(avg_r - target[0]), abs(avg_g - target[1]), abs(avg_b - target[2]))

    issues = []
    if dev > rgb_tolerance:
        issues.append(f"RGB deviation {dev:.1f} exceeds tolerance {rgb_tolerance}")

    target_brightness = sum(target) / 3.0
    if brightness < max(target_brightness - 30, 0):
        issues.append(f"Brightness {brightness:.0f} too low for '{target_color}'")

    if variance > max_variance:
        issues.append(f"Variance {variance:.1f} exceeds max {max_variance}")

    is_valid = len(issues) == 0
    logger.info(f"BG validation: valid={is_valid}, rgb=({avg_r:.0f},{avg_g:.0f},{avg_b:.0f}), dev={dev:.1f}")

    return BackgroundValidationResult(
        is_valid=is_valid,
        avg_rgb=(avg_r, avg_g, avg_b),
        brightness=brightness,
        rgb_deviation=dev,
        variance=variance,
        message="Background valid" if is_valid else "; ".join(issues),
    )


def correct_background(
    image: np.ndarray,
    target_color: str = "plain white",
) -> tuple[np.ndarray, bool]:
    """
    Remove background and replace with target color.

    Args:
        image: BGR numpy array
        target_color: target background color string

    Returns:
        (corrected_bgr_image, success) tuple
    """
    try:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Remove background → RGBA
        fg_rgba = _remove_background(rgb)

        # Composite onto target color
        target = _parse_bg_color(target_color)
        bg = Image.new("RGBA", fg_rgba.size, (*target, 255))
        bg.paste(fg_rgba, mask=fg_rgba.split()[3])

        # Convert back to BGR
        result = cv2.cvtColor(np.array(bg.convert("RGB")), cv2.COLOR_RGB2BGR)
        logger.info(f"Background replaced with {target_color}")
        return result, True

    except Exception as e:
        logger.error(f"Background correction failed: {e}")
        return image, False
