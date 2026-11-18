"""
Robust image decoding for uploads.

* applies EXIF orientation (phone photos)
* transparent pixels -> white (PNG cutouts) instead of black
* 16-bit / greyscale / palette / CMYK -> 8-bit RGB
* embedded non-sRGB ICC profile (Adobe RGB, Display P3...) -> sRGB
* caps the longest side to bound memory and latency
"""

import io
import logging
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageCms, ImageOps

logger = logging.getLogger(__name__)

MAX_SIDE_PX = 4096
_SRGB = None


@dataclass
class Decoded:
    bgr: np.ndarray
    orig_w: int          # size after EXIF rotation, before any downscale
    orig_h: int
    format: str          # normalised: JPEG / PNG / ...
    dpi: tuple | None


def _srgb_profile():
    global _SRGB
    if _SRGB is None:
        _SRGB = ImageCms.createProfile("sRGB")
    return _SRGB


def _to_srgb(im: Image.Image) -> Image.Image:
    icc = im.info.get("icc_profile")
    if not icc:
        return im
    try:
        src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        desc = (ImageCms.getProfileDescription(src) or "").lower()
        if "srgb" in desc:
            return im
        return ImageCms.profileToProfile(im, src, _srgb_profile(), outputMode="RGB")
    except Exception as e:                      # corrupt profile: keep pixels as-is
        logger.warning(f"ICC conversion skipped: {e}")
        return im


def decode_image(data: bytes) -> Decoded:
    try:
        im = Image.open(io.BytesIO(data))
        im.load() if im.format != "JPEG" else im.draft("RGB", (MAX_SIDE_PX * 2, MAX_SIDE_PX * 2))
        fmt = {"MPO": "JPEG", "JPG": "JPEG"}.get(im.format or "", im.format or "unknown")
        dpi = im.info.get("dpi")
        im = ImageOps.exif_transpose(im)
        if im.mode in ("I;16", "I;16B", "I;16L", "I"):
            arr = np.asarray(im, dtype=np.float32)
            im = Image.fromarray(np.clip(arr / (256.0 if arr.max() > 255 else 1.0), 0, 255).astype(np.uint8))
        if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, im).convert("RGB")
        elif im.mode == "CMYK":
            im = ImageCms.profileToProfile(im, ImageCms.ImageCmsProfile(io.BytesIO(im.info["icc_profile"])),
                                           _srgb_profile(), outputMode="RGB") if im.info.get("icc_profile") else im.convert("RGB")
        elif im.mode != "RGB":
            im = im.convert("RGB")
        im = _to_srgb(im)
        ow, oh = im.size
        arr = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise ValueError(f"Invalid image file: {e}") from e

    longest = max(arr.shape[:2])
    if longest > MAX_SIDE_PX:
        k = MAX_SIDE_PX / longest
        arr = cv2.resize(arr, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    return Decoded(arr, ow, oh, fmt, dpi)
