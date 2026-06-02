"""
Local file storage for uploads and processed results.
"""

import logging
import uuid
from pathlib import Path

from config.settings import settings

logger = logging.getLogger(__name__)


def _ensure_dirs():
    """Create upload and result directories if they don't exist."""
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.result_dir.mkdir(parents=True, exist_ok=True)


def save_upload(file_bytes: bytes, original_filename: str) -> str:
    """
    Save an uploaded file and return a unique upload ID.

    Args:
        file_bytes: raw file bytes
        original_filename: original filename for extension detection

    Returns:
        upload_id: unique identifier for the upload
    """
    _ensure_dirs()

    upload_id = str(uuid.uuid4())
    ext = Path(original_filename).suffix.lower() or ".jpg"
    filepath = settings.upload_dir / f"{upload_id}{ext}"

    filepath.write_bytes(file_bytes)
    logger.info(f"Saved upload: {filepath} ({len(file_bytes)} bytes)")

    return upload_id


def get_upload_path(upload_id: str) -> Path | None:
    """Find the upload file by ID (checks common extensions)."""
    _ensure_dirs()

    for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"]:
        path = settings.upload_dir / f"{upload_id}{ext}"
        if path.exists():
            return path

    return None


def save_result(
    result_id: str,
    photo_bytes: bytes,
    preview_bytes: bytes | None = None,
) -> dict:
    """
    Save processed result files.

    Args:
        result_id: unique result ID
        photo_bytes: processed photo JPEG bytes
        preview_bytes: optional preview JPEG bytes

    Returns:
        dict with file paths
    """
    _ensure_dirs()

    photo_path = settings.result_dir / f"{result_id}.jpg"
    photo_path.write_bytes(photo_bytes)

    paths = {"photo": str(photo_path)}

    if preview_bytes:
        preview_path = settings.result_dir / f"{result_id}_preview.jpg"
        preview_path.write_bytes(preview_bytes)
        paths["preview"] = str(preview_path)

    logger.info(f"Saved result: {result_id} ({len(photo_bytes)} bytes)")
    return paths


def get_result_path(result_id: str, file_type: str = "photo") -> Path | None:
    """
    Find a result file by ID.

    Args:
        result_id: result UUID
        file_type: "photo" or "print_sheet"

    Returns:
        Path or None
    """
    _ensure_dirs()

    if file_type == "preview":
        path = settings.result_dir / f"{result_id}_preview.jpg"
    else:
        path = settings.result_dir / f"{result_id}.jpg"

    return path if path.exists() else None
