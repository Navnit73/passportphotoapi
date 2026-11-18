"""
US passport photo routes (independent engine in us_passport/).

Mounted in main.py under /us:
    POST /us/process   -> same response schema as /process
    POST /us/validate  -> same response schema as /validate
    GET  /us/requirements
"""

import logging
import secrets
import time
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Security, UploadFile
from fastapi.security import APIKeyHeader
from starlette.concurrency import run_in_threadpool

from api.models import (ErrorResponse, ProcessingMetrics,
                        ProcessResponse, ValidationItem, ValidationResponse)
from config.settings import settings

from . import spec
from .compliance import evaluate, overall, score
from .engine import process
from .imageio import Decoded, decode_image
from .measure import USPhotoError, measure
from .preview import create_preview

logger = logging.getLogger("us_passport")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)


async def verify_api_key(api_key: str = Security(api_key_header)):
    if not settings.api_key or not secrets.compare_digest(api_key.encode(), settings.api_key.encode()):
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key


router = APIRouter(tags=["US Passport"])

_ALLOWED = {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp", "image/tiff"}


async def _decode(image: UploadFile) -> tuple[bytes, Decoded]:
    if image.content_type not in _ALLOWED:
        raise HTTPException(400, f"Unsupported file type: {image.content_type}. Allowed: {', '.join(sorted(_ALLOWED))}")
    data = await image.read()
    if not data:
        raise HTTPException(400, "Empty image file")
    if len(data) > settings.max_upload_size_mb * 1024 * 1024:
        raise HTTPException(400, f"File too large. Max: {settings.max_upload_size_mb}MB")
    try:
        dec = await run_in_threadpool(decode_image, data)
    except ValueError:
        raise HTTPException(400, "Invalid image file")
    return data, dec


def _store(result_id: str, photo: bytes, preview: bytes) -> dict:
    """Cloudinary (primary); local results/ directory if Cloudinary is unavailable."""
    try:
        from storage.cloud import upload_results
        return upload_results(result_id, photo, preview)
    except Exception as e:
        logger.warning(f"Cloudinary upload failed ({e}); saving locally")
        settings.result_dir.mkdir(parents=True, exist_ok=True)
        (settings.result_dir / f"{result_id}_us.jpg").write_bytes(photo)
        (settings.result_dir / f"{result_id}_us_preview.jpg").write_bytes(preview)
        return {"image_url": f"results/{result_id}_us.jpg", "preview_url": f"results/{result_id}_us_preview.jpg"}


def _upload_original(data: bytes, name: str):
    try:
        from storage.cloud import upload_original
        upload_original(data, name)
    except Exception as e:
        logger.warning(f"Original upload skipped: {e}")


@router.post("/process", response_model=ProcessResponse,
          responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
          summary="Produce a US-compliant 600x600 passport photo")
async def process_us(background_tasks: BackgroundTasks, image: UploadFile = File(...),
                     api_key: str = Depends(verify_api_key)):
    t0 = time.time()
    data, dec = await _decode(image)
    background_tasks.add_task(_upload_original, data, image.filename or "photo.jpg")
    try:
        res = await run_in_threadpool(process, dec.bgr)
    except USPhotoError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error("Processing failed", exc_info=True)
        raise HTTPException(500, f"Processing failed: {e}")

    if not res.compliant:
        bad = "; ".join(f"{c.label}: {c.value} (need {c.target})" for c in res.checks if c.status == "error")
        raise HTTPException(400, f"Could not produce a compliant US photo — {bad}")

    preview = create_preview(res.image, res.measurement, res.metrics, res.compliant)
    result_id = str(uuid.uuid4())
    urls = await run_in_threadpool(_store, result_id, res.jpeg, preview)
    mt = res.metrics
    logger.info(f"US process done in {time.time()-t0:.2f}s passes={res.passes} head={mt.head_pct:.1f}% eye={mt.eye_pct:.1f}%")
    return ProcessResponse(
        status="success", result_id=result_id,
        image_url=urls.get("image_url", ""), preview_url=urls.get("preview_url"),
        dimensions=f"{spec.OUT_PX}x{spec.OUT_PX}", format="JPEG", size_kb=len(res.jpeg) // 1024,
        metrics=ProcessingMetrics(
            head_height_pct=round(mt.head_pct, 2),
            eye_position_pct=round(mt.eye_pct, 2),
            top_margin_pct=round(mt.top_margin_pct, 2),
            background_valid=True,
            background_corrected=res.background_corrected,
        ),
    )


def _item(checks, key, default_label):
    for c in checks:
        if c.key == key:
            return ValidationItem(label=c.label, value=c.value, status=c.status, details=c.details, target=c.target)
    return ValidationItem(label=default_label, value="n/a", status="warning")


def _validate(data: bytes, dec: Decoded) -> ValidationResponse:
    dpi = dec.dpi
    m = measure(dec.bgr)
    checks, mt = evaluate(dec.bgr, m, file_bytes=data, file_format=dec.format, dpi=dpi,
                          size=(dec.orig_w, dec.orig_h))
    fails = [c for c in checks if c.status == "error"]
    warns = [c for c in checks if c.status == "warning"]
    summary = [f"{c.label}: {c.value} — required {c.target}" for c in fails] or ["Photo meets all US requirements."]
    suggestions = [f"Fix {c.label.lower()} (required {c.target})" for c in fails + warns]
    if fails:
        suggestions.append("Use POST /us/process to auto-crop, align and fix the background.")
    return ValidationResponse(
        status="success", overall_result=overall(checks), score=score(checks),
        summary=summary, suggestions=suggestions,
        file_format=_item(checks, "format", "File format"), file_size=_item(checks, "size", "File size"),
        resolution=_item(checks, "resolution", "Resolution"),
        lighting=_item(checks, "lighting", "Lighting"),
        face_detection=_item(checks, "eyes", "Face detection"),
        eye_level=_item(checks, "eye", "Eye level"), head_size=_item(checks, "head", "Head size"),
        head_tilt=_item(checks, "tilt", "Head tilt"), orientation=_item(checks, "orientation", "Orientation"),
        dpi=_item(checks, "dpi", "DPI") if dpi else None,
    )


@router.post("/validate", response_model=ValidationResponse,
          summary="Check an already-prepared file against US rules (no processing)")
async def validate_us(image: UploadFile = File(...), api_key: str = Depends(verify_api_key)):
    data, dec = await _decode(image)
    try:
        return await run_in_threadpool(_validate, data, dec)
    except USPhotoError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error("Validation failed", exc_info=True)
        raise HTTPException(500, f"Validation failed: {e}")


@router.get("/requirements", summary="US photo requirements used by the engine")
async def requirements():
    return {k: getattr(spec, k) for k in dir(spec) if k.isupper()}
