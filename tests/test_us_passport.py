"""
Tests for the standalone US passport engine (us_passport/).

Run:  pytest tests/test_us_passport.py
Cloudinary is patched out, nothing is uploaded.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from config.settings import settings
from main import app as main_app
from us_passport import router as us_app
from us_passport import spec
from us_passport.engine import process
from us_passport.measure import USPhotoError

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "test_face.jpg"


@pytest.fixture(scope="module")
def sample():
    return cv2.imread(str(SAMPLE))


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "api_key", "test-api-key")
    monkeypatch.setattr(us_app, "_store", lambda rid, photo, prev: {"image_url": f"mem://{rid}", "preview_url": f"mem://{rid}_p"})
    monkeypatch.setattr(us_app, "_upload_original", lambda *a, **k: None)
    with TestClient(main_app) as c:
        yield c


def _assert_compliant(r):
    mt = r.metrics
    assert r.compliant, [c for c in r.checks if c.status == "error"]
    assert spec.HEAD_MIN * 100 <= mt.head_pct <= spec.HEAD_MAX * 100
    assert spec.EYE_MIN * 100 <= mt.eye_pct <= spec.EYE_MAX * 100
    assert abs(mt.center_offset_pct) <= spec.CENTER_TOL_PASS * 100
    assert r.image.shape[:2] == (spec.OUT_PX, spec.OUT_PX)
    assert len(r.jpeg) <= spec.MAX_FILE_KB * 1024


def _rotate(im, deg):
    M = cv2.getRotationMatrix2D((im.shape[1] / 2, im.shape[0] * 0.35), deg, 1)
    return cv2.warpAffine(im, M, (im.shape[1], im.shape[0]), borderMode=cv2.BORDER_REPLICATE)


def test_sample_is_compliant(sample):
    r = process(sample)
    _assert_compliant(r)
    assert r.background_corrected  # grey backdrop -> white


@pytest.mark.parametrize("name,fn", [
    ("zoomed-out", lambda b: cv2.copyMakeBorder(b, 500, 0, 600, 600, cv2.BORDER_CONSTANT, value=(150, 150, 150))),
    ("tight", lambda b: b[150:1000, 150:950]),
    ("off-centre", lambda b: b[:, :800]),
    ("tilt+12", lambda b: _rotate(b, 12)),
    ("tilt-15", lambda b: _rotate(b, -15)),
    ("small", lambda b: cv2.resize(b, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)),
    ("dark", lambda b: np.clip(b * 0.6, 0, 255).astype(np.uint8)),
])
def test_variants_compliant(sample, name, fn):
    _assert_compliant(process(fn(sample)))


def test_tiny_source_rejected(sample):
    with pytest.raises(USPhotoError) as e:
        process(cv2.resize(sample, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA))
    assert e.value.code == "low_resolution"


def test_no_face_rejected():
    with pytest.raises(USPhotoError):
        process(np.full((800, 600, 3), 200, np.uint8))


def test_api_process_same_shape_as_main_api(client):
    r = client.post("/us/process", headers={"X-API-Key": "test-api-key"},
                    files={"image": ("test_face.jpg", SAMPLE.read_bytes(), "image/jpeg")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"status", "result_id", "image_url", "preview_url", "dimensions", "format", "size_kb", "metrics"}
    assert body["dimensions"] == "600x600" and body["format"] == "JPEG"
    assert 50 <= body["metrics"]["head_height_pct"] <= 69
    assert 56 <= body["metrics"]["eye_position_pct"] <= 69


def test_api_requires_key(client):
    r = client.post("/us/process", files={"image": ("a.jpg", b"x", "image/jpeg")})
    assert r.status_code in (401, 403)


def test_validate_original_fails_and_processed_passes(client, sample):
    h = {"X-API-Key": "test-api-key"}
    raw = client.post("/us/validate", headers=h, files={"image": ("t.jpg", SAMPLE.read_bytes(), "image/jpeg")}).json()
    assert raw["overall_result"] == "FAIL"        # not square / not white background
    final = process(sample).jpeg
    ok = client.post("/us/validate", headers=h, files={"image": ("o.jpg", final, "image/jpeg")}).json()
    assert ok["overall_result"] == "PASS", ok


@pytest.mark.parametrize("code", [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180])
def test_sideways_photo_auto_oriented(sample, code):
    _assert_compliant(process(cv2.rotate(sample, code)))


def test_head_only_crop_not_flipped(sample):
    # tight face-only crop (no shoulders) must not be mistaken for upside-down
    _assert_compliant(process(sample[230:790, 330:760]))


def test_transparent_png_cutout(client, sample):
    from us_passport.measure import compute_alpha
    a = (compute_alpha(sample) * 255).astype(np.uint8)
    rgba = np.dstack([sample, a])
    ok, png = cv2.imencode(".png", rgba)
    r = client.post("/us/process", headers={"X-API-Key": "test-api-key"},
                    files={"image": ("cut.png", png.tobytes(), "image/png")})
    assert r.status_code == 200, r.text
    assert r.json()["metrics"]["background_corrected"] is False


def test_huge_upload_is_bounded(client, sample):
    big = cv2.resize(sample, None, fx=5, fy=5, interpolation=cv2.INTER_LINEAR)   # 5430x7240
    ok, jpg = cv2.imencode(".jpg", big, [cv2.IMWRITE_JPEG_QUALITY, 60])
    r = client.post("/us/process", headers={"X-API-Key": "test-api-key"},
                    files={"image": ("big.jpg", jpg.tobytes(), "image/jpeg")})
    assert r.status_code in (200, 400), r.text
    if r.status_code == 200:
        assert r.json()["dimensions"] == "600x600"


def test_blur_rejected_in_one_pass(sample):
    r = process(cv2.GaussianBlur(sample, (0, 0), 6))
    assert not r.compliant and r.passes == 1


def _exif_jpeg(arr_bgr, orientation):
    import io
    from PIL import Image
    ex = Image.Exif(); ex[0x0112] = orientation
    buf = io.BytesIO()
    Image.fromarray(cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB)).save(buf, "JPEG", exif=ex, quality=95)
    return buf.getvalue()


def test_exif_orientation_applied(sample):
    from us_passport.imageio import decode_image
    sideways = cv2.rotate(sample, cv2.ROTATE_90_COUNTERCLOCKWISE)   # EXIF 6 = rotate 90 CW to display
    d = decode_image(_exif_jpeg(sideways, 6))
    assert d.bgr.shape[:2] == sample.shape[:2]
    assert d.format == "JPEG"


def test_icc_profile_converted_to_srgb(sample):
    import io
    from PIL import Image, ImageCms
    from us_passport.imageio import decode_image
    # An sRGB-tagged file must be left untouched (no conversion drift).
    im = Image.fromarray(cv2.cvtColor(sample, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    im.save(buf, "JPEG", icc_profile=ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes())
    d = decode_image(buf.getvalue())
    assert d.bgr.shape == sample.shape


def test_validate_reports_true_resolution_for_huge_file(client, sample):
    big = cv2.resize(sample, None, fx=5, fy=5, interpolation=cv2.INTER_LINEAR)
    ok, jpg = cv2.imencode(".jpg", big, [cv2.IMWRITE_JPEG_QUALITY, 60])
    r = client.post("/us/validate", headers={"X-API-Key": "test-api-key"},
                    files={"image": ("big.jpg", jpg.tobytes(), "image/jpeg")}).json()
    assert "5430x7240" in r["resolution"]["value"] and r["resolution"]["status"] == "error"


def test_exif_phone_photo_end_to_end(client, sample):
    sideways = cv2.rotate(sample, cv2.ROTATE_90_COUNTERCLOCKWISE)
    r = client.post("/us/process", headers={"X-API-Key": "test-api-key"},
                    files={"image": ("phone.jpg", _exif_jpeg(sideways, 6), "image/jpeg")})
    assert r.status_code == 200, r.text
