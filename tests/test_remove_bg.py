import io
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from config.settings import settings
from main import app
from services import background as bg_service


@pytest.fixture
def client(monkeypatch):
    if not settings.api_key:
        monkeypatch.setattr(settings, "api_key", "test-api-key")
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth_headers(client):
    return {"X-API-Key": settings.api_key}


def test_remove_bg_success_returns_transparent_png(client, auth_headers):
    sample_path = Path(__file__).resolve().parent.parent / "test_face.jpg"
    sample_bytes = sample_path.read_bytes()
    with Image.open(io.BytesIO(sample_bytes)) as src:
        expected_size = src.size

    response = client.post(
        "/remove-bg",
        headers=auth_headers,
        files={"image": ("my portrait (1).jpg", sample_bytes, "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["content-disposition"] == 'inline; filename="my_portrait__1_no_bg.png"'

    with Image.open(io.BytesIO(response.content)) as out_img:
        assert out_img.format == "PNG"
        assert out_img.mode == "RGBA"
        assert out_img.size == expected_size
        alpha = np.array(out_img)[:, :, 3]
        assert int(alpha.min()) == 0
        assert int(alpha.max()) == 255


def test_remove_bg_rembg_fallback_returns_transparent_png(client, auth_headers):
    sample_path = Path(__file__).resolve().parent.parent / "test_face.jpg"
    sample_bytes = sample_path.read_bytes()

    with patch.object(bg_service, "_run_modnet", return_value=None):
        response = client.post(
            "/remove-bg",
            headers=auth_headers,
            files={"image": ("face.jpg", sample_bytes, "image/jpeg")},
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    with Image.open(io.BytesIO(response.content)) as out_img:
        assert out_img.format == "PNG"
        assert out_img.mode == "RGBA"
        alpha = np.array(out_img)[:, :, 3]
        assert int(alpha.min()) < 128


def test_remove_bg_auth_required(client):
    sample_path = Path(__file__).resolve().parent.parent / "test_face.jpg"
    sample_bytes = sample_path.read_bytes()

    missing = client.post(
        "/remove-bg",
        files={"image": ("face.jpg", sample_bytes, "image/jpeg")},
    )
    assert missing.status_code in (401, 403)

    invalid = client.post(
        "/remove-bg",
        headers={"X-API-Key": "wrong-key"},
        files={"image": ("face.jpg", sample_bytes, "image/jpeg")},
    )
    assert invalid.status_code == 403


def test_remove_bg_rejects_invalid_inputs(client, auth_headers, monkeypatch):
    bad_type = client.post(
        "/remove-bg",
        headers=auth_headers,
        files={"image": ("notes.txt", b"hello", "text/plain")},
    )
    assert bad_type.status_code == 400
    assert "Unsupported file type" in bad_type.json()["detail"]

    empty_file = client.post(
        "/remove-bg",
        headers=auth_headers,
        files={"image": ("empty.jpg", b"", "image/jpeg")},
    )
    assert empty_file.status_code == 400
    assert empty_file.json()["detail"] == "Empty image file"

    corrupt_file = client.post(
        "/remove-bg",
        headers=auth_headers,
        files={"image": ("corrupt.jpg", b"not-a-valid-jpeg", "image/jpeg")},
    )
    assert corrupt_file.status_code == 400
    assert corrupt_file.json()["detail"] == "Invalid image file"

    monkeypatch.setattr(settings, "max_upload_size_mb", 0)
    oversized = client.post(
        "/remove-bg",
        headers=auth_headers,
        files={"image": ("small.png", b"1234", "image/png")},
    )
    assert oversized.status_code == 400
    assert "File too large" in oversized.json()["detail"]


def test_remove_bg_returns_500_when_both_models_fail(client, auth_headers):
    img = Image.new("RGB", (64, 64), (100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    with (
        patch.object(bg_service, "_run_modnet", return_value=None),
        patch.object(bg_service, "_get_rembg_session", return_value=False),
    ):
        response = client.post(
            "/remove-bg",
            headers=auth_headers,
            files={"image": ("square.png", buf.getvalue(), "image/png")},
        )

    assert response.status_code == 500
    assert "Background removal failed" in response.json()["detail"]


def test_correct_background_returns_false_when_both_models_fail():
    dummy_bgr = np.zeros((32, 32, 3), dtype=np.uint8)
    with (
        patch.object(bg_service, "_run_modnet", return_value=None),
        patch.object(bg_service, "_get_rembg_session", return_value=False),
    ):
        result, success = bg_service.correct_background(dummy_bgr)

    assert success is False
    assert np.array_equal(result, dummy_bgr)


def test_process_rejects_empty_and_unsupported_uploads(client, auth_headers):
    empty_file = client.post(
        "/process",
        headers=auth_headers,
        data={"country_code": "US", "document_type": "passport"},
        files={"image": ("empty.jpg", b"", "image/jpeg")},
    )
    assert empty_file.status_code == 400
    assert empty_file.json()["detail"] == "Empty image file"
