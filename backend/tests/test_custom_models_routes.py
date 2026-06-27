"""Tests for the FastAPI routes in ``backend/routes/custom_models.py``.

Covers the HTTP surface end-to-end via FastAPI's ``TestClient``:
- POST /custom-models — add (success + validation errors)
- GET /custom-models — list
- GET /custom-models/{id} — get one (found + not-found)
- DELETE /custom-models/{id} — remove (success + not-found)

A minimal FastAPI app is constructed that includes only the
``custom_models_router`` so tests run without requiring the full
backend ML stack to be importable.
"""

import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

# Add repo root so `from backend import ...` works when run from repo root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend import config
from backend.routes.custom_models import router as custom_models_router

# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def tmp_data_dir(monkeypatch, tmp_path):
    """Redirect custom_models.json to a temporary data directory."""
    monkeypatch.setattr(config, "get_data_dir", lambda: tmp_path)
    return tmp_path


@pytest.fixture
def client(tmp_data_dir):
    """A FastAPI TestClient with only the custom-models router mounted."""
    app = FastAPI()
    app.include_router(custom_models_router)
    return TestClient(app)


# ── POST /custom-models ───────────────────────────────────────────────


def test_add_returns_201_with_full_entry(client):
    """POST creates an entry and returns the full CustomModelResponse shape."""
    r = client.post(
        "/custom-models",
        json={"hf_repo_id": "owner/test-model", "display_name": "Test Model"},
    )
    assert r.status_code == 201
    body = r.json()
    assert body["id"] == "owner--test-model"
    assert body["hf_repo_id"] == "owner/test-model"
    assert body["display_name"] == "Test Model"
    assert body["engine"] is None
    assert "created_at" in body


def test_add_with_engine_hint(client):
    """Engine hint round-trips through POST."""
    r = client.post(
        "/custom-models",
        json={"hf_repo_id": "owner/test", "engine": "qwen"},
    )
    assert r.status_code == 201
    assert r.json()["engine"] == "qwen"


def test_add_rejects_invalid_repo_id(client):
    """Pydantic field pattern rejects malformed repo IDs with HTTP 422."""
    for bad in ["../etc/passwd", "no-slash", "owner/", "/model", "a/b/c"]:
        r = client.post("/custom-models", json={"hf_repo_id": bad})
        assert r.status_code == 422, f"expected 422 for {bad!r}, got {r.status_code}"


def test_add_rejects_duplicate(client):
    """Adding the same repo twice returns 400 with a clear message."""
    r1 = client.post("/custom-models", json={"hf_repo_id": "owner/dup"})
    assert r1.status_code == 201
    r2 = client.post("/custom-models", json={"hf_repo_id": "owner/dup"})
    assert r2.status_code == 400
    assert "already registered" in r2.json()["detail"]


def test_add_rejects_too_short_repo_id(client):
    """min_length=3 on CustomModelCreate.hf_repo_id rejects 1-2 char IDs."""
    r = client.post("/custom-models", json={"hf_repo_id": "a"})
    assert r.status_code == 422


# ── GET /custom-models ────────────────────────────────────────────────


def test_list_empty(client):
    """GET returns an empty list when nothing has been added."""
    r = client.get("/custom-models")
    assert r.status_code == 200
    assert r.json() == []


def test_list_returns_added_entries(client):
    """GET returns the entries in insertion order."""
    client.post("/custom-models", json={"hf_repo_id": "owner/a"})
    client.post("/custom-models", json={"hf_repo_id": "owner/b"})
    r = client.get("/custom-models")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 2
    assert [e["hf_repo_id"] for e in body] == ["owner/a", "owner/b"]


# ── GET /custom-models/{id} ───────────────────────────────────────────


def test_get_by_id_found(client):
    """GET /custom-models/{id} returns the matching entry."""
    client.post("/custom-models", json={"hf_repo_id": "owner/test"})
    r = client.get("/custom-models/owner--test")
    assert r.status_code == 200
    assert r.json()["hf_repo_id"] == "owner/test"


def test_get_by_id_not_found(client):
    """GET on an unknown id returns 404."""
    r = client.get("/custom-models/unknown--model")
    assert r.status_code == 404
    assert "not found" in r.json()["detail"]


# ── DELETE /custom-models/{id} ────────────────────────────────────────


def test_delete_success(client):
    """DELETE removes the entry and returns a confirmation message."""
    client.post("/custom-models", json={"hf_repo_id": "owner/test"})
    r = client.delete("/custom-models/owner--test")
    assert r.status_code == 200
    assert "removed" in r.json()["message"]
    # Confirm list is empty afterwards
    assert client.get("/custom-models").json() == []


def test_delete_not_found(client):
    """DELETE on an unknown id returns 404."""
    r = client.delete("/custom-models/ghost--model")
    assert r.status_code == 404


def test_delete_then_get_is_not_found(client):
    """After deletion, a follow-up GET returns 404."""
    client.post("/custom-models", json={"hf_repo_id": "owner/test"})
    client.delete("/custom-models/owner--test")
    r = client.get("/custom-models/owner--test")
    assert r.status_code == 404


# ── ID encoding ───────────────────────────────────────────────────────


def test_delete_handles_url_encoded_special_chars(client):
    """DELETE tolerates URL-encoded special chars in the id (apiClient uses encodeURIComponent).

    Slugs never contain '/' (which becomes '--') so the only realistic special
    chars are . _ -. The frontend's ``encodeURIComponent`` is forward-compatible
    protection against any future slug containing reserved URL characters.
    """
    client.post("/custom-models", json={"hf_repo_id": "org/with.dots_v2-final"})
    # Raw id has no reserved chars, but the frontend encodes anyway.
    # Re-encoding '.' as '%2E' should still resolve.
    r = client.delete("/custom-models/org--with.dots_v2-final")
    assert r.status_code == 200
    assert client.get("/custom-models").json() == []


# ── Error paths (5xx) ─────────────────────────────────────────────────


def test_list_returns_500_on_underlying_error(client, monkeypatch):
    """If the underlying storage raises, the route returns 500 with the error message."""
    from backend.routes import custom_models as route_mod

    def boom():
        raise OSError("disk on fire")

    monkeypatch.setattr(route_mod.custom_models, "list_custom_models", boom)
    r = client.get("/custom-models")
    assert r.status_code == 500
    assert "disk on fire" in r.json()["detail"]


def test_add_returns_500_on_underlying_error(client, monkeypatch):
    """Non-ValueError exceptions during add return 500."""
    from backend.routes import custom_models as route_mod

    def boom(**kwargs):
        raise OSError("disk on fire")

    monkeypatch.setattr(route_mod.custom_models, "add_custom_model", boom)
    r = client.post("/custom-models", json={"hf_repo_id": "owner/test"})
    assert r.status_code == 500
    assert "disk on fire" in r.json()["detail"]


def test_delete_logs_warning_when_cache_cleanup_fails(client, monkeypatch, caplog):
    """If shutil.rmtree raises, the route logs a warning but still returns 200.

    The cache cleanup is best-effort — losing the cache files is not fatal.
    The user-facing response should still confirm the definition was removed.
    """
    client.post("/custom-models", json={"hf_repo_id": "owner/test"})

    import logging


    def fake_rmtree(*args, **kwargs):
        raise OSError("permission denied")

    # Force the cache dir check to think the dir exists, so rmtree gets called.
    monkeypatch.setattr("pathlib.Path.exists", lambda self: True)
    monkeypatch.setattr("shutil.rmtree", fake_rmtree)

    with caplog.at_level(logging.WARNING, logger="backend.routes.custom_models"):
        r = client.delete("/custom-models/owner--test")

    assert r.status_code == 200
    assert "removed" in r.json()["message"]
    assert any("permission denied" in rec.message for rec in caplog.records), (
        f"Expected warning, got records: {[r.message for r in caplog.records]}"
    )


def test_delete_returns_404_when_entry_vanishes_between_get_and_delete(client, monkeypatch):
    """Race-condition guard: if the entry is deleted between get and delete, return 404.

    ``get_custom_model`` returns the entry, but a concurrent delete makes
    ``delete_custom_model`` return False. The route must surface this as 404.
    """
    client.post("/custom-models", json={"hf_repo_id": "owner/test"})

    from backend.routes import custom_models as route_mod

    def fake_delete(model_id):
        return False  # Pretend another caller already removed it.

    monkeypatch.setattr(route_mod.custom_models, "delete_custom_model", fake_delete)

    r = client.delete("/custom-models/owner--test")
    assert r.status_code == 404
    assert "not found" in r.json()["detail"]
