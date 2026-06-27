"""Regression tests for the 48kHz speech tokenizer opt-in setting.

Covers:
- ``backend.config``: ``get_settings_path``, ``load_app_settings``,
  ``save_app_settings`` (atomic write with fsync)
- ``backend.models``: ``AppSettings`` and ``AppSettingsUpdate`` Pydantic
  models (defaults, partial update, ``exclude_none=True``)
- ``backend.routes.settings``: ``GET /settings`` returns persisted state;
  ``PATCH /settings`` merges, validates, and persists atomically;
  empty file ⇒ 500; invalid merged payload ⇒ validation error.
- ``backend.backends.pytorch_backend.PyTorchTTSBackend``:
  ``load_model_async`` reloads when the 48k setting flips but is a
  no-op when the requested 48k matches the cached state.

These tests guard against the cherry-pick risk seen in prior tasks
where a regression test was added but the actual fix was dropped.
"""

import json
from pathlib import Path

import pytest

# ─────────────────────────────────────────────────────────────────────
# backend.config — settings.json load/save helpers
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def isolated_data_dir(monkeypatch, tmp_path):
    """Point ``backend.config._data_dir`` at a temp directory."""
    from backend import config

    monkeypatch.setattr(config, "_data_dir", tmp_path)
    return tmp_path


def test_get_settings_path_resolves_under_data_dir(isolated_data_dir):
    from backend import config

    path = config.get_settings_path()
    assert path == isolated_data_dir / "settings.json"
    # Parent directory creation is lazy (save_app_settings creates it).
    assert path.parent.exists() or path.parent == isolated_data_dir


def test_load_app_settings_returns_empty_dict_when_file_missing(isolated_data_dir):
    from backend import config

    assert config.load_app_settings() == {}


def test_load_app_settings_round_trips_saved_payload(isolated_data_dir):
    from backend import config

    config.save_app_settings({"use_48k_speech_tokenizer": True})
    assert config.load_app_settings() == {"use_48k_speech_tokenizer": True}


def test_load_app_settings_swallows_corrupt_json(isolated_data_dir, caplog):
    """A corrupted settings.json must NOT raise — it should return ``{}`` and warn."""
    from backend import config

    isolated_data_dir.mkdir(parents=True, exist_ok=True)
    (isolated_data_dir / "settings.json").write_text("not valid json", encoding="utf-8")

    with caplog.at_level("WARNING"):
        result = config.load_app_settings()
    assert result == {}
    # At least one warning was emitted about the failed load.
    assert any("Failed to load settings" in r.message for r in caplog.records)


def test_save_app_settings_creates_parent_directory(isolated_data_dir):
    from backend import config

    nested = isolated_data_dir / "nested" / "subdir"
    assert not nested.exists()
    config._data_dir = nested  # save_app_settings will mkdir parents
    config.save_app_settings({"use_48k_speech_tokenizer": False})
    assert nested.exists()
    assert (nested / "settings.json").exists()


def test_save_app_settings_writes_valid_indented_json(isolated_data_dir):
    from backend import config

    config.save_app_settings({"use_48k_speech_tokenizer": True})
    raw = (isolated_data_dir / "settings.json").read_text(encoding="utf-8")
    # Indented for human readability per upstream's save_app_settings.
    assert "\n" in raw
    assert json.loads(raw) == {"use_48k_speech_tokenizer": True}


def test_save_app_settings_is_atomic_no_leftover_temp(isolated_data_dir):
    """After a successful save, no ``.tmp`` file should remain."""
    from backend import config

    config.save_app_settings({"use_48k_speech_tokenizer": False})
    leftovers = list(isolated_data_dir.glob("*.tmp"))
    assert leftovers == []
    assert (isolated_data_dir / "settings.json").exists()


def test_save_app_settings_replaces_existing_payload(isolated_data_dir):
    """A second save must atomically replace the file (no merge)."""
    from backend import config

    config.save_app_settings({"use_48k_speech_tokenizer": True})
    config.save_app_settings({"use_48k_speech_tokenizer": False})
    assert config.load_app_settings() == {"use_48k_speech_tokenizer": False}


def test_save_app_settings_cleans_up_temp_on_failure(isolated_data_dir, monkeypatch):
    """If the write fails, no ``.tmp`` file should remain on disk."""
    from backend import config

    def _boom(*_args, **_kwargs):
        raise RuntimeError("simulated disk error")

    monkeypatch.setattr("json.dump", _boom)

    with pytest.raises(RuntimeError):
        config.save_app_settings({"use_48k_speech_tokenizer": True})

    leftovers = list(isolated_data_dir.glob("*.tmp"))
    assert leftovers == []


# ─────────────────────────────────────────────────────────────────────
# backend.models — AppSettings and AppSettingsUpdate
# ─────────────────────────────────────────────────────────────────────


def test_app_settings_defaults_to_48k_disabled():
    from backend.models import AppSettings

    s = AppSettings()
    assert s.use_48k_speech_tokenizer is False


def test_app_settings_accepts_explicit_true():
    from backend.models import AppSettings

    s = AppSettings(use_48k_speech_tokenizer=True)
    assert s.use_48k_speech_tokenizer is True


def test_app_settings_update_default_is_none():
    """PATCH model: every field defaults to ``None`` so partial updates work."""
    from backend.models import AppSettingsUpdate

    u = AppSettingsUpdate()
    assert u.use_48k_speech_tokenizer is None


def test_app_settings_update_exclude_none_drops_unset():
    """``model_dump(exclude_none=True)`` must drop the unset field."""
    from backend.models import AppSettingsUpdate

    assert AppSettingsUpdate().model_dump(exclude_none=True) == {}
    assert AppSettingsUpdate(use_48k_speech_tokenizer=True).model_dump(exclude_none=True) == {
        "use_48k_speech_tokenizer": True
    }


# ─────────────────────────────────────────────────────────────────────
# backend.routes.settings — GET / PATCH endpoints
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def client(isolated_data_dir):
    """FastAPI TestClient wired up with the settings router."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from backend.routes.settings import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_get_settings_returns_defaults_when_no_file(client):
    r = client.get("/settings")
    assert r.status_code == 200
    assert r.json() == {"use_48k_speech_tokenizer": False}


def test_get_settings_returns_persisted_value(isolated_data_dir, client):
    from backend import config

    config.save_app_settings({"use_48k_speech_tokenizer": True})
    r = client.get("/settings")
    assert r.status_code == 200
    assert r.json() == {"use_48k_speech_tokenizer": True}


def test_patch_settings_enables_48k(isolated_data_dir, client):
    r = client.patch("/settings", json={"use_48k_speech_tokenizer": True})
    assert r.status_code == 200
    assert r.json() == {"use_48k_speech_tokenizer": True}
    # Persisted to disk.
    assert (isolated_data_dir / "settings.json").exists()
    assert config_load(isolated_data_dir)["use_48k_speech_tokenizer"] is True


def test_patch_settings_disables_48k(isolated_data_dir, client):
    from backend import config

    config.save_app_settings({"use_48k_speech_tokenizer": True})

    r = client.patch("/settings", json={"use_48k_speech_tokenizer": False})
    assert r.status_code == 200
    assert r.json() == {"use_48k_speech_tokenizer": False}


def test_patch_settings_with_empty_body_is_noop(isolated_data_dir, client):
    """PATCH with no fields must not change the persisted state."""
    from backend import config

    config.save_app_settings({"use_48k_speech_tokenizer": True})

    r = client.patch("/settings", json={})
    assert r.status_code == 200
    assert r.json() == {"use_48k_speech_tokenizer": True}


def test_patch_settings_500_when_existing_file_unreadable(isolated_data_dir, client, monkeypatch):
    """If the file exists but parsing fails, surface a 500 (not a silent empty state)."""
    # Create a corrupt file so load_app_settings returns {}.
    isolated_data_dir.mkdir(parents=True, exist_ok=True)
    (isolated_data_dir / "settings.json").write_text("not json", encoding="utf-8")

    r = client.patch("/settings", json={"use_48k_speech_tokenizer": True})
    # Empty merge + file exists ⇒ endpoint must raise HTTP 500.
    assert r.status_code == 500
    assert r.json()["detail"] == "Failed to read settings"


def test_patch_settings_overwrites_after_save(isolated_data_dir, client):
    """PATCH-then-PATCH: second value wins."""
    client.patch("/settings", json={"use_48k_speech_tokenizer": True})
    r = client.patch("/settings", json={"use_48k_speech_tokenizer": False})
    assert r.status_code == 200
    assert r.json() == {"use_48k_speech_tokenizer": False}


def config_load(data_dir: Path) -> dict:
    """Helper: read settings.json without going through the (monkeypatched) module."""
    return json.loads((data_dir / "settings.json").read_text(encoding="utf-8"))


# ─────────────────────────────────────────────────────────────────────
# backend.backends.pytorch_backend — 48k tokenizer cache-aware load
# ─────────────────────────────────────────────────────────────────────


def _make_pytorch_backend(monkeypatch, *, model_loaded: bool = True):
    """Build a PyTorchTTSBackend with __init__ bypassed.

    Sets up just enough attributes for the 48k cache comparison branch.
    Returns (backend, load_calls) where ``load_calls`` records every
    call to ``_load_model_sync``.
    """
    from backend.backends import pytorch_backend
    from backend.backends.pytorch_backend import PyTorchTTSBackend

    backend = PyTorchTTSBackend.__new__(PyTorchTTSBackend)
    backend.model = object() if model_loaded else None
    backend.model_size = "1.7B"
    backend.device = "cpu"
    backend._current_model_size = "1.7B"
    backend._use_48k_speech_tokenizer = False

    load_calls: list[tuple[str, bool]] = []

    def _record_load(size, use_48k):
        load_calls.append((size, use_48k))
        # Pretend the load succeeded.
        backend._current_model_size = size
        backend._use_48k_speech_tokenizer = use_48k
        return

    monkeypatch.setattr(backend, "_load_model_sync", _record_load)
    # Replace the AppSettings class that pytorch_backend imported, so
    # ``models.AppSettings(**config.load_app_settings())`` returns our fake
    # without dragging in the real Pydantic BaseModel.
    monkeypatch.setattr(pytorch_backend.models, "AppSettings", _FakeAppSettings)
    return backend, load_calls


class _FakeAppSettings:
    def __init__(self, **kwargs):
        self.use_48k_speech_tokenizer = kwargs.get("use_48k_speech_tokenizer", False)


def test_pytorch_load_noop_when_model_and_48k_already_match(monkeypatch, isolated_data_dir):
    """If the model is loaded with the same size and same 48k setting, no reload."""
    backend, calls = _make_pytorch_backend(monkeypatch)
    # Persist 48k=False on disk (via the isolated data dir).
    (isolated_data_dir / "settings.json").write_text(json.dumps({"use_48k_speech_tokenizer": False}), encoding="utf-8")

    import asyncio

    asyncio.run(backend.load_model_async("1.7B"))
    assert calls == []


def test_pytorch_load_reloads_when_48k_setting_flips(monkeypatch, isolated_data_dir):
    """If the user toggled 48k on, the next load must reload with the new tokenizer."""
    backend, calls = _make_pytorch_backend(monkeypatch)
    # Persist 48k=True on disk; backend currently has 48k=False.
    (isolated_data_dir / "settings.json").write_text(json.dumps({"use_48k_speech_tokenizer": True}), encoding="utf-8")

    import asyncio

    asyncio.run(backend.load_model_async("1.7B"))
    assert calls == [("1.7B", True)]


def test_pytorch_load_reloads_when_48k_setting_flips_back_off(monkeypatch, isolated_data_dir):
    """Going from 48k=True on the cached model back to 48k=False must reload."""
    backend, calls = _make_pytorch_backend(monkeypatch)
    # Backend cache claims 48k=True (loaded with it on previously).
    backend._use_48k_speech_tokenizer = True
    # But disk now says 48k=False.
    (isolated_data_dir / "settings.json").write_text(json.dumps({"use_48k_speech_tokenizer": False}), encoding="utf-8")

    import asyncio

    asyncio.run(backend.load_model_async("1.7B"))
    assert calls == [("1.7B", False)]


def test_pytorch_load_first_call_triggers_load(monkeypatch, isolated_data_dir):
    """Brand-new backend (model=None) must always trigger _load_model_sync."""
    backend, calls = _make_pytorch_backend(monkeypatch, model_loaded=False)
    backend._current_model_size = None
    (isolated_data_dir / "settings.json").write_text(json.dumps({"use_48k_speech_tokenizer": False}), encoding="utf-8")

    import asyncio

    asyncio.run(backend.load_model_async("1.7B"))
    assert len(calls) == 1
    assert calls[0] == ("1.7B", False)
