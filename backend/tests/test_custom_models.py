"""Tests for the custom HuggingFace TTS model CRUD module.

Covers:
- Add / list / get / delete lifecycle
- HF repo ID regex validation (path-traversal prevention)
- Slug generation (filesystem-safe)
- Duplicate model detection
- Corrupt JSON recovery (backup + re-raise)
- Atomic write temp-file cleanup on failure
- Thread-safe lock serialisation
"""

import json
import sys
import threading
from pathlib import Path

import pytest

# Add backend parent so `from backend import ...` works when run from repo root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend import config, custom_models

# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def tmp_custom_models_file(monkeypatch, tmp_path):
    """Redirect custom_models.json to a temporary data directory."""
    monkeypatch.setattr(config, "get_data_dir", lambda: tmp_path)
    return tmp_path


# ── Add / List / Get / Delete ─────────────────────────────────────────


def test_add_then_list(tmp_custom_models_file):
    """Adding a model makes it appear in list_custom_models."""
    entry = custom_models.add_custom_model("org/my-tts-model", display_name="My Model")
    assert entry["id"] == "org--my-tts-model"
    assert entry["hf_repo_id"] == "org/my-tts-model"
    assert entry["display_name"] == "My Model"

    models = custom_models.list_custom_models()
    assert len(models) == 1
    assert models[0]["id"] == "org--my-tts-model"


def test_add_defaults_display_name_to_repo_id(tmp_custom_models_file):
    """If no display_name is given, it defaults to the hf_repo_id."""
    entry = custom_models.add_custom_model("user/voice-v1")
    assert entry["display_name"] == "user/voice-v1"


def test_add_with_engine_hint(tmp_custom_models_file):
    """Engine hint is stored in the entry."""
    entry = custom_models.add_custom_model("org/model", engine="qwen")
    assert entry["engine"] == "qwen"


def test_get_custom_model_found(tmp_custom_models_file):
    """get_custom_model returns the entry by slug id."""
    custom_models.add_custom_model("org/my-model")
    result = custom_models.get_custom_model("org--my-model")
    assert result is not None
    assert result["hf_repo_id"] == "org/my-model"


def test_get_custom_model_not_found(tmp_custom_models_file):
    """get_custom_model returns None for unknown id."""
    assert custom_models.get_custom_model("nonexistent--model") is None


def test_delete_custom_model_success(tmp_custom_models_file):
    """delete_custom_model returns True and removes the entry."""
    custom_models.add_custom_model("org/my-model")
    assert custom_models.delete_custom_model("org--my-model") is True
    assert custom_models.list_custom_models() == []


def test_delete_custom_model_not_found(tmp_custom_models_file):
    """delete_custom_model returns False when the model doesn't exist."""
    assert custom_models.delete_custom_model("nonexistent--model") is False


def test_list_empty_when_no_file(tmp_custom_models_file):
    """list_custom_models returns [] when the JSON file doesn't exist yet."""
    assert custom_models.list_custom_models() == []


# ── Validation ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "repo_id",
    [
        "../etc/passwd",
        "no-slash",
        "has space/model",
        "model",
        "/model",
        "owner/",
        "a/b/c",
    ],
)
def test_invalid_repo_id_rejected(tmp_custom_models_file, repo_id):
    """Malformed repo IDs are rejected with ValueError."""
    with pytest.raises(ValueError, match="Invalid HuggingFace repo ID"):
        custom_models.add_custom_model(repo_id)


@pytest.mark.parametrize(
    "repo_id",
    [
        "org/model-name",
        "user123/my_model.v2",
        "a-b.c_d/e-f.g_h",
        "org/model.with.dots",
    ],
)
def test_valid_repo_id_accepted(tmp_custom_models_file, repo_id):
    """Valid repo IDs are accepted."""
    entry = custom_models.add_custom_model(repo_id)
    assert entry["id"] == repo_id.replace("/", "--")


def test_duplicate_repo_id_rejected(tmp_custom_models_file):
    """Adding the same repo ID twice raises ValueError."""
    custom_models.add_custom_model("org/my-model")
    with pytest.raises(ValueError, match="already registered"):
        custom_models.add_custom_model("org/my-model")


# ── Slug generation ───────────────────────────────────────────────────


def test_slug_replaces_slash_with_double_dash(tmp_custom_models_file):
    """The slug uses '--' to replace '/' for filesystem safety."""
    entry = custom_models.add_custom_model("myorg/my-model")
    assert "--" in entry["id"]
    assert "/" not in entry["id"]


# ── Corrupt JSON recovery ─────────────────────────────────────────────


def test_corrupt_json_backed_up_and_raises(tmp_custom_models_file):
    """A corrupt custom_models.json is backed up and the error re-raised."""
    path = custom_models._config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{broken json}", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        custom_models.list_custom_models()

    # Verify a backup file was created
    backups = list(path.parent.glob("custom_models.json.corrupt.*"))
    assert len(backups) == 1
    # Original file should have been renamed away
    assert not path.exists()


def test_corrupt_json_then_clean_state(tmp_custom_models_file):
    """After corrupt-JSON recovery, the file is gone and list returns []."""
    path = custom_models._config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        custom_models.list_custom_models()

    # After recovery, listing returns empty (file was renamed)
    assert custom_models.list_custom_models() == []


def test_non_list_json_treated_as_empty(tmp_custom_models_file):
    """If the JSON file contains a dict (not a list), it's treated as empty."""
    path = custom_models._config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"key": "value"}', encoding="utf-8")

    assert custom_models.list_custom_models() == []


# ── Atomic write ──────────────────────────────────────────────────────


def test_write_persists_to_disk(tmp_custom_models_file):
    """add_custom_model persists data that survives a fresh read."""
    custom_models.add_custom_model("org/model-a")
    custom_models.add_custom_model("org/model-b")

    # Re-read from disk
    path = custom_models._config_path()
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert len(raw) == 2
    assert {e["hf_repo_id"] for e in raw} == {"org/model-a", "org/model-b"}


def test_write_creates_parent_dir(tmp_path, monkeypatch):
    """_write creates the parent directory if it doesn't exist."""
    deep_dir = tmp_path / "deep" / "nested" / "data"
    monkeypatch.setattr(config, "get_data_dir", lambda: deep_dir)
    custom_models.add_custom_model("org/model")
    assert (deep_dir / "custom_models.json").exists()


def test_write_failure_cleans_up_tmp_file(tmp_custom_models_file, monkeypatch):
    """If os.replace fails, the temp file is removed and the error is re-raised."""
    def fake_replace(src, dst):
        raise OSError("simulated rename failure")

    monkeypatch.setattr(custom_models.os, "replace", fake_replace)

    with pytest.raises(OSError, match="simulated rename failure"):
        custom_models.add_custom_model("org/model")

    # No leftover .tmp files in the data dir (the except branch ran os.unlink)
    leftover = list(tmp_custom_models_file.glob("custom_models_*.tmp"))
    assert leftover == []


# ── Thread safety ────────────────────────────────────────────────────


def test_concurrent_adds_all_persisted(tmp_custom_models_file):
    """Concurrent adds from multiple threads all land on disk."""
    repo_ids = [f"org/model-{i}" for i in range(20)]
    threads = [threading.Thread(target=custom_models.add_custom_model, args=(rid,)) for rid in repo_ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    models = custom_models.list_custom_models()
    assert len(models) == 20
    ids = {m["id"] for m in models}
    for rid in repo_ids:
        assert rid.replace("/", "--") in ids
