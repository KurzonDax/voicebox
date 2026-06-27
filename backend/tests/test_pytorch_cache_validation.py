"""Regression tests for PyTorchSTTBackend._is_model_cached.

The Whisper cache validator must require all three config files
(config.json, preprocessor_config.json, tokenizer_config.json) to
exist in the snapshots directory, otherwise WhisperProcessor falls
back to remote download even though model weights are present.
"""

import sys
import types
from pathlib import Path

import pytest


def _make_hf_constants(hub_dir: Path) -> types.SimpleNamespace:
    return types.SimpleNamespace(HF_HUB_CACHE=str(hub_dir))


@pytest.fixture
def fake_hub_cache(tmp_path, monkeypatch):
    """Provide a writable HF_HUB_CACHE and a stub huggingface_hub.constants."""
    hub_dir = tmp_path / "hub"
    hub_dir.mkdir()
    constants = _make_hf_constants(hub_dir)
    fake_module = types.ModuleType("huggingface_hub.constants")
    fake_module.HF_HUB_CACHE = constants.HF_HUB_CACHE

    # Make sure the import inside _is_model_cached resolves to our stub even
    # if huggingface_hub is installed.
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.ModuleType("huggingface_hub"))
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", fake_module)
    monkeypatch.setattr("huggingface_hub.constants", fake_module, raising=False)
    return hub_dir


def _write_repo(hub_dir: Path, repo: str, files: dict[str, str]) -> Path:
    repo_cache = hub_dir / ("models--" + repo.replace("/", "--"))
    snapshots_dir = repo_cache / "snapshots" / "main"
    snapshots_dir.mkdir(parents=True)
    for fname, body in files.items():
        (snapshots_dir / fname).write_text(body)
    return snapshots_dir


def test_is_model_cached_requires_all_three_configs(tmp_path, monkeypatch, fake_hub_cache):
    """Missing preprocessor_config.json → False even with weights present.

    The original partial-cache bug: weights downloaded, processor config
    missing → ``is_model_cached`` returns True (weights found) but
    WhisperProcessor later fails to load ``preprocessor_config.json``.
    The post-fix ``_is_model_cached`` short-circuits to False so the
    caller knows it needs to (re)download.
    """
    from backend.backends.pytorch_backend import PyTorchSTTBackend

    backend = PyTorchSTTBackend.__new__(PyTorchSTTBackend)

    repo = "openai/whisper-base"
    _write_repo(fake_hub_cache, repo, {
        "config.json": "{}",
        "tokenizer_config.json": "{}",
        "model.safetensors": "fake-weights",
        # preprocessor_config.json deliberately absent
    })

    assert backend._is_model_cached("base") is False


def test_is_model_cached_returns_true_when_complete(tmp_path, monkeypatch, fake_hub_cache):
    """All three config files + weights present → True."""
    from backend.backends.pytorch_backend import PyTorchSTTBackend

    backend = PyTorchSTTBackend.__new__(PyTorchSTTBackend)

    repo = "openai/whisper-base"
    _write_repo(fake_hub_cache, repo, {
        "config.json": "{}",
        "preprocessor_config.json": "{}",
        "tokenizer_config.json": "{}",
        "model.safetensors": "fake-weights",
    })

    assert backend._is_model_cached("base") is True


def test_is_model_cached_false_when_snapshot_dir_missing(tmp_path, monkeypatch, fake_hub_cache):
    """No snapshot directory at all → False (base.is_model_cached returns False first)."""
    from backend.backends import pytorch_backend
    from backend.backends.pytorch_backend import PyTorchSTTBackend

    # Replace the module-level is_model_cached import to always return True
    # so we exercise the post-check (snapshot existence) branch in isolation.
    monkeypatch.setattr(pytorch_backend, "is_model_cached", lambda repo: True)

    backend = PyTorchSTTBackend.__new__(PyTorchSTTBackend)
    assert backend._is_model_cached("base") is False
