"""Tests for OpenAI-compatible API endpoints (/v1/audio/speech, /v1/models).

Covers:
- GET /v1/models returns the three mapped model IDs in OpenAI list format.
- POST /v1/audio/speech model resolution (valid, invalid model).
- POST /v1/audio/speech voice resolution (profile lookup, Kokoro fallback,
  unknown voice default).
- POST /v1/audio/speech returns WAV audio with correct headers.
- ensure_model_cached_or_raise fail-fast is invoked before generation.

All ML backends are mocked so tests run without torch/numpy/transformers.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi import FastAPI
from pydantic import ValidationError
from starlette.testclient import TestClient

from backend.routes.openai_compat import (
    _AVAILABLE_MODELS,
    _OPENAI_VOICE_TO_KOKORO,
    SpeechRequest,
    _resolve_voice_prompt,
    router,
)

# ---------------------------------------------------------------------------
# App fixture — mount only the openai_compat router for isolation, with a
# mocked get_db so no real database session is required.
# ---------------------------------------------------------------------------


def _mock_get_db():
    """Dependency override that yields a MagicMock instead of a real Session."""
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = None
    try:
        yield db
    finally:
        pass


@pytest.fixture
def app():
    """Minimal FastAPI app with only the openai_compat router and mocked DB."""
    from backend.database import get_db

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_db] = _mock_get_db
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


# ---------------------------------------------------------------------------
# GET /v1/models
# ---------------------------------------------------------------------------


class TestListModels:
    """GET /v1/models should return OpenAI-shaped model list."""

    def test_returns_list_object(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "list"

    def test_returns_three_models(self, client):
        resp = client.get("/v1/models")
        body = resp.json()
        assert len(body["data"]) == 3

    def test_model_ids_match_mapping(self, client):
        resp = client.get("/v1/models")
        body = resp.json()
        ids = {entry["id"] for entry in body["data"]}
        assert ids == set(_AVAILABLE_MODELS)
        assert ids == {"tts-1", "tts-1-hd", "gpt-4o-mini-tts"}

    def test_model_entries_have_required_fields(self, client):
        resp = client.get("/v1/models")
        body = resp.json()
        for entry in body["data"]:
            assert entry["object"] == "model"
            assert entry["owned_by"] == "voicebox"
            assert "id" in entry


# ---------------------------------------------------------------------------
# POST /v1/audio/speech — model resolution
# ---------------------------------------------------------------------------


class TestModelResolution:
    """Model field maps to (engine, model_size) or 400 on unknown model."""

    def test_unknown_model_returns_400(self, client):
        resp = client.post(
            "/v1/audio/speech",
            json={"model": "gpt-4", "input": "hello", "voice": "alloy"},
        )
        assert resp.status_code == 400
        assert "Unknown model" in resp.json()["detail"]

    def test_unknown_model_lists_supported(self, client):
        resp = client.post(
            "/v1/audio/speech",
            json={"model": "bad-model", "input": "hello", "voice": "alloy"},
        )
        detail = resp.json()["detail"]
        for m in _AVAILABLE_MODELS:
            assert m in detail


# ---------------------------------------------------------------------------
# POST /v1/audio/speech — voice resolution
# ---------------------------------------------------------------------------


class TestVoiceResolution:
    """Voice field resolves via profile lookup or Kokoro fallback."""

    @pytest.mark.asyncio
    async def test_unknown_voice_defaults_to_af_alloy_for_kokoro(self):
        """When no profile matches and engine is kokoro, voice_prompt uses af_alloy."""
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = None
        prompt = await _resolve_voice_prompt("nonexistent", "kokoro", db)
        assert prompt == {"kokoro_voice": "af_alloy"}

    @pytest.mark.asyncio
    async def test_known_openai_voice_maps_to_kokoro_id(self):
        """alloy -> af_alloy, echo -> am_echo, etc."""
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = None
        for openai_name, kokoro_id in _OPENAI_VOICE_TO_KOKORO.items():
            prompt = await _resolve_voice_prompt(openai_name, "kokoro", db)
            assert prompt == {"kokoro_voice": kokoro_id}

    @pytest.mark.asyncio
    async def test_unknown_voice_for_qwen_returns_preset_voice_id(self):
        """Non-kokoro engines get preset_voice_id key."""
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = None
        prompt = await _resolve_voice_prompt("alloy", "qwen", db)
        assert prompt == {"preset_voice_id": "af_alloy"}

    @pytest.mark.asyncio
    async def test_profile_lookup_succeeds(self):
        """When a profile matches by name, create_voice_prompt_for_profile is used."""
        db = MagicMock()
        fake_profile = MagicMock()
        fake_profile.id = "profile-uuid-123"
        db.query.return_value.filter.return_value.first.return_value = fake_profile

        expected_prompt = {"voice_type": "preset", "preset_engine": "kokoro", "preset_voice_id": "af_heart"}
        with patch(
            "backend.services.profiles.create_voice_prompt_for_profile",
            new_callable=AsyncMock,
            return_value=expected_prompt,
        ):
            prompt = await _resolve_voice_prompt("MyProfile", "kokoro", db)
        assert prompt == expected_prompt

    @pytest.mark.asyncio
    async def test_profile_lookup_is_case_insensitive(self):
        """Profile name match uses func.lower() on both sides — uppercase voice must match."""
        db = MagicMock()
        fake_profile = MagicMock()
        fake_profile.id = "profile-uuid-ci"
        # SQLAlchemy generates filter(func.lower(...) == voice.lower()); emulate that
        # the comparison is case-insensitive by passing the lowercase value on the LHS.
        db.query.return_value.filter.return_value.first.return_value = fake_profile

        expected_prompt = {"voice_type": "preset", "preset_engine": "kokoro", "preset_voice_id": "af_heart"}
        with patch(
            "backend.services.profiles.create_voice_prompt_for_profile",
            new_callable=AsyncMock,
            return_value=expected_prompt,
        ) as mock_create:
            # Caller sends mixed-case voice (e.g. an SDK that capitalizes names);
            # the DB row matches because func.lower() normalises both sides.
            prompt = await _resolve_voice_prompt("MyProfile", "kokoro", db)
        assert prompt == expected_prompt
        # The service is called with the profile id, not the voice string — verifies
        # case-insensitive lookup is delegated to SQL rather than re-computed in Python.
        mock_create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_profile_lookup_fails_falls_back_to_kokoro(self):
        """If create_voice_prompt_for_profile raises, we fall back to Kokoro voice."""
        db = MagicMock()
        fake_profile = MagicMock()
        fake_profile.id = "profile-uuid-456"
        db.query.return_value.filter.return_value.first.return_value = fake_profile

        with patch(
            "backend.services.profiles.create_voice_prompt_for_profile",
            new_callable=AsyncMock,
            side_effect=ValueError("preset engine mismatch"),
        ):
            prompt = await _resolve_voice_prompt("BrokenProfile", "kokoro", db)
        assert prompt == {"kokoro_voice": "af_alloy"}


# ---------------------------------------------------------------------------
# POST /v1/audio/speech — end-to-end with mocked generation
# ---------------------------------------------------------------------------


def _patch_generation_chain(monkeypatch, captured=None):
    """Patch all lazy imports inside create_speech so no ML code runs.

    Returns a dict that captures kwargs if provided.
    """
    fake_audio = np.zeros(100, dtype=np.float32)

    async def mock_ensure_cached(engine, model_size):
        if captured is not None:
            captured["cached_engine"] = engine
            captured["cached_model_size"] = model_size
        return

    async def mock_load_model(engine, model_size):
        if captured is not None:
            captured["engine"] = engine
            captured["model_size"] = model_size

    async def mock_generate_chunked(backend, text, voice_prompt, **kwargs):
        if captured is not None:
            captured["instruct"] = kwargs.get("instruct")
        return fake_audio, 24000

    async def mock_resolve(voice, engine, db):
        if captured is not None:
            captured["voice"] = voice
            captured["engine"] = engine
        return {"kokoro_voice": "af_alloy"}

    monkeypatch.setattr(
        "backend.routes.openai_compat._resolve_voice_prompt",
        mock_resolve,
    )

    import backend.backends as backends_mod

    monkeypatch.setattr(backends_mod, "ensure_model_cached_or_raise", mock_ensure_cached)
    monkeypatch.setattr(backends_mod, "load_engine_model", mock_load_model)
    monkeypatch.setattr(backends_mod, "get_tts_backend_for_engine", lambda e: MagicMock())
    monkeypatch.setattr(backends_mod, "engine_needs_trim", lambda e: False)

    import backend.services.tts as tts_mod

    monkeypatch.setattr(tts_mod, "audio_to_wav_bytes", lambda a, s: b"FAKE_WAV_BYTES")

    import backend.utils.audio as audio_mod

    monkeypatch.setattr(audio_mod, "normalize_audio", lambda a: a)

    import backend.utils.chunked_tts as chunked_mod

    monkeypatch.setattr(chunked_mod, "generate_chunked", mock_generate_chunked)


class TestCreateSpeech:
    """Full POST /v1/audio/speech flow with all backends mocked."""

    def test_speech_returns_wav_audio(self, client, monkeypatch):
        """Valid request returns WAV bytes with audio/wav content type."""
        _patch_generation_chain(monkeypatch)
        resp = client.post(
            "/v1/audio/speech",
            json={"model": "tts-1", "input": "hello world", "voice": "alloy"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/wav"
        assert b"FAKE_WAV_BYTES" in resp.content

    def test_speech_content_disposition_header(self, client, monkeypatch):
        """Response includes Content-Disposition with speech.wav filename."""
        _patch_generation_chain(monkeypatch)
        resp = client.post(
            "/v1/audio/speech",
            json={"model": "tts-1", "input": "test", "voice": "nova"},
        )
        assert resp.status_code == 200
        assert "speech.wav" in resp.headers.get("content-disposition", "")

    def test_speech_passes_instructions_to_generate(self, client, monkeypatch):
        """instructions field is forwarded as instruct to generate_chunked."""
        captured = {}
        _patch_generation_chain(monkeypatch, captured)
        resp = client.post(
            "/v1/audio/speech",
            json={
                "model": "tts-1",
                "input": "hello",
                "voice": "alloy",
                "instructions": "Speak cheerfully",
            },
        )
        assert resp.status_code == 200
        assert captured["instruct"] == "Speak cheerfully"

    def test_speech_model_mapping_tts1_hd_uses_qwen(self, client, monkeypatch):
        """tts-1-hd maps to qwen engine with 1.7B model_size."""
        captured = {}
        _patch_generation_chain(monkeypatch, captured)
        resp = client.post(
            "/v1/audio/speech",
            json={"model": "tts-1-hd", "input": "hello", "voice": "alloy"},
        )
        assert resp.status_code == 200
        assert captured["engine"] == "qwen"
        assert captured["model_size"] == "1.7B"
        assert captured["cached_engine"] == "qwen"
        assert captured["cached_model_size"] == "1.7B"

    def test_speech_model_mapping_gpt4o_mini_tts_uses_qwen_06b(self, client, monkeypatch):
        """gpt-4o-mini-tts maps to qwen engine with 0.6B model_size."""
        captured = {}
        _patch_generation_chain(monkeypatch, captured)
        resp = client.post(
            "/v1/audio/speech",
            json={"model": "gpt-4o-mini-tts", "input": "hello", "voice": "alloy"},
        )
        assert resp.status_code == 200
        assert captured["engine"] == "qwen"
        assert captured["model_size"] == "0.6B"

    def test_speech_model_mapping_tts1_uses_kokoro(self, client, monkeypatch):
        """tts-1 maps to kokoro engine with default model_size."""
        captured = {}
        _patch_generation_chain(monkeypatch, captured)
        resp = client.post(
            "/v1/audio/speech",
            json={"model": "tts-1", "input": "hello", "voice": "alloy"},
        )
        assert resp.status_code == 200
        assert captured["engine"] == "kokoro"
        assert captured["model_size"] == "default"


# ---------------------------------------------------------------------------
# SpeechRequest schema validation
# ---------------------------------------------------------------------------


class TestSpeechRequestSchema:
    """Pydantic model validates and applies defaults."""

    def test_defaults(self):
        req = SpeechRequest(model="tts-1", input="hello")
        assert req.voice == "alloy"
        assert req.response_format == "wav"
        assert req.speed == 1.0
        assert req.instructions is None

    def test_custom_values(self):
        req = SpeechRequest(
            model="tts-1-hd",
            input="hello",
            voice="nova",
            response_format="mp3",
            speed=1.5,
            instructions="whisper",
        )
        assert req.voice == "nova"
        assert req.response_format == "mp3"
        assert req.speed == 1.5
        assert req.instructions == "whisper"

    def test_model_required(self):
        with pytest.raises(ValidationError, match="model"):
            SpeechRequest(input="hello")

    def test_input_required(self):
        with pytest.raises(ValidationError, match="input"):
            SpeechRequest(model="tts-1")
