"""Unit tests for MiniMax Cloud TTS backend."""

import asyncio
import json
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _run(coro):
    """Run a coroutine on a fresh event loop (avoids deprecation of get_event_loop)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_backend():
    from backend.backends.minimax_backend import MiniMaxTTSBackend

    return MiniMaxTTSBackend()


class TestMiniMaxTTSBackend:
    """Tests for MiniMaxTTSBackend."""

    def test_initial_state(self):
        backend = _make_backend()
        assert not backend.is_loaded()
        assert backend._is_model_cached()

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-123"})
    def test_load_model_sets_ready(self):
        backend = _make_backend()
        _run(backend.load_model())
        assert backend.is_loaded()
        assert backend._api_key == "test-key-123"

    @patch.dict(os.environ, {}, clear=True)
    def test_load_model_without_api_key_raises(self):
        os.environ.pop("MINIMAX_API_KEY", None)
        backend = _make_backend()
        with pytest.raises(RuntimeError, match="MINIMAX_API_KEY"):
            _run(backend.load_model())

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    def test_unload_model(self):
        backend = _make_backend()
        _run(backend.load_model())
        assert backend.is_loaded()
        backend.unload_model()
        assert not backend.is_loaded()
        assert backend._api_key is None

    def test_create_voice_prompt_returns_preset(self):
        backend = _make_backend()
        prompt, cached = _run(backend.create_voice_prompt("/fake/audio.wav", "test text"))
        assert prompt["voice_type"] == "preset"
        assert prompt["preset_engine"] == "minimax"
        assert "preset_voice_id" in prompt
        assert not cached

    def test_combine_voice_prompts_raises(self):
        backend = _make_backend()
        with pytest.raises(NotImplementedError):
            _run(backend.combine_voice_prompts(["/a.wav"], ["text"]))

    def test_get_model_path(self):
        backend = _make_backend()
        assert backend._get_model_path() == "speech-2.8-hd"

    def test_is_model_cached_always_true(self):
        backend = _make_backend()
        assert backend._is_model_cached()
        assert backend._is_model_cached("anything")


class TestMiniMaxVoices:
    """Tests for MiniMax voice definitions."""

    def test_voices_structure(self):
        from backend.backends.minimax_backend import MINIMAX_VOICES

        assert len(MINIMAX_VOICES) > 0
        for voice_id, name, gender, lang in MINIMAX_VOICES:
            assert isinstance(voice_id, str)
            assert isinstance(name, str)
            assert gender in ("male", "female")
            assert isinstance(lang, str)

    def test_default_voice_id_in_list(self):
        from backend.backends.minimax_backend import DEFAULT_VOICE_ID, MINIMAX_VOICES

        voice_ids = [v[0] for v in MINIMAX_VOICES]
        assert DEFAULT_VOICE_ID in voice_ids


def _make_mock_response(audio_samples=None):
    """Create a mock API response with valid PCM audio."""
    if audio_samples is None:
        # Generate 1 second of silence at 24kHz
        audio_samples = np.zeros(24000, dtype=np.int16)
    audio_hex = audio_samples.tobytes().hex()
    return json.dumps(
        {
            "base_resp": {"status_code": 0, "status_msg": "success"},
            "data": {"audio": audio_hex},
        }
    ).encode("utf-8")


class TestMiniMaxGenerate:
    """Tests for MiniMax TTS generate with mocked API."""

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    @patch("urllib.request.urlopen")
    def test_generate_returns_audio(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = _make_mock_response()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        backend = _make_backend()
        audio, sr = _run(
            backend.generate(
                "Hello world",
                {"voice_type": "preset", "preset_voice_id": "English_Graceful_Lady"},
            )
        )
        assert sr == 24000
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert len(audio) == 24000  # 1 second

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    @patch("urllib.request.urlopen")
    def test_generate_sends_correct_payload(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = _make_mock_response()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        backend = _make_backend()
        _run(backend.generate("Test text", {"preset_voice_id": "Deep_Voice_Man"}))

        # Verify the request was made with correct parameters
        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        payload = json.loads(request.data.decode("utf-8"))

        assert payload["model"] == "speech-2.8-hd"
        assert payload["text"] == "Test text"
        assert not payload["stream"]
        assert payload["voice_setting"]["voice_id"] == "Deep_Voice_Man"
        assert payload["audio_setting"]["format"] == "pcm"
        assert payload["audio_setting"]["sample_rate"] == 24000

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    @patch("urllib.request.urlopen")
    def test_generate_api_error_raises(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(
            {
                "base_resp": {"status_code": 1001, "status_msg": "Invalid API key"},
                "data": {},
            }
        ).encode("utf-8")
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        backend = _make_backend()
        with pytest.raises(RuntimeError, match="Invalid API key"):
            _run(backend.generate("test", {"preset_voice_id": "English_Graceful_Lady"}))

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    @patch("urllib.request.urlopen")
    def test_generate_uses_default_voice_id(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = _make_mock_response()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        backend = _make_backend()
        _run(backend.generate("test", {}))

        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        payload = json.loads(request.data.decode("utf-8"))
        assert payload["voice_setting"]["voice_id"] == "English_Graceful_Lady"

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    @patch("urllib.request.urlopen")
    def test_generate_empty_audio_raises(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(
            {
                "base_resp": {"status_code": 0, "status_msg": "success"},
                "data": {"audio": ""},
            }
        ).encode("utf-8")
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        backend = _make_backend()
        with pytest.raises(RuntimeError, match="empty audio"):
            _run(backend.generate("test", {"preset_voice_id": "English_Graceful_Lady"}))

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    @patch("urllib.request.urlopen")
    def test_generate_http_error_wraps_runtime_error(self, mock_urlopen):
        import urllib.error
        from email.message import Message

        hdrs = Message()
        hdrs["Content-Type"] = "application/json"
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://api.minimax.io/v1/t2a_v2",
            code=401,
            msg="Unauthorized",
            hdrs=hdrs,
            fp=MagicMock(read=lambda: b"invalid api key"),
        )

        backend = _make_backend()
        with pytest.raises(RuntimeError, match="401"):
            _run(backend.generate("test", {"preset_voice_id": "English_Graceful_Lady"}))

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    def test_load_model_idempotent_when_already_ready(self):
        backend = _make_backend()
        _run(backend.load_model())
        assert backend.is_loaded()
        original_key = backend._api_key
        # Calling again should be a no-op (must NOT re-read env or raise)
        _run(backend.load_model())
        assert backend._api_key == original_key


class TestMiniMaxEngineRegistration:
    """Tests for MiniMax engine registration in backends __init__."""

    def test_minimax_in_tts_engines(self):
        from backend.backends import TTS_ENGINES

        assert "minimax" in TTS_ENGINES

    def test_get_tts_backend_for_engine(self):
        from backend.backends import get_tts_backend_for_engine, reset_backends

        reset_backends()
        backend = get_tts_backend_for_engine("minimax")
        from backend.backends.minimax_backend import MiniMaxTTSBackend

        assert isinstance(backend, MiniMaxTTSBackend)
        reset_backends()
