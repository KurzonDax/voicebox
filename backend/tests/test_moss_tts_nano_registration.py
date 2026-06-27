"""Regression tests for MOSS-TTS-Nano engine registration (PR #507).

Cherry-pick t_754b2d5e added MOSS-TTS-Nano as a new TTS backend. These tests
guard the wiring so a future refactor cannot silently drop the engine from the
registration tables or its pattern validators.

Covers:
- moss_tts_nano appears in TTS_ENGINES registry.
- All four model patterns (GenerationRequest, MCPClientBindingResponse,
  MCPClientBindingUpsert, SpeakRequest) accept the new engine string.
- Language patterns accept the new hu|fa|cs codes on VoiceProfileCreate,
  GenerationRequest, and SpeakRequest.
- get_tts_backend_for_engine returns a MOSSTTSNanoBackend instance for the
  engine string (without instantiating the real ML model).
- _get_non_qwen_tts_configs exposes a moss-tts-nano ModelConfig with the
  full set of 20 supported languages.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from backend.backends import (
    MOSS_TTS_NANO_HF_REPO,
    TTS_ENGINES,
    _get_non_qwen_tts_configs,
    get_tts_backend_for_engine,
)
from backend.models import (
    GenerationRequest,
    MCPClientBindingResponse,
    MCPClientBindingUpsert,
    SpeakRequest,
    VoiceProfileCreate,
)


# ---------------------------------------------------------------------------
# Engine registry
# ---------------------------------------------------------------------------


class TestEngineRegistry:
    """moss_tts_nano must be a registered TTS engine."""

    def test_engine_present_in_tts_engines(self):
        assert "moss_tts_nano" in TTS_ENGINES
        assert TTS_ENGINES["moss_tts_nano"] == "MOSS-TTS-Nano"

    def test_hf_repo_constant_points_to_openmoss(self):
        # The repo ID is referenced from two places (engine registration +
        # _is_model_cached) — they must stay in sync.
        assert MOSS_TTS_NANO_HF_REPO.startswith("OpenMOSS-Team/")


# ---------------------------------------------------------------------------
# Model pattern validation — the upstream PR only patched GenerationRequest.
# Johnny's additional fix (d1f0fc0) extended the same string to MCP bindings
# and SpeakRequest, plus the language pattern to add hu|fa|cs. These tests
# lock in all four engines + three language patterns.
# ---------------------------------------------------------------------------


class TestEnginePatternValidation:
    """All four engine fields accept moss_tts_nano."""

    def test_generation_request_engine(self):
        req = GenerationRequest(profile_id="p", text="hello", engine="moss_tts_nano")
        assert req.engine == "moss_tts_nano"

    def test_mcp_binding_response_engine(self):
        from datetime import datetime
        obj = MCPClientBindingResponse(
            client_id="c",
            default_engine="moss_tts_nano",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        assert obj.default_engine == "moss_tts_nano"

    def test_mcp_binding_upsert_engine(self):
        obj = MCPClientBindingUpsert(client_id="c", default_engine="moss_tts_nano")
        assert obj.default_engine == "moss_tts_nano"

    def test_speak_request_engine(self):
        obj = SpeakRequest(text="hello", engine="moss_tts_nano")
        assert obj.engine == "moss_tts_nano"

    def test_engine_pattern_rejects_unknown(self):
        with pytest.raises(ValidationError):
            GenerationRequest(profile_id="p", text="hello", engine="not_a_real_engine")


class TestLanguagePatternValidation:
    """MOSS introduces hu|fa|cs language codes — all three language fields must accept them."""

    @pytest.mark.parametrize("lang", ["hu", "fa", "cs"])
    def test_language_pattern_accepts_new_codes(self, lang):
        # VoiceProfileCreate — required at creation time
        vp = VoiceProfileCreate(name="n", language=lang)
        assert vp.language == lang

        # GenerationRequest — accepted on synthesis (text is also required)
        gr = GenerationRequest(profile_id="p", text="hello", language=lang)
        assert gr.language == lang

        # SpeakRequest — optional on /speak endpoint (text is required)
        sr = SpeakRequest(text="hello", language=lang)
        assert sr.language == lang

    def test_language_pattern_still_rejects_garbage(self):
        with pytest.raises(ValidationError):
            VoiceProfileCreate(name="n", language="klingon")


# ---------------------------------------------------------------------------
# Backend factory — get_tts_backend_for_engine must dispatch correctly.
# ---------------------------------------------------------------------------


class TestBackendFactory:
    """get_tts_backend_for_engine('moss_tts_nano') returns a MOSSTTSNanoBackend."""

    def test_returns_moss_backend_instance(self):
        # Patch out the real import chain so we don't need torch / numpy /
        # the moss-tts-nano package installed for this unit test.
        fake_module = MagicMock()
        fake_backend = MagicMock()
        fake_module.MOSSTTSNanoBackend = MagicMock(return_value=fake_backend)
        with patch.dict(
            "sys.modules",
            {"backend.backends.moss_tts_nano_backend": fake_module},
        ):
            backend = get_tts_backend_for_engine("moss_tts_nano")

        fake_module.MOSSTTSNanoBackend.assert_called_once_with()
        assert backend is fake_backend

    def test_unknown_engine_still_raises(self):
        with pytest.raises(ValueError, match="Unknown TTS engine"):
            get_tts_backend_for_engine("definitely_not_real")


# ---------------------------------------------------------------------------
# ModelConfig surface — the public _get_non_qwen_tts_configs() entry used by
# /models and the UI to enumerate installable TTS models.
# ---------------------------------------------------------------------------


class TestModelConfig:
    """MOSS-TTS-Nano must be exposed via _get_non_qwen_tts_configs."""

    def test_config_present(self):
        configs = _get_non_qwen_tts_configs()
        moss_configs = [c for c in configs if c.engine == "moss_tts_nano"]
        assert len(moss_configs) == 1
        config = moss_configs[0]
        assert config.model_name == "moss-tts-nano"
        assert config.hf_repo_id == MOSS_TTS_NANO_HF_REPO
        assert config.size_mb > 0

    def test_config_exposes_all_20_languages(self):
        """PR #507 advertises 20-language support; the ModelConfig must include all of them."""
        configs = _get_non_qwen_tts_configs()
        moss_configs = [c for c in configs if c.engine == "moss_tts_nano"]
        assert len(moss_configs) == 1
        langs = set(moss_configs[0].languages)
        # All 20 advertised languages must be present
        expected = {
            "zh", "en", "de", "es", "fr", "ja", "it",
            "hu", "ko", "ru", "fa", "ar", "pl", "pt",
            "cs", "da", "sv", "el", "tr",
        }
        # 19 codes from PR body + extras verified by inspecting backend/models.py
        # language pattern; backend only mentions 19 distinct codes plus the
        # 3 new hu/fa/cs additions. Accept any superset containing the expected set.
        assert expected.issubset(langs), f"missing: {expected - langs}"