"""Tests for Chatterbox exaggeration/cfg_weight parameter passthrough.

Covers:
- Pydantic model accepts/rejects exaggeration and cfg_weight values
- generate_chunked threads kwargs to backend.generate() for both
  single-shot and chunked paths
- run_generation injects kwargs only when engine == 'chatterbox'
- ChatterboxTTSBackend.generate() falls back to language defaults when
  exaggeration/cfg_weight are None
"""

import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from backend.models import GenerationRequest
from backend.utils.chunked_tts import generate_chunked


# ── Pydantic model ───────────────────────────────────────────────────


class TestGenerationRequestModel:
    """Validation for exaggeration / cfg_weight fields on GenerationRequest."""

    def _base_payload(self, **overrides):
        payload = {
            "profile_id": "test-profile",
            "text": "Hello world",
            "language": "en",
        }
        payload.update(overrides)
        return payload

    def test_exaggeration_accepted_within_range(self):
        req = GenerationRequest(**self._base_payload(exaggeration=0.5))
        assert req.exaggeration == 0.5

    def test_exaggeration_none_default(self):
        req = GenerationRequest(**self._base_payload())
        assert req.exaggeration is None

    def test_exaggeration_rejected_above_one(self):
        with pytest.raises(Exception):
            GenerationRequest(**self._base_payload(exaggeration=1.5))

    def test_exaggeration_rejected_below_zero(self):
        with pytest.raises(Exception):
            GenerationRequest(**self._base_payload(exaggeration=-0.1))

    def test_cfg_weight_accepted_within_range(self):
        req = GenerationRequest(**self._base_payload(cfg_weight=0.8))
        assert req.cfg_weight == 0.8

    def test_cfg_weight_none_default(self):
        req = GenerationRequest(**self._base_payload())
        assert req.cfg_weight is None

    def test_cfg_weight_rejected_above_one(self):
        with pytest.raises(Exception):
            GenerationRequest(**self._base_payload(cfg_weight=1.01))

    def test_cfg_weight_rejected_below_zero(self):
        with pytest.raises(Exception):
            GenerationRequest(**self._base_payload(cfg_weight=-0.01))

    def test_both_fields_accepted_together(self):
        req = GenerationRequest(
            **self._base_payload(exaggeration=0.0, cfg_weight=1.0)
        )
        assert req.exaggeration == 0.0
        assert req.cfg_weight == 1.0


# ── generate_chunked passthrough ─────────────────────────────────────


class TestGenerateChunkedPassthrough:
    """generate_chunked must forward exaggeration/cfg_weight to backend.generate()."""

    def _make_backend(self, audio_len=1000):
        """Create an async mock backend whose generate() returns audio + sr."""
        audio = MagicMock()
        audio.__len__ = lambda self: audio_len
        backend = AsyncMock()
        backend.generate.return_value = (audio, 24000)
        return backend

    @pytest.mark.asyncio
    async def test_single_shot_forwards_kwargs(self):
        """Short text path must pass exaggeration and cfg_weight to backend.generate."""
        backend = self._make_backend()
        await generate_chunked(
            backend,
            "short text",
            {"ref_audio": "/dev/null"},
            language="en",
            exaggeration=0.7,
            cfg_weight=0.3,
        )

        call_kwargs = backend.generate.call_args
        assert call_kwargs.kwargs.get("exaggeration") == 0.7
        assert call_kwargs.kwargs.get("cfg_weight") == 0.3

    @pytest.mark.asyncio
    async def test_single_shot_omits_none_kwargs(self):
        """When exaggeration/cfg_weight are None they must not be passed."""
        backend = self._make_backend()
        await generate_chunked(
            backend,
            "short text",
            {"ref_audio": "/dev/null"},
            language="en",
            exaggeration=None,
            cfg_weight=None,
        )

        call_kwargs = backend.generate.call_args
        assert "exaggeration" not in call_kwargs.kwargs
        assert "cfg_weight" not in call_kwargs.kwargs

    @pytest.mark.asyncio
    async def test_single_shot_partial_kwargs(self):
        """Only exaggeration provided — cfg_weight should be absent."""
        backend = self._make_backend()
        await generate_chunked(
            backend,
            "short text",
            {"ref_audio": "/dev/null"},
            language="en",
            exaggeration=0.9,
        )

        call_kwargs = backend.generate.call_args
        assert call_kwargs.kwargs.get("exaggeration") == 0.9
        assert "cfg_weight" not in call_kwargs.kwargs

    @pytest.mark.asyncio
    async def test_chunked_forwards_kwargs_to_each_chunk(self):
        """Long text path must pass kwargs to every chunk's backend.generate call."""
        backend = self._make_backend()
        long_text = " ".join(["word"] * 500)  # well above default 800 chars

        await generate_chunked(
            backend,
            long_text,
            {"ref_audio": "/dev/null"},
            language="en",
            max_chunk_chars=100,
            exaggeration=0.6,
            cfg_weight=0.4,
        )

        assert backend.generate.call_count > 1
        for call in backend.generate.call_args_list:
            assert call.kwargs.get("exaggeration") == 0.6
            assert call.kwargs.get("cfg_weight") == 0.4


# ── run_generation engine gating ─────────────────────────────────────


class TestRunGenerationGating:
    """run_generation must inject exaggeration/cfg_weight only for chatterbox."""

    def test_run_generation_accepts_exaggeration_params(self):
        """The function signature must include exaggeration and cfg_weight."""
        from backend.services.generation import run_generation

        sig = inspect.signature(run_generation)
        assert "exaggeration" in sig.parameters
        assert "cfg_weight" in sig.parameters

    @pytest.mark.asyncio
    async def test_chatterbox_engine_injects_kwargs(self):
        """When engine == 'chatterbox', gen_kwargs must include exaggeration/cfg_weight."""
        from backend.services.generation import run_generation

        captured_kwargs = {}

        async def fake_generate_chunked(backend, text, voice_prompt, **kwargs):
            captured_kwargs.update(kwargs)
            return MagicMock(), 24000

        mock_profiles = MagicMock()
        mock_profiles.create_voice_prompt_for_profile = AsyncMock(return_value={})
        mock_history = MagicMock()
        mock_history.update_generation_status = AsyncMock()
        mock_history.create_generation = AsyncMock()

        with (
            patch(
                "backend.backends.load_engine_model",
                new_callable=AsyncMock,
            ),
            patch(
                "backend.backends.get_tts_backend_for_engine",
                return_value=MagicMock(
                    is_loaded=MagicMock(return_value=True),
                    _is_model_cached=MagicMock(return_value=True),
                ),
            ),
            patch(
                "backend.backends.engine_needs_trim",
                return_value=False,
            ),
            patch(
                "backend.utils.chunked_tts.generate_chunked",
                side_effect=fake_generate_chunked,
            ),
            patch("backend.services.generation.profiles", mock_profiles),
            patch("backend.services.generation.history", mock_history),
            patch("backend.utils.tasks.get_task_manager"),
            patch("backend.utils.audio.normalize_audio"),
            patch("backend.utils.audio.save_audio"),
            patch("backend.services.generation.get_db") as mock_get_db,
        ):
            mock_db = MagicMock()
            mock_get_db.return_value = iter([mock_db])

            await run_generation(
                generation_id="test-gen",
                profile_id="test-profile",
                text="hello",
                language="en",
                engine="chatterbox",
                model_size="1.7B",
                seed=None,
                mode="generate",
                exaggeration=0.8,
                cfg_weight=0.2,
            )

        assert captured_kwargs.get("exaggeration") == 0.8
        assert captured_kwargs.get("cfg_weight") == 0.2

    @pytest.mark.asyncio
    async def test_non_chatterbox_engine_omits_kwargs(self):
        """When engine != 'chatterbox', exaggeration/cfg_weight must not be in gen_kwargs."""
        from backend.services.generation import run_generation

        captured_kwargs = {}

        async def fake_generate_chunked(backend, text, voice_prompt, **kwargs):
            captured_kwargs.update(kwargs)
            return MagicMock(), 24000

        mock_profiles = MagicMock()
        mock_profiles.create_voice_prompt_for_profile = AsyncMock(return_value={})
        mock_history = MagicMock()
        mock_history.update_generation_status = AsyncMock()
        mock_history.create_generation = AsyncMock()

        with (
            patch(
                "backend.backends.load_engine_model",
                new_callable=AsyncMock,
            ),
            patch(
                "backend.backends.get_tts_backend_for_engine",
                return_value=MagicMock(
                    is_loaded=MagicMock(return_value=True),
                    _is_model_cached=MagicMock(return_value=True),
                ),
            ),
            patch(
                "backend.backends.engine_needs_trim",
                return_value=False,
            ),
            patch(
                "backend.utils.chunked_tts.generate_chunked",
                side_effect=fake_generate_chunked,
            ),
            patch("backend.services.generation.profiles", mock_profiles),
            patch("backend.services.generation.history", mock_history),
            patch("backend.utils.tasks.get_task_manager"),
            patch("backend.utils.audio.normalize_audio"),
            patch("backend.utils.audio.save_audio"),
            patch("backend.services.generation.get_db") as mock_get_db,
        ):
            mock_db = MagicMock()
            mock_get_db.return_value = iter([mock_db])

            await run_generation(
                generation_id="test-gen",
                profile_id="test-profile",
                text="hello",
                language="en",
                engine="qwen",
                model_size="1.7B",
                seed=None,
                mode="generate",
                exaggeration=0.8,
                cfg_weight=0.2,
            )

        assert "exaggeration" not in captured_kwargs
        assert "cfg_weight" not in captured_kwargs


# ── ChatterboxTTSBackend.generate() ternary fallback ─────────────────


class TestChatterboxBackendFallback:
    """ChatterboxTTSBackend.generate() must pass through explicit
    exaggeration/cfg_weight when provided, and fall back to the per-language
    defaults (or global defaults for unknown languages) when None.

    These tests mock the underlying `model.generate` call so no real model is
    loaded — only the new ternary fallback logic at the call site is exercised.
    """

    @pytest.mark.asyncio
    async def test_explicit_params_passed_through(self):
        """When exaggeration/cfg_weight are provided, they must reach model.generate."""
        from backend.backends.chatterbox_backend import ChatterboxTTSBackend

        backend = ChatterboxTTSBackend()
        # model.generate is called inside asyncio.to_thread; we replace the
        # mock AFTER to_thread calls it, so the captured call_kwargs reflect
        # what the call site actually sent.
        fake_audio = np.zeros(8, dtype=np.float32)
        backend.model = MagicMock()
        backend.model.generate = MagicMock(return_value=fake_audio)

        async def fake_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch(
            "backend.backends.chatterbox_backend.asyncio.to_thread",
            side_effect=fake_to_thread,
        ):
            with patch.object(backend, "load_model", new=AsyncMock()):
                await backend.generate(
                    text="hello",
                    voice_prompt={"ref_audio": None},
                    language="en",
                    exaggeration=0.42,
                    cfg_weight=0.73,
                )

        call_kwargs = backend.model.generate.call_args.kwargs
        assert call_kwargs["exaggeration"] == 0.42
        assert call_kwargs["cfg_weight"] == 0.73

    @pytest.mark.asyncio
    async def test_none_params_fall_back_to_global_defaults(self):
        """When None, exaggeration/cfg_weight must be replaced by the global
        defaults for languages without a per-language entry (e.g. 'fr')."""
        from backend.backends.chatterbox_backend import ChatterboxTTSBackend

        backend = ChatterboxTTSBackend()
        fake_audio = np.zeros(8, dtype=np.float32)
        backend.model = MagicMock()
        backend.model.generate = MagicMock(return_value=fake_audio)

        async def fake_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch(
            "backend.backends.chatterbox_backend.asyncio.to_thread",
            side_effect=fake_to_thread,
        ):
            with patch.object(backend, "load_model", new=AsyncMock()):
                # 'fr' is NOT in _LANG_DEFAULTS → falls back to _GLOBAL_DEFAULTS.
                await backend.generate(
                    text="bonjour",
                    voice_prompt={"ref_audio": None},
                    language="fr",
                    exaggeration=None,
                    cfg_weight=None,
                )

        call_kwargs = backend.model.generate.call_args.kwargs
        # The fallback path must NOT pass None through to the model.
        assert call_kwargs["exaggeration"] is not None
        assert call_kwargs["cfg_weight"] is not None
        # Global defaults (from _GLOBAL_DEFAULTS).
        assert call_kwargs["exaggeration"] == ChatterboxTTSBackend._GLOBAL_DEFAULTS["exaggeration"]
        assert call_kwargs["cfg_weight"] == ChatterboxTTSBackend._GLOBAL_DEFAULTS["cfg_weight"]

    @pytest.mark.asyncio
    async def test_none_params_fall_back_to_hebrew_lang_defaults(self):
        """When None and language has its own defaults ('he'), those must win."""
        from backend.backends.chatterbox_backend import ChatterboxTTSBackend

        backend = ChatterboxTTSBackend()
        fake_audio = np.zeros(8, dtype=np.float32)
        backend.model = MagicMock()
        backend.model.generate = MagicMock(return_value=fake_audio)

        async def fake_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch(
            "backend.backends.chatterbox_backend.asyncio.to_thread",
            side_effect=fake_to_thread,
        ):
            with patch.object(backend, "load_model", new=AsyncMock()):
                # 'he' HAS a _LANG_DEFAULTS entry (0.4 / 0.7).
                await backend.generate(
                    text="שלום",
                    voice_prompt={"ref_audio": None},
                    language="he",
                    exaggeration=None,
                    cfg_weight=None,
                )

        call_kwargs = backend.model.generate.call_args.kwargs
        # Hebrew-specific defaults must win over the global defaults.
        assert call_kwargs["exaggeration"] == ChatterboxTTSBackend._LANG_DEFAULTS["he"]["exaggeration"]
        assert call_kwargs["cfg_weight"] == ChatterboxTTSBackend._LANG_DEFAULTS["he"]["cfg_weight"]
        # Sanity: the Hebrew defaults are intentionally different from globals.
        assert call_kwargs["exaggeration"] != ChatterboxTTSBackend._GLOBAL_DEFAULTS["exaggeration"]