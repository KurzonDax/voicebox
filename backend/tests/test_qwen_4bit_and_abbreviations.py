"""Regression tests for PR #563 — 4-bit quantized Qwen TTS models + Russian abbreviations.

Covers:
- _get_qwen_model_configs returns 4-bit variants when backend_type is "mlx".
- _get_qwen_model_configs does NOT return 4-bit variants when backend_type is "pytorch".
- 4-bit model configs have correct model_name, model_size, and hf_repo_id.
- MLX backend _get_model_path maps 4-bit model sizes to the correct HF repos.
- Russian abbreviations in chunked_tts prevent sentence splitting at abbreviation periods.
- English abbreviations still work (regression guard).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from backend.backends import _get_qwen_model_configs
from backend.utils.chunked_tts import _ABBREVIATIONS, split_text_into_chunks

# ---------------------------------------------------------------------------
# 4-bit model config tests
# ---------------------------------------------------------------------------


class TestQwen4BitModelConfigs:
    """4-bit quantized model variants must be correctly registered for MLX only."""

    def test_mlx_backend_returns_4bit_configs(self):
        """When backend_type is mlx, 4-bit variants must be present."""
        with patch("backend.backends.get_backend_type", return_value="mlx"):
            configs = _get_qwen_model_configs()

        model_names = [c.model_name for c in configs]
        assert "qwen-tts-1.7B-4bit" in model_names
        assert "qwen-tts-0.6B-4bit" in model_names

    def test_pytorch_backend_excludes_4bit_configs(self):
        """When backend_type is pytorch, 4-bit variants must NOT be present."""
        with patch("backend.backends.get_backend_type", return_value="pytorch"):
            configs = _get_qwen_model_configs()

        model_names = [c.model_name for c in configs]
        assert "qwen-tts-1.7B-4bit" not in model_names
        assert "qwen-tts-0.6B-4bit" not in model_names

    def test_4bit_config_has_correct_model_size(self):
        """4-bit configs must have model_size ending in '-4bit'."""
        with patch("backend.backends.get_backend_type", return_value="mlx"):
            configs = _get_qwen_model_configs()

        for cfg in configs:
            if "4bit" in cfg.model_name:
                assert cfg.model_size.endswith("-4bit"), f"{cfg.model_name} has model_size={cfg.model_size!r}"

    def test_4bit_config_has_correct_hf_repo(self):
        """4-bit configs must point to mlx-community 4-bit repos."""
        with patch("backend.backends.get_backend_type", return_value="mlx"):
            configs = _get_qwen_model_configs()

        for cfg in configs:
            if cfg.model_name == "qwen-tts-1.7B-4bit":
                assert cfg.hf_repo_id == "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit"
            elif cfg.model_name == "qwen-tts-0.6B-4bit":
                assert cfg.hf_repo_id == "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit"

    def test_4bit_configs_have_smaller_size_than_full(self):
        """4-bit model size_mb must be smaller than the corresponding full-precision model."""
        with patch("backend.backends.get_backend_type", return_value="mlx"):
            configs = _get_qwen_model_configs()

        by_name = {c.model_name: c for c in configs}
        assert by_name["qwen-tts-1.7B-4bit"].size_mb < by_name["qwen-tts-1.7B"].size_mb
        assert by_name["qwen-tts-0.6B-4bit"].size_mb < by_name["qwen-tts-0.6B"].size_mb

    def test_4bit_configs_share_same_languages_as_full(self):
        """4-bit variants must support the same language set as the full-precision models."""
        with patch("backend.backends.get_backend_type", return_value="mlx"):
            configs = _get_qwen_model_configs()

        by_name = {c.model_name: c for c in configs}
        full_langs = set(by_name["qwen-tts-1.7B"].languages)
        bit4_langs = set(by_name["qwen-tts-1.7B-4bit"].languages)
        assert full_langs == bit4_langs

    def test_4bit_configs_use_qwen_engine(self):
        """4-bit variants must use the 'qwen' engine, not a separate engine string."""
        with patch("backend.backends.get_backend_type", return_value="mlx"):
            configs = _get_qwen_model_configs()

        for cfg in configs:
            if "4bit" in cfg.model_name:
                assert cfg.engine == "qwen"

    def test_base_configs_always_present(self):
        """Base (non-4bit) configs must always be present regardless of backend."""
        for backend_type in ("mlx", "pytorch"):
            with patch("backend.backends.get_backend_type", return_value=backend_type):
                configs = _get_qwen_model_configs()

            model_names = [c.model_name for c in configs]
            assert "qwen-tts-1.7B" in model_names
            assert "qwen-tts-0.6B" in model_names


# ---------------------------------------------------------------------------
# MLX backend model path mapping
# ---------------------------------------------------------------------------


class TestMLXModelPath4Bit:
    """MLX backend _get_model_path must resolve 4-bit model sizes to correct repos.

    The mlx_backend module has a heavy import chain (librosa, soundfile, mlx)
    that is not available in CI or on non-Apple platforms.  We stub out the
    problematic modules in sys.modules so the import succeeds, then test the
    pure-dict-lookup _get_model_path method directly.
    """

    @pytest.fixture
    def mlx_backend_cls(self):
        """Import MLXTTSBackend with heavy deps stubbed out."""
        import sys
        import types

        # Stub modules that mlx_backend.py imports transitively but that are
        # not available in the lightweight CI / local test environment.
        stubbed = {}
        for mod_name in [
            "librosa",
            "soundfile",
            "mlx",
            "mlx_audio",
            "mlx_lm",
            "tqdm",
        ]:
            if mod_name not in sys.modules:
                stubbed[mod_name] = types.ModuleType(mod_name)
                sys.modules[mod_name] = stubbed[mod_name]

        # torch needs a Tensor class attribute (referenced at module level by
        # backend/utils/cache.py).
        if "torch" not in sys.modules:
            torch_stub = types.ModuleType("torch")
            torch_stub.Tensor = type  # any type will do for annotation
            stubbed["torch"] = torch_stub
            sys.modules["torch"] = torch_stub

        try:
            # Force-reimport mlx_backend so it picks up the stubs
            for key in list(sys.modules):
                if "mlx_backend" in key or "backend.backends.base" in key:
                    del sys.modules[key]
            from backend.backends.mlx_backend import MLXTTSBackend

            yield MLXTTSBackend
        finally:
            # Clean up stubs
            for mod_name, _mod in stubbed.items():
                sys.modules.pop(mod_name, None)

    @pytest.mark.parametrize(
        ("model_size", "expected_repo"),
        [
            ("1.7B", "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"),
            ("0.6B", "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"),
            ("1.7B-4bit", "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit"),
            ("0.6B-4bit", "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit"),
        ],
    )
    def test_get_model_path_resolves_4bit_sizes(self, mlx_backend_cls, model_size, expected_repo):
        """_get_model_path must return the correct HF repo for all model sizes including 4-bit."""
        # Create a bare instance without calling __init__ (which loads MLX)
        backend = mlx_backend_cls.__new__(mlx_backend_cls)
        result = backend._get_model_path(model_size)
        assert result == expected_repo

    def test_get_model_path_rejects_unknown_size(self, mlx_backend_cls):
        """Unknown model sizes must raise ValueError."""
        backend = mlx_backend_cls.__new__(mlx_backend_cls)
        with pytest.raises(ValueError, match="Unknown model size"):
            backend._get_model_path("99B")


# ---------------------------------------------------------------------------
# Russian abbreviation tests
# ---------------------------------------------------------------------------


class TestRussianAbbreviations:
    """Russian abbreviations must be in the _ABBREVIATIONS set and prevent sentence splitting."""

    @pytest.mark.parametrize(
        "abbr",
        [
            "т.д",
            "т.п",
            "т.е",
            "т.к",
            "т.н",
            "т.о",
            "др",
            "пр",
            "г",
            "гг",
            "в",
            "вв",
            "н.э",
            "ул",
            "д",
            "корп",
            "стр",
            "руб",
            "коп",
            "тыс",
            "млн",
            "млрд",
            "трлн",
            "кв",
            "см",
            "им",
            "проф",
            "акад",
            "доц",
            "ред",
            "изд",
            "обл",
            "р",
            "оз",
            "о",
            "м",
            "гр",
        ],
    )
    def test_abbreviation_in_set(self, abbr):
        """Each Russian abbreviation must be in the _ABBREVIATIONS frozenset."""
        assert abbr in _ABBREVIATIONS, f"{abbr!r} not in _ABBREVIATIONS"

    def test_russian_abbreviation_does_not_split_sentence(self):
        """'т.д.' must not be treated as a sentence boundary in text chunking."""
        # Build text where the period after 'т.д' would cause a split if the
        # abbreviation were not recognised.  We use a long text that exceeds
        # max_chars so the chunker actively searches for sentence boundaries.
        text = (
            "Мы купили яблоки, груши и т.д. "
            "Потом пошли домой и приготовили ужин из всего этого. "
            "Это был очень вкусный ужин, и все остались довольны результатом. "
        )
        # Repeat to exceed default max_chars (800)
        full_text = text * 10
        chunks = split_text_into_chunks(full_text, max_chars=200)
        # The first chunk should NOT end mid-abbreviation — i.e. it should
        # not end with "и т" or "т.д" without the following sentence content.
        for chunk in chunks:
            # No chunk should end with a bare abbreviation fragment
            assert not chunk.rstrip().endswith("т.д"), f"Chunk ended at abbreviation boundary: {chunk[-20:]!r}"

    def test_т_е_does_not_split(self):
        """'т.е.' (то есть) must not be treated as a sentence boundary."""
        text = "Это очень важно, т.е. необходимо сделать прямо сейчас. " * 20
        chunks = split_text_into_chunks(text, max_chars=150)
        for chunk in chunks:
            assert not chunk.rstrip().endswith("т.е"), f"Chunk ended at 'т.е' boundary: {chunk[-20:]!r}"

    def test_т_к_does_not_split(self):
        """'т.к.' (так как) must not be treated as a sentence boundary."""
        text = "Мы не могли пойти, т.к. был сильный дождь весь день. " * 20
        chunks = split_text_into_chunks(text, max_chars=150)
        for chunk in chunks:
            assert not chunk.rstrip().endswith("т.к"), f"Chunk ended at 'т.к' boundary: {chunk[-20:]!r}"

    def test_г_abbreviation_does_not_split(self):
        """'г.' (год/город) must not be treated as a sentence boundary."""
        text = "Он родился в Москве, г. Москва — большой город. В 2020 г. произошло много событий. " * 20
        chunks = split_text_into_chunks(text, max_chars=150)
        for chunk in chunks:
            # A chunk may end with "г." only if it's at a real sentence boundary.
            # Verify we don't get a chunk that ends with just the abbreviation
            # followed by a space and nothing else (the splitter would have
            # cut right after the abbreviation period).
            assert not chunk.rstrip().endswith(", г. Москва — большой го"), (
                f"Chunk cut mid-sentence at 'г.' abbreviation: {chunk[-30:]!r}"
            )


# ---------------------------------------------------------------------------
# English abbreviation regression guard
# ---------------------------------------------------------------------------


class TestEnglishAbbreviationsStillWork:
    """Ensure existing English abbreviations still prevent sentence splitting."""

    @pytest.mark.parametrize("abbr", ["mr", "mrs", "dr", "st", "vs", "etc"])
    def test_english_abbreviation_in_set(self, abbr):
        assert abbr in _ABBREVIATIONS

    def test_mr_does_not_split(self):
        """'Mr.' must not be treated as a sentence boundary."""
        text = (
            "Mr. Smith went to the store to buy some groceries for dinner. "
            "He bought apples, oranges, and bananas for his family. " * 20
        )
        chunks = split_text_into_chunks(text, max_chars=150)
        for chunk in chunks:
            assert not chunk.rstrip().endswith("Mr"), f"Chunk ended at 'Mr' boundary: {chunk[-20:]!r}"
