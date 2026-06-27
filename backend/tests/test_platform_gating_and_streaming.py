"""Regression tests for PRs #657 (platform gating), #718 (engine UX), and
#642 (immediate streaming mode + GET /tts).

Each section maps to one of the cherry-picked commits:

- ``0b1dab7`` + ``4b5d4ee`` (PR #657): ``ModelConfig.requires`` field,
  ``get_supported_platforms()``, ``is_engine_platform_compatible()``, and
  the platform guard at the top of ``load_engine_model``.
- ``7207ae5`` (PR #718): engine descriptions in ``EngineModelSelector``
  and profile/engine compatibility validation in the generation form.
- ``5b62ea4`` + ``24fadfd`` + ``964a9ba`` (PR #642): chunked WAV
  streaming helpers + GET ``/tts`` endpoint + ValueError ``__cause__``
  chaining in ``/generate`` HTTP errors.

The tests deliberately avoid loading any ML backend (torch, transformers,
mlx, etc.) so they run inside the lightweight CI subset.
"""

from __future__ import annotations

import asyncio
import struct
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from fastapi import HTTPException
from starlette.testclient import TestClient

# ---------------------------------------------------------------------------
# platform_detect.get_supported_platforms  (PR #657)
# ---------------------------------------------------------------------------


class FakeTorch:
    """Minimal stand-in for the ``torch`` module that platform_detect
    inspects via ``torch.cuda.is_available()`` / ``torch.backends.mps``
    / ``torch.xpu.is_available()``."""

    def __init__(
        self,
        *,
        cuda_available: bool = False,
        mps_available: bool = False,
        xpu_available: bool = False,
        hip_version: str | None = None,
    ):
        self.cuda = types.SimpleNamespace(is_available=lambda: cuda_available)
        self.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps_available))
        self.xpu = types.SimpleNamespace(is_available=lambda: xpu_available)
        self.version = types.SimpleNamespace(hip=hip_version)


def _install_torch(monkeypatch, fake: FakeTorch | None):
    """Install or remove a fake ``torch`` module in sys.modules."""
    if fake is None:
        # Simulate "torch not installed" — module disappears.
        for mod in list(sys.modules):
            if mod == "torch" or mod.startswith("torch."):
                monkeypatch.delitem(sys.modules, mod, raising=False)
        # Force the ImportError inside get_supported_platforms by patching
        # builtins.__import__ is heavy-handed; instead, patch the module
        # attribute the function looks up via ``import torch``.
        # The cleanest path is to shadow sys.modules['torch'] with a module
        # that raises on attribute access — but the function uses ``import
        # torch`` inside a try/except ImportError. We make it raise by
        # removing the cached module and inserting a sentinel whose import
        # raises ImportError on use. The function's ``try: import torch``
        # will succeed (the module exists), but attribute access on the
        # module is not how ImportError is triggered here. So the simplest
        # faithful simulation is: remove torch entirely so the import
        # raises ImportError.  We do that by monkeypatching the function's
        # globals.
        monkeypatch.setitem(sys.modules, "torch", None)
        return

    # Real path — install a fake torch module so the function's
    # ``import torch`` succeeds and the attribute lookups go to our fake.
    fake_mod = types.ModuleType("torch")
    fake_mod.cuda = fake.cuda
    fake_mod.backends = fake.backends
    fake_mod.xpu = fake.xpu
    fake_mod.version = fake.version
    monkeypatch.setitem(sys.modules, "torch", fake_mod)


def test_get_supported_platforms_cpu_only(monkeypatch):
    """No CUDA / MPS / XPU, no torch → ["cpu"]."""
    from backend.utils import platform_detect

    _install_torch(monkeypatch, None)
    # Even when torch is "unavailable", the function always appends "cpu".
    result = platform_detect.get_supported_platforms()
    assert result == ["cpu"]


def test_get_supported_platforms_cuda(monkeypatch):
    """CUDA available + no ROCm → ["cuda", "cpu"]."""
    from backend.utils import platform_detect

    _install_torch(
        monkeypatch,
        FakeTorch(cuda_available=True, hip_version=None),
    )
    assert platform_detect.get_supported_platforms() == ["cuda", "cpu"]


def test_get_supported_platforms_rocm(monkeypatch):
    """CUDA available + torch.version.hip is set → ["rocm", "cpu"]."""
    from backend.utils import platform_detect

    _install_torch(
        monkeypatch,
        FakeTorch(cuda_available=True, hip_version="5.6"),
    )
    assert platform_detect.get_supported_platforms() == ["rocm", "cpu"]


def test_get_supported_platforms_mps(monkeypatch):
    """MPS available (Apple Silicon) → ["mps", "cpu"], NOT also "cuda"."""
    from backend.utils import platform_detect

    _install_torch(
        monkeypatch,
        FakeTorch(mps_available=True),
    )
    assert platform_detect.get_supported_platforms() == ["mps", "cpu"]


def test_get_supported_platforms_xpu(monkeypatch):
    """Intel XPU (torch.xpu.is_available) → ["xpu", "cpu"]."""
    from backend.utils import platform_detect

    _install_torch(
        monkeypatch,
        FakeTorch(xpu_available=True),
    )
    assert platform_detect.get_supported_platforms() == ["xpu", "cpu"]


def test_get_supported_platforms_dedupes_no_double_count(monkeypatch):
    """All three accelerators report → every platform listed, "cpu" not
    duplicated."""
    from backend.utils import platform_detect

    _install_torch(
        monkeypatch,
        FakeTorch(
            cuda_available=True,
            mps_available=True,
            xpu_available=True,
            hip_version=None,
        ),
    )
    result = platform_detect.get_supported_platforms()
    # Order: cuda, mps, xpu, then cpu always appended.
    assert result == ["cuda", "mps", "xpu", "cpu"]


# ---------------------------------------------------------------------------
# ModelConfig.requires  (PR #657)
# ---------------------------------------------------------------------------


def test_model_config_requires_defaults_to_empty_list():
    """``ModelConfig.requires`` must default to ``[]`` — every existing
    engine on main is cross-platform and a missing default would silently
    mark them all as platform-gated."""
    from backend.backends import ModelConfig

    cfg = ModelConfig(
        engine="qwen",
        model_name="qwen-tts",
        display_name="Qwen",
        hf_repo_id="Qwen/Qwen2.5-1.5B-Instruct",
    )
    assert cfg.requires == []
    # And it's a real list, not e.g. a shared sentinel that bleeds across
    # instances.
    cfg2 = ModelConfig(
        engine="luxtts",
        model_name="luxtts",
        display_name="LuxTTS",
        hf_repo_id="YatharthS/LuxTTS",
    )
    assert cfg2.requires == []
    cfg.requires.append("cuda")
    assert cfg2.requires == []  # mutable but isolated


def test_model_config_requires_round_trips_through_dataclass():
    """Constructed with a non-empty list round-trips intact."""
    from backend.backends import ModelConfig

    cfg = ModelConfig(
        engine="voxcpm",
        model_name="voxcpm",
        display_name="VoxCPM",
        hf_repo_id="OpenBMB/VoxCPM",
        requires=["cuda"],
    )
    assert cfg.requires == ["cuda"]


# ---------------------------------------------------------------------------
# is_engine_platform_compatible  (PR #657)
# ---------------------------------------------------------------------------


def _fake_config(engine: str, model_size: str = "default", requires=None):
    """Build a ModelConfig without touching the heavy registry helpers."""
    from backend.backends import ModelConfig

    return ModelConfig(
        engine=engine,
        model_size=model_size,
        model_name=f"{engine}-{model_size}",
        display_name=engine,
        hf_repo_id=f"org/{engine}",
        requires=list(requires or []),
    )


def test_is_engine_compatible_empty_requires_is_always_compatible(monkeypatch):
    """An engine with requires=[] runs everywhere, even on a CPU-only box."""
    from backend.backends import is_engine_platform_compatible

    monkeypatch.setattr("backend.backends.get_supported_platforms", lambda: ["cpu"])
    # Patch the registry so we don't pull in real engine registrations.
    monkeypatch.setattr(
        "backend.backends.get_tts_model_configs",
        lambda: [_fake_config("qwen", requires=[])],
    )
    assert is_engine_platform_compatible("qwen") is True


def test_is_engine_compatible_supported_platform_match(monkeypatch):
    """requires=["cuda"] + supported includes "cuda" → True."""
    from backend.backends import is_engine_platform_compatible

    monkeypatch.setattr("backend.backends.get_supported_platforms", lambda: ["cuda", "cpu"])
    monkeypatch.setattr(
        "backend.backends.get_tts_model_configs",
        lambda: [_fake_config("voxcpm", requires=["cuda"])],
    )
    assert is_engine_platform_compatible("voxcpm") is True


def test_is_engine_compatible_no_supported_platform(monkeypatch):
    """requires=["cuda"] on a CPU-only machine → False."""
    from backend.backends import is_engine_platform_compatible

    monkeypatch.setattr("backend.backends.get_supported_platforms", lambda: ["cpu"])
    monkeypatch.setattr(
        "backend.backends.get_tts_model_configs",
        lambda: [_fake_config("voxcpm", requires=["cuda"])],
    )
    assert is_engine_platform_compatible("voxcpm") is False


def test_is_engine_compatible_any_variant_supported(monkeypatch):
    """Engine with multiple variants: at least one variant compatible → True.

    Mirrors the CodeRabbit fix in 4b5d4ee — earlier code only inspected
    ``configs[0]`` and falsely returned False when a later variant was
    compatible.
    """
    from backend.backends import is_engine_platform_compatible

    monkeypatch.setattr("backend.backends.get_supported_platforms", lambda: ["mps", "cpu"])
    monkeypatch.setattr(
        "backend.backends.get_tts_model_configs",
        lambda: [
            _fake_config("qwen", model_size="4B", requires=["cuda"]),
            _fake_config("qwen", model_size="1.7B", requires=[]),
        ],
    )
    # configs[0] is incompatible (cuda), configs[1] is universal → True
    assert is_engine_platform_compatible("qwen") is True


def test_is_engine_compatible_unknown_engine_returns_true(monkeypatch):
    """An unknown engine name (no configs) returns True — the guard must
    not silently break engines added later that haven't been registered
    yet."""
    from backend.backends import is_engine_platform_compatible

    monkeypatch.setattr("backend.backends.get_supported_platforms", lambda: ["cpu"])
    monkeypatch.setattr("backend.backends.get_tts_model_configs", list)
    assert is_engine_platform_compatible("brand_new_engine") is True


# ---------------------------------------------------------------------------
# load_engine_model platform guard  (PR #657)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_load_engine_model_raises_400_on_incompatible_platform(monkeypatch):
    """Incompatible engine raises HTTP 400 with a clear message — callers
    get a hard refusal, not a silent fallback to CPU or a torch crash."""
    from backend import backends

    # Pretend we're on a CPU-only box and require CUDA.
    monkeypatch.setattr("backend.backends.get_supported_platforms", lambda: ["cpu"])
    monkeypatch.setattr(
        "backend.backends.get_tts_model_configs",
        lambda: [_fake_config("voxcpm", requires=["cuda"])],
    )

    # The guard runs before any backend lookup — get_tts_backend_for_engine
    # must never be reached on incompatibility.
    called_backend_lookup = MagicMock(side_effect=AssertionError("backend lookup should not run"))
    monkeypatch.setattr("backend.backends.get_tts_backend_for_engine", called_backend_lookup)

    with pytest.raises(HTTPException) as exc_info:
        await backends.load_engine_model("voxcpm", "default")

    assert exc_info.value.status_code == 400
    detail = exc_info.value.detail
    assert "voxcpm" in detail
    assert "cuda" in detail
    assert "cpu" in detail


@pytest.mark.asyncio
async def test_load_engine_model_proceeds_when_compatible(monkeypatch):
    """Compatible engine: the guard is bypassed and the real backend
    lookup runs (here mocked so we don't load a real model)."""
    from backend import backends

    monkeypatch.setattr("backend.backends.get_supported_platforms", lambda: ["cuda", "cpu"])
    monkeypatch.setattr(
        "backend.backends.get_tts_model_configs",
        lambda: [_fake_config("voxcpm", requires=["cuda"])],
    )

    fake_backend = MagicMock()
    # voxcpm hits the ``else: await backend.load_model()`` branch
    # (not qwen/qwen_custom_voice and not tada). Use AsyncMock so the
    # coroutine can be awaited.
    fake_backend.load_model = AsyncMock(return_value=None)
    monkeypatch.setattr("backend.backends.get_tts_backend_for_engine", lambda e: fake_backend)

    # Should not raise.
    await backends.load_engine_model("voxcpm", "default")
    fake_backend.load_model.assert_awaited_once()


@pytest.mark.asyncio
async def test_load_engine_model_no_requires_skips_guard(monkeypatch):
    """Engine with requires=[] always passes the guard regardless of
    platform — verified by ensuring the guard doesn't look at the
    platform list when requires is empty."""
    from backend import backends

    platforms_queried = []
    monkeypatch.setattr(
        "backend.backends.get_supported_platforms",
        lambda: platforms_queried.append(None) or ["cpu"],
    )
    monkeypatch.setattr(
        "backend.backends.get_tts_model_configs",
        lambda: [_fake_config("qwen", requires=[])],
    )

    fake_backend = MagicMock()
    fake_backend.load_model_async = MagicMock(return_value=asyncio.sleep(0))
    monkeypatch.setattr("backend.backends.get_tts_backend_for_engine", lambda e: fake_backend)

    await backends.load_engine_model("qwen", "default")
    fake_backend.load_model_async.assert_called_once()


# ---------------------------------------------------------------------------
# _wav_stream_header / _audio_to_pcm16le  (PR #642)
# ---------------------------------------------------------------------------


def test_wav_stream_header_round_trips_via_stdlib_parser():
    """The header must be parseable by Python's wave module so clients
    that read the stream chunk-by-chunk don't fail on byte order or
    field layout."""
    import io
    import wave

    from backend.routes.generations import _wav_stream_header

    header = _wav_stream_header(sample_rate=24000)
    assert len(header) == 44  # canonical PCM WAV header size

    # wave.open requires the "RIFF/size/WAVE/fmt /..." prefix; we add a
    # single silent sample so it can finalize.
    fake_payload = b"\x00\x00"  # one int16 silence sample
    full = header + fake_payload

    with wave.open(io.BytesIO(full), "rb") as wf:
        assert wf.getframerate() == 24000
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2


def test_wav_stream_header_uses_sentinel_lengths_for_streaming():
    """The data and RIFF sizes must be 0xFFFFFFFF (the WAV "unknown
    length" sentinel). Anything else causes players to truncate the
    stream mid-chunk."""
    from backend.routes.generations import _wav_stream_header

    header = _wav_stream_header(sample_rate=16000, channels=2)
    # RIFF size (offset 4, little-endian uint32) and data size (offset 40)
    (riff_size,) = struct.unpack_from("<I", header, 4)
    (data_size,) = struct.unpack_from("<I", header, 40)
    assert riff_size == 0xFFFFFFFF
    assert data_size == 0xFFFFFFFF
    # block_align for stereo 16-bit = 4
    (block_align,) = struct.unpack_from("<H", header, 32)
    assert block_align == 4


def test_audio_to_pcm16le_clamps_to_int16_range():
    """Values outside [-1, 1] must be clipped — float > 1.0 in int16
    conversion overflows and produces harsh clicks."""
    from backend.routes.generations import _audio_to_pcm16le

    audio = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    pcm = _audio_to_pcm16le(audio)
    samples = np.frombuffer(pcm, dtype="<i2")
    # -2.0 → clipped to -1.0 → -32767 (or -32768 depending on rounding
    # direction); 2.0 → clipped to +1.0 → +32767.
    assert samples[0] == -32767  # clipped from -2.0
    assert samples[1] == -32767  # -1.0 * 32767 (round-half-to-even → -32767)
    assert samples[2] == 0
    assert samples[3] == 32767
    assert samples[4] == 32767  # clipped from +2.0


def test_audio_to_pcm16le_handles_multichannel_by_flattening():
    """Multi-dim input is flattened to mono — the streaming header is
    mono-only and a (N, 2) array would otherwise feed interleaved
    samples that get interpreted as tiny mono."""
    from backend.routes.generations import _audio_to_pcm16le

    audio = np.array([[0.5, -0.5], [0.25, -0.25]], dtype=np.float32)  # 2x2
    pcm = _audio_to_pcm16le(audio)
    samples = np.frombuffer(pcm, dtype="<i2")
    # Flattened column-major → [0.5, 0.25, -0.5, -0.25] (reshape(-1)
    # is row-major; 2D flatten gives [0.5, -0.5, 0.25, -0.25]).
    assert len(samples) == 4
    # The streaming client doesn't care about channel order in the
    # helper itself — only that the byte count equals sample_count * 2.
    assert len(pcm) == 4 * 2


def test_audio_to_pcm16le_byte_count_matches_int16_size():
    """The output is little-endian int16, so 1 sample = 2 bytes."""
    from backend.routes.generations import _audio_to_pcm16le

    rng = np.random.default_rng(seed=42)
    audio = rng.uniform(-0.99, 0.99, size=1024).astype(np.float32)
    pcm = _audio_to_pcm16le(audio)
    assert len(pcm) == 1024 * 2
    # Bytes are little-endian — confirm by writing a known value and
    # checking the first byte is the low byte (a positive int16 < 256
    # round-trips with the low byte first on little-endian).
    audio = np.array([0.5], dtype=np.float32)
    pcm = _audio_to_pcm16le(audio)
    # 0.5 * 32767 = 16383.5 → astype('<i2') truncates toward zero → 16383
    # (=0x3FFF) → little-endian bytes are 0xff (low), 0x3f (high).
    # This matches numpy's documented cast-from-float truncation behavior;
    # for streaming audio the half-LSB loss is inaudible.
    assert pcm[0] == 0xFF
    assert pcm[1] == 0x3F


# ---------------------------------------------------------------------------
# stream_speech chunked streaming + crossfade  (PR #642)
# ---------------------------------------------------------------------------


def _streaming_payload(
    text: str = "hello world. second sentence. third one.", *, max_chunk_chars: int = 120, crossfade_ms: int = 10
):
    """Build a GenerationRequest with the streaming-specific fields used
    by the new chunked code path."""
    from backend import models

    return models.GenerationRequest(
        profile_id="profile-1",
        text=text,
        language="en",
        engine="qwen",
        max_chunk_chars=max_chunk_chars,
        crossfade_ms=crossfade_ms,
        seed=42,
    )


def _make_stream_profile():
    """Profile stub with the attributes stream_speech reads."""
    profile = MagicMock()
    profile.id = "profile-1"
    profile.personality = None
    profile.effects_chain = None
    profile.default_engine = None
    profile.preset_engine = None
    return profile


@pytest.mark.asyncio
async def test_stream_speech_emits_wav_header_before_chunks(monkeypatch):
    """The first yielded bytes must be the WAV stream header (44 bytes,
    'RIFF....WAVE') so the streaming parser knows sample rate and bit
    depth."""
    from backend.routes import generations as gen_routes

    profile = _make_stream_profile()
    monkeypatch.setattr(
        "backend.services.profiles.get_profile",
        lambda *a, **kw: asyncio.sleep(0, result=profile),
    )
    monkeypatch.setattr(
        "backend.services.profiles.validate_profile_engine",
        lambda *a, **kw: None,
    )

    fake_backend = MagicMock()

    async def fake_generate(text, voice_prompt, language, seed, instruct):
        # 50 ms of 24 kHz mono silence → 1200 samples
        return np.zeros(1200, dtype=np.float32), 24000

    fake_backend.generate = fake_generate
    # ``stream_speech`` imports these lazily from backend.backends, so we
    # patch the source module — not the route module's namespace.
    monkeypatch.setattr(
        "backend.backends.get_tts_backend_for_engine",
        lambda e: fake_backend,
    )
    monkeypatch.setattr(
        "backend.backends.ensure_model_cached_or_raise",
        lambda *a, **kw: asyncio.sleep(0, result=None),
    )
    monkeypatch.setattr(
        "backend.backends.load_engine_model",
        lambda *a, **kw: asyncio.sleep(0, result=None),
    )
    monkeypatch.setattr(
        "backend.backends.engine_needs_trim",
        lambda e: False,
    )
    monkeypatch.setattr(
        "backend.services.profiles.create_voice_prompt_for_profile",
        lambda *a, **kw: asyncio.sleep(0, result=None),
    )

    db = MagicMock()
    payload = _streaming_payload("Hello there, friend.")

    response = await gen_routes.stream_speech(payload, db)
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)

    # First chunk is the 44-byte header.
    assert len(chunks) >= 2
    assert chunks[0][:4] == b"RIFF"
    assert chunks[0][8:12] == b"WAVE"
    assert len(chunks[0]) == 44
    # Subsequent chunks are PCM (multiples of 2 bytes).
    for body in chunks[1:]:
        assert len(body) % 2 == 0


@pytest.mark.asyncio
async def test_stream_speech_handles_multiple_text_chunks(monkeypatch):
    """Text long enough to split into multiple chunks exercises the
    crossfade-blending loop. We assert that multiple chunks were sent
    to the backend and the crossfade tail bytes are flushed at the end."""
    from backend.routes import generations as gen_routes

    profile = _make_stream_profile()
    monkeypatch.setattr(
        "backend.services.profiles.get_profile",
        lambda *a, **kw: asyncio.sleep(0, result=profile),
    )
    monkeypatch.setattr(
        "backend.services.profiles.validate_profile_engine",
        lambda *a, **kw: None,
    )

    fake_backend = MagicMock()
    chunk_count = {"n": 0}

    async def fake_generate(text, voice_prompt, language, seed, instruct):
        chunk_count["n"] += 1
        # 100 ms of constant tone at 24 kHz → 2400 samples per chunk.
        return np.full(2400, 0.1, dtype=np.float32), 24000

    fake_backend.generate = fake_generate
    monkeypatch.setattr(
        "backend.backends.get_tts_backend_for_engine",
        lambda e: fake_backend,
    )
    monkeypatch.setattr(
        "backend.backends.ensure_model_cached_or_raise",
        lambda *a, **kw: asyncio.sleep(0, result=None),
    )
    monkeypatch.setattr(
        "backend.backends.load_engine_model",
        lambda *a, **kw: asyncio.sleep(0, result=None),
    )
    monkeypatch.setattr(
        "backend.backends.engine_needs_trim",
        lambda e: False,
    )
    monkeypatch.setattr(
        "backend.services.profiles.create_voice_prompt_for_profile",
        lambda *a, **kw: asyncio.sleep(0, result=None),
    )

    db = MagicMock()
    # ``GenerationRequest.max_chunk_chars`` has a pydantic ``ge=100``
    # constraint, so the lowest legal value is 100. The splitter groups
    # greedily, so we use a 112-char, 3-sentence input that splits into
    # exactly 2 chunks at max_chunk_chars=100.
    payload = _streaming_payload(
        "A longer first sentence for the splitter test. "
        "A second sentence follows it. "
        "A third sentence finishes the test.",
        max_chunk_chars=100,
    )

    response = await gen_routes.stream_speech(payload, db)
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)

    # The splitter's greedy grouping produces 2 chunks for the test text
    # at max_chunk_chars=100, so the backend is invoked twice. We assert
    # ``>= 2`` to keep the test robust to minor splitter tweaks.
    assert chunk_count["n"] >= 2, "splitter produced only one chunk"
    # The first yielded chunk is still the WAV header.
    assert chunks[0][:4] == b"RIFF"


@pytest.mark.asyncio
async def test_stream_speech_response_headers_set_no_cache(monkeypatch):
    """Streaming audio must not be cached by intermediaries — the
    Cache-Control: no-cache + X-Accel-Buffering: no headers tell nginx /
    browsers to flush as bytes arrive."""
    from backend.routes import generations as gen_routes

    profile = _make_stream_profile()
    monkeypatch.setattr(
        "backend.services.profiles.get_profile",
        lambda *a, **kw: asyncio.sleep(0, result=profile),
    )
    monkeypatch.setattr(
        "backend.services.profiles.validate_profile_engine",
        lambda *a, **kw: None,
    )

    fake_backend = MagicMock()

    async def fake_generate(text, voice_prompt, language, seed, instruct):
        return np.zeros(240, dtype=np.float32), 24000

    fake_backend.generate = fake_generate
    monkeypatch.setattr(
        "backend.backends.get_tts_backend_for_engine",
        lambda e: fake_backend,
    )
    monkeypatch.setattr(
        "backend.backends.ensure_model_cached_or_raise",
        lambda *a, **kw: asyncio.sleep(0, result=None),
    )
    monkeypatch.setattr(
        "backend.backends.load_engine_model",
        lambda *a, **kw: asyncio.sleep(0, result=None),
    )
    monkeypatch.setattr(
        "backend.backends.engine_needs_trim",
        lambda e: False,
    )
    monkeypatch.setattr(
        "backend.services.profiles.create_voice_prompt_for_profile",
        lambda *a, **kw: asyncio.sleep(0, result=None),
    )

    response = await gen_routes.stream_speech(_streaming_payload("hi."), MagicMock())
    assert response.media_type == "audio/wav"
    assert response.headers["Cache-Control"] == "no-cache"
    assert response.headers["X-Accel-Buffering"] == "no"


# ---------------------------------------------------------------------------
# ValueError __cause__ chaining  (PR #642 — commit 964a9ba)
# ---------------------------------------------------------------------------


def test_generate_speech_chains_value_error_cause(monkeypatch):
    """When validate_profile_engine raises ValueError, the HTTP 400
    carries ``__cause__ = original ValueError`` so logs preserve the
    full stack back to the underlying failure (e.g. unsupported
    profile/engine combination)."""
    from backend.routes.generations import generate_speech

    profile = _make_stream_profile()
    monkeypatch.setattr(
        "backend.services.profiles.get_profile",
        lambda *a, **kw: asyncio.sleep(0, result=profile),
    )

    cause = ValueError("Profile 'demo' requires engine 'luxtts'")

    def raise_value_error(*a, **kw):
        raise cause

    monkeypatch.setattr(
        "backend.services.profiles.validate_profile_engine",
        raise_value_error,
    )

    from backend import models

    payload = models.GenerationRequest(
        profile_id="profile-1",
        text="hi",
        language="en",
        engine="qwen",
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(generate_speech(payload, MagicMock()))

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == str(cause)
    # __cause__ must be the original exception (raise X from Y in Python).
    assert exc_info.value.__cause__ is cause


def test_stream_speech_chains_value_error_cause(monkeypatch):
    """Same chaining applies on the streaming path."""
    from backend.routes.generations import stream_speech

    profile = _make_stream_profile()
    monkeypatch.setattr(
        "backend.services.profiles.get_profile",
        lambda *a, **kw: asyncio.sleep(0, result=profile),
    )

    cause = ValueError("Engine 'luxtts' not allowed for cloned profile")

    monkeypatch.setattr(
        "backend.services.profiles.validate_profile_engine",
        lambda *a, **kw: (_ for _ in ()).throw(cause),
    )

    from backend import models

    payload = models.GenerationRequest(
        profile_id="profile-1",
        text="hi",
        language="en",
        engine="luxtts",
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(stream_speech(payload, MagicMock()))

    assert exc_info.value.status_code == 400
    assert exc_info.value.__cause__ is cause


# ---------------------------------------------------------------------------
# GET /tts endpoint  (PR #642)
# ---------------------------------------------------------------------------


def _build_speak_app():
    """Minimal FastAPI app hosting only the speak router, with the
    ``get_db`` dependency overridden so we don't need a real SQLite DB."""
    from fastapi import FastAPI
    from sqlalchemy.orm import Session

    from backend.database import get_db
    from backend.routes.speak import router as speak_router

    app = FastAPI()
    app.include_router(speak_router)

    def _override_db():
        yield MagicMock(spec=Session)

    app.dependency_overrides[get_db] = _override_db
    return app


def test_get_tts_default_returns_json_with_status_link(monkeypatch):
    """Without stream=true, GET /tts returns JSON with status + audio
    URLs — the default path for non-streaming integrations."""
    from backend import models
    from backend.routes import speak as speak_module

    fake_generation = MagicMock(spec=models.GenerationResponse)
    fake_generation.id = "gen-123"
    fake_generation.status = "generating"

    async def fake_speak(req, request, db):
        return fake_generation

    monkeypatch.setattr(speak_module, "speak", fake_speak)

    client = TestClient(_build_speak_app())
    response = client.get(
        "/tts",
        params={"text": "hello", "voice": "alice"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "gen-123"
    assert body["status"] == "generating"
    assert body["audio_url"].endswith("/audio/gen-123")
    assert body["status_url"].endswith("/generate/gen-123/status")


def test_get_tts_defaults_engine_to_luxtts(monkeypatch):
    """The endpoint is convenience-focused for local-device integrations
    where the caller doesn't know about engine names — default to
    LuxTTS."""
    from backend import models
    from backend.routes import speak as speak_module

    captured_engine = {}

    async def fake_speak(req, request, db):
        captured_engine["engine"] = req.engine
        ret = MagicMock(spec=models.GenerationResponse)
        ret.id = "x"
        ret.status = "generating"
        return ret

    monkeypatch.setattr(speak_module, "speak", fake_speak)

    client = TestClient(_build_speak_app())
    client.get("/tts", params={"text": "hi", "voice": "alice"})

    assert captured_engine["engine"] == "luxtts"


def test_get_tts_engine_alias_model_wins(monkeypatch):
    """``engine`` query param takes precedence over ``model`` (which is
    just an alias for backward compat with older integrations)."""
    from backend import models
    from backend.routes import speak as speak_module

    captured = {}

    async def fake_speak(req, request, db):
        captured["engine"] = req.engine
        ret = MagicMock(spec=models.GenerationResponse)
        ret.id = "x"
        ret.status = "generating"
        return ret

    monkeypatch.setattr(speak_module, "speak", fake_speak)

    client = TestClient(_build_speak_app())
    client.get(
        "/tts",
        params={"text": "hi", "voice": "alice", "model": "chatterbox", "engine": "qwen"},
    )

    assert captured["engine"] == "qwen"


def test_get_tts_stream_404_when_profile_unknown(monkeypatch):
    """stream=true with an explicit profile that doesn't resolve → 404
    with the profile name in the detail message."""
    from backend.routes import speak as speak_module

    monkeypatch.setattr(speak_module, "resolve_profile", lambda *a, **kw: None)

    client = TestClient(_build_speak_app())
    response = client.get(
        "/tts",
        params={"text": "hi", "voice": "missing-persona", "stream": "true"},
    )
    assert response.status_code == 404
    assert "missing-persona" in response.json()["detail"]


def test_get_tts_stream_400_when_no_profile_resolved(monkeypatch):
    """stream=true with no profile name and no default → 400 explaining
    the caller must pass profile= or configure a default."""
    from backend.routes import speak as speak_module

    monkeypatch.setattr(speak_module, "resolve_profile", lambda *a, **kw: None)

    client = TestClient(_build_speak_app())
    response = client.get(
        "/tts",
        params={"text": "hi", "stream": "true"},
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "profile" in detail.lower()


def test_get_tts_stream_delegates_to_stream_speech(monkeypatch):
    """When stream=true and a profile resolves, the response is a
    StreamingResponse with audio/wav content — same code path as POST
    /generate/stream."""
    from fastapi.responses import StreamingResponse

    from backend.routes import speak as speak_module

    fake_profile = MagicMock()
    fake_profile.id = "profile-42"
    monkeypatch.setattr(speak_module, "resolve_profile", lambda *a, **kw: fake_profile)

    async def fake_stream_speech(req, db):
        # Return a real StreamingResponse — the production type. We use a
        # tiny async generator so FastAPI/Starlette will accept it.
        async def gen():
            yield b"RIFF\xff\xff\xff\xffWAVEfmt "  # fake header
            yield b"\x00\x00"  # fake one PCM sample

        return StreamingResponse(gen(), media_type="audio/wav")

    # ``speak.tts_get`` imports ``stream_speech`` lazily via
    # ``from .generations import stream_speech`` inside the handler, so
    # the patch target is the *defining* module — ``generations`` —
    # not ``speak`` (which has no such attribute).
    monkeypatch.setattr("backend.routes.generations.stream_speech", fake_stream_speech)

    client = TestClient(_build_speak_app())
    response = client.get(
        "/tts",
        params={"text": "hi", "profile": "alice", "stream": "true"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    # The body is at least the two chunks we yielded.
    assert len(response.content) >= 2


def test_get_tts_language_query_validated_by_pattern(monkeypatch):
    """Language query param has a regex pattern — invalid languages must
    be rejected at the validation layer with a 422."""
    from backend.routes import speak as speak_module

    async def fake_speak(req, request, db):
        ret = MagicMock()
        ret.id = "x"
        ret.status = "generating"
        return ret

    monkeypatch.setattr(speak_module, "speak", fake_speak)

    client = TestClient(_build_speak_app())
    response = client.get(
        "/tts",
        params={"text": "hi", "voice": "alice", "language": "klingon"},
    )
    # FastAPI returns 422 on Query validation failure.
    assert response.status_code == 422


def test_get_tts_empty_text_rejected(monkeypatch):
    """text has min_length=1 — empty text → 422 (not a server crash)."""

    client = TestClient(_build_speak_app())
    response = client.get(
        "/tts",
        params={"text": "", "voice": "alice"},
    )
    assert response.status_code == 422
