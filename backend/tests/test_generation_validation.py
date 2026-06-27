"""Tests for backend.services.generation._validate_generated_audio.

Validates that the guard function catches invalid TTS output (None audio,
empty arrays, non-finite values, invalid sample rates) before it can be
saved as a completed generation.
"""

import numpy as np
import pytest

from backend.services.generation import _validate_generated_audio


class TestValidateGeneratedAudio:
    """Coverage for _validate_generated_audio guard."""

    def test_valid_audio_passes(self):
        """Normal finite audio with a positive sample rate should not raise."""
        audio = np.ones(16000, dtype=np.float32)
        _validate_generated_audio(audio, 22050)

    def test_valid_list_audio_passes(self):
        """A plain Python list (not numpy) should be coerced and accepted."""
        _validate_generated_audio([0.1, -0.2, 0.3], 24000)

    def test_none_audio_raises(self):
        """None audio must raise RuntimeError."""
        with pytest.raises(RuntimeError, match="no audio"):
            _validate_generated_audio(None, 22050)

    def test_empty_audio_raises(self):
        """Empty array must raise RuntimeError."""
        with pytest.raises(RuntimeError, match="empty audio"):
            _validate_generated_audio(np.array([], dtype=np.float32), 22050)

    def test_empty_list_raises(self):
        """Empty list must raise RuntimeError."""
        with pytest.raises(RuntimeError, match="empty audio"):
            _validate_generated_audio([], 22050)

    def test_zero_sample_rate_raises(self):
        """sample_rate of 0 must raise RuntimeError."""
        with pytest.raises(RuntimeError, match="invalid sample rate"):
            _validate_generated_audio(np.ones(100, dtype=np.float32), 0)

    def test_negative_sample_rate_raises(self):
        """Negative sample_rate must raise RuntimeError."""
        with pytest.raises(RuntimeError, match="invalid sample rate"):
            _validate_generated_audio(np.ones(100, dtype=np.float32), -44100)

    def test_none_sample_rate_raises(self):
        """None sample_rate must raise RuntimeError."""
        with pytest.raises(RuntimeError, match="invalid sample rate"):
            _validate_generated_audio(np.ones(100, dtype=np.float32), None)

    def test_nan_audio_raises(self):
        """Audio containing NaN must raise RuntimeError."""
        audio = np.array([1.0, float("nan"), 0.5], dtype=np.float32)
        with pytest.raises(RuntimeError, match="non-finite"):
            _validate_generated_audio(audio, 22050)

    def test_inf_audio_raises(self):
        """Audio containing +Inf must raise RuntimeError."""
        audio = np.array([1.0, float("inf"), 0.5], dtype=np.float32)
        with pytest.raises(RuntimeError, match="non-finite"):
            _validate_generated_audio(audio, 22050)

    def test_neg_inf_audio_raises(self):
        """Audio containing -Inf must raise RuntimeError."""
        audio = np.array([1.0, float("-inf"), 0.5], dtype=np.float32)
        with pytest.raises(RuntimeError, match="non-finite"):
            _validate_generated_audio(audio, 22050)

    def test_multichannel_valid_passes(self):
        """2-D (multichannel) finite audio should pass."""
        audio = np.ones((2, 16000), dtype=np.float32)
        _validate_generated_audio(audio, 22050)