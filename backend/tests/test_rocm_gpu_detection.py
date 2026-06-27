"""Regression tests for AMD ROCm GPU detection (PR #785).

Tests cover all three acceptance criteria from the task:
- Non-AMD systems: no override set, no crash
- RDNA 2 (gfx < 1100): override set to 10.3.0
- RDNA 3+ (gfx >= 1100): no override set

Additional edge cases: multi-GPU (oldest decides), rocminfo non-zero exit,
rocminfo timeout, rocminfo not found, env var pre-set (respected), parse errors.

The detection logic lives in ``backend.utils.platform_detect`` which has
no heavy ML dependencies, so tests run without torch/fastapi installed.
"""

import os
import subprocess
from unittest.mock import patch

import pytest

from backend.utils.platform_detect import configure_rocm_gpu


def _fake_completed(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess:
    """Build a CompletedProcess mimicking a successful rocminfo run."""
    return subprocess.CompletedProcess(
        args=["rocminfo"],
        returncode=returncode,
        stdout=stdout,
        stderr="",
    )


@pytest.fixture
def clean_env(monkeypatch):
    """Ensure HSA_OVERRIDE_GFX_VERSION is not set before each test."""
    monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)


# ─── Acceptance criteria ────────────────────────────────────────────────


class TestAcceptanceCriteria:
    """The three scenarios spelled out in the task body."""

    def test_non_amd_system_no_override_no_crash(self, clean_env):
        """rocminfo not installed (FileNotFoundError) → no override, no crash."""
        with patch("backend.utils.platform_detect.subprocess.run", side_effect=FileNotFoundError("rocminfo not found")):
            configure_rocm_gpu()
        assert "HSA_OVERRIDE_GFX_VERSION" not in os.environ

    def test_rdna2_sets_override(self, clean_env):
        """RDNA 2 (gfx1030) → HSA_OVERRIDE_GFX_VERSION set to 10.3.0."""
        rocminfo_output = (
            "ROCm Systems Management Interface\n"
            "Agent 0: gfx1030\n"
            "  Name: gfx1030\n"
            "  Marketing Name: AMD Radeon RX 6800\n"
        )
        with patch("backend.utils.platform_detect.subprocess.run", return_value=_fake_completed(rocminfo_output)):
            configure_rocm_gpu()
        assert os.environ.get("HSA_OVERRIDE_GFX_VERSION") == "10.3.0"

    def test_rdna3_no_override(self, clean_env):
        """RDNA 3 (gfx1101) → no override set (native ROCm support)."""
        rocminfo_output = (
            "ROCm Systems Management Interface\n"
            "Agent 0: gfx1101\n"
            "  Name: gfx1101\n"
            "  Marketing Name: AMD Radeon RX 7900 XT\n"
        )
        with patch("backend.utils.platform_detect.subprocess.run", return_value=_fake_completed(rocminfo_output)):
            configure_rocm_gpu()
        assert "HSA_OVERRIDE_GFX_VERSION" not in os.environ


# ─── Multi-GPU ──────────────────────────────────────────────────────────


class TestMultiGPU:
    """Oldest GPU in a multi-GPU system decides whether to override."""

    def test_mixed_rdna2_and_rdna3_oldest_wins(self, clean_env):
        """gfx1030 + gfx1101 → oldest is gfx1030 (< 1100) → override set."""
        rocminfo_output = (
            "Agent 0: gfx1030\n"
            "  Name: gfx1030\n"
            "Agent 1: gfx1101\n"
            "  Name: gfx1101\n"
        )
        with patch("backend.utils.platform_detect.subprocess.run", return_value=_fake_completed(rocminfo_output)):
            configure_rocm_gpu()
        assert os.environ.get("HSA_OVERRIDE_GFX_VERSION") == "10.3.0"

    def test_two_rdna3_cards_no_override(self, clean_env):
        """gfx1100 + gfx1101 → both >= 1100 → no override."""
        rocminfo_output = (
            "Agent 0: gfx1100\n"
            "Agent 1: gfx1101\n"
        )
        with patch("backend.utils.platform_detect.subprocess.run", return_value=_fake_completed(rocminfo_output)):
            configure_rocm_gpu()
        assert "HSA_OVERRIDE_GFX_VERSION" not in os.environ

    def test_rdna4_no_override(self, clean_env):
        """RDNA 4 (gfx1201) → no override."""
        rocminfo_output = "Agent 0: gfx1201\n  Name: gfx1201\n"
        with patch("backend.utils.platform_detect.subprocess.run", return_value=_fake_completed(rocminfo_output)):
            configure_rocm_gpu()
        assert "HSA_OVERRIDE_GFX_VERSION" not in os.environ


# ─── Error handling / safe fallback ─────────────────────────────────────


class TestErrorHandling:
    """No crash on any failure path; no override set on failure."""

    def test_rocminfo_nonzero_exit_no_override(self, clean_env):
        """rocminfo returns non-zero → no override, no crash."""
        with patch("backend.utils.platform_detect.subprocess.run", return_value=_fake_completed("", returncode=1)):
            configure_rocm_gpu()
        assert "HSA_OVERRIDE_GFX_VERSION" not in os.environ

    def test_rocminfo_timeout_no_override(self, clean_env):
        """rocminfo times out → no override, no crash."""
        with patch("backend.utils.platform_detect.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="rocminfo", timeout=5)):
            configure_rocm_gpu()
        assert "HSA_OVERRIDE_GFX_VERSION" not in os.environ

    def test_no_gfx_in_output_no_override(self, clean_env):
        """rocminfo runs but no gfx lines → no override."""
        with patch("backend.utils.platform_detect.subprocess.run", return_value=_fake_completed("No GPU info here\n")):
            configure_rocm_gpu()
        assert "HSA_OVERRIDE_GFX_VERSION" not in os.environ

    def test_empty_output_no_override(self, clean_env):
        """rocminfo returns empty stdout → no override."""
        with patch("backend.utils.platform_detect.subprocess.run", return_value=_fake_completed("")):
            configure_rocm_gpu()
        assert "HSA_OVERRIDE_GFX_VERSION" not in os.environ

    def test_generic_exception_no_crash(self, clean_env):
        """Any unexpected exception → caught, no crash, no override."""
        with patch("backend.utils.platform_detect.subprocess.run", side_effect=RuntimeError("unexpected")):
            configure_rocm_gpu()
        assert "HSA_OVERRIDE_GFX_VERSION" not in os.environ


# ─── Env var pre-set (respected) ────────────────────────────────────────


class TestPreSetEnvVar:
    """If HSA_OVERRIDE_GFX_VERSION is already set, the function is a no-op."""

    def test_preset_env_var_respected(self, monkeypatch):
        """User-set override must not be overwritten."""
        monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
        with patch("backend.utils.platform_detect.subprocess.run") as mock_run:
            configure_rocm_gpu()
        mock_run.assert_not_called()
        assert os.environ.get("HSA_OVERRIDE_GFX_VERSION") == "11.0.0"


# ─── Boundary conditions ───────────────────────────────────────────────


class TestBoundary:
    """The gfx1100 boundary is the critical threshold."""

    def test_exactly_gfx1100_no_override(self, clean_env):
        """gfx1100 is the first RDNA 3 → no override (>= 1100)."""
        with patch("backend.utils.platform_detect.subprocess.run", return_value=_fake_completed("Agent 0: gfx1100\n")):
            configure_rocm_gpu()
        assert "HSA_OVERRIDE_GFX_VERSION" not in os.environ

    def test_gfx1099_sets_override(self, clean_env):
        """gfx1099 is the last RDNA 2 → override set (< 1100)."""
        with patch("backend.utils.platform_detect.subprocess.run", return_value=_fake_completed("Agent 0: gfx1099\n")):
            configure_rocm_gpu()
        assert os.environ.get("HSA_OVERRIDE_GFX_VERSION") == "10.3.0"


# ─── Defensive guards (synthetic failures) ──────────────────────────────


class TestDefensiveGuards:
    """Cover the inner try/except and ``if not gfx_nums: return`` guards.

    These branches cannot be reached through real rocminfo output (the outer
    regex ``gfx\\d+`` guarantees the inner ``\\d+`` always matches), so the
    tests below inject synthetic failures via ``unittest.mock.patch`` to
    exercise the defensive paths. The guards themselves came from upstream
    PR #785 verbatim — preserving them keeps the cherry-pick faithful.
    """

    def test_inner_regex_raises_attributeerror_no_crash(self, clean_env):
        """``re.search`` raising AttributeError inside the parsing block
        is caught by the inner ``except (ValueError, AttributeError)`` and
        logged — no override, no crash."""
        rocminfo_output = "Agent 0: gfx1030\n"
        original_search = __import__("re").search

        def fake_search(pattern, string, *args, **kwargs):
            # The outer regex call passes — return a match so gfx_versions
            # gets populated. The inner call (r"\\d+", v) raises AttributeError.
            if pattern.startswith(r"(gfx"):
                return original_search(pattern, string, *args, **kwargs)
            raise AttributeError("synthetic inner-search failure")

        with patch("backend.utils.platform_detect.subprocess.run", return_value=_fake_completed(rocminfo_output)), \
             patch("backend.utils.platform_detect.re.search", side_effect=fake_search):
            configure_rocm_gpu()
        assert "HSA_OVERRIDE_GFX_VERSION" not in os.environ

    def test_gfx_nums_empty_inner_guard(self, clean_env):
        """All gfx matches yield strings without digits (synthetic) →
        ``gfx_nums`` ends up empty → function returns without setting override."""
        # Use a non-standard pattern where the outer regex picks up "gfx"
        # but the inner "\\d+" extraction is forced to return None by mocking.
        rocminfo_output = "Agent 0: gfx1030\n"

        def fake_inner_search(pattern, string, *args, **kwargs):
            # pattern is r"\\d+"; return None for every call → no nums.
            return None

        original_search = __import__("re").search

        def selective_search(pattern, string, *args, **kwargs):
            if pattern.startswith(r"(gfx"):
                return original_search(pattern, string, *args, **kwargs)
            return fake_inner_search(pattern, string, *args, **kwargs)

        with patch("backend.utils.platform_detect.subprocess.run", return_value=_fake_completed(rocminfo_output)), \
             patch("backend.utils.platform_detect.re.search", side_effect=selective_search):
            configure_rocm_gpu()
        assert "HSA_OVERRIDE_GFX_VERSION" not in os.environ
