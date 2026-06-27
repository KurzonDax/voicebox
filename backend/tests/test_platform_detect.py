"""Tests for pre-existing functions in backend.utils.platform_detect.

These tests cover ``is_apple_silicon()`` and ``get_backend_type()``, which
share a module with the newer ``configure_rocm_gpu()`` (PR #785). They were
not previously tested and exist on disk; the ROCm PR brought them into the
review spotlight, so we cover them here.
"""

import sys
from unittest.mock import patch

import pytest

from backend.utils.platform_detect import (
    get_backend_type,
    is_apple_silicon,
)


class TestIsAppleSilicon:
    """``is_apple_silicon()`` returns True only for Darwin + arm64."""

    def test_apple_silicon(self):
        with patch("backend.utils.platform_detect.platform.system", return_value="Darwin"), \
             patch("backend.utils.platform_detect.platform.machine", return_value="arm64"):
            assert is_apple_silicon() is True

    def test_intel_mac(self):
        with patch("backend.utils.platform_detect.platform.system", return_value="Darwin"), \
             patch("backend.utils.platform_detect.platform.machine", return_value="x86_64"):
            assert is_apple_silicon() is False

    def test_linux_arm64(self):
        with patch("backend.utils.platform_detect.platform.system", return_value="Linux"), \
             patch("backend.utils.platform_detect.platform.machine", return_value="aarch64"):
            assert is_apple_silicon() is False

    def test_windows_x64(self):
        with patch("backend.utils.platform_detect.platform.system", return_value="Windows"), \
             patch("backend.utils.platform_detect.platform.machine", return_value="AMD64"):
            assert is_apple_silicon() is False


class TestGetBackendType:
    """``get_backend_type()`` returns 'mlx' on Apple Silicon with mlx available, 'pytorch' otherwise."""

    def test_non_apple_returns_pytorch(self):
        with patch("backend.utils.platform_detect.is_apple_silicon", return_value=False):
            assert get_backend_type() == "pytorch"

    def test_apple_silicon_with_mlx_returns_mlx(self):
        # Simulate mlx.core importable
        fake_mlx = type(sys)("fake_mlx")  # build a module-like namespace
        fake_mlx_core = type(sys)("fake_mlx_core")
        with patch.dict(sys.modules, {"mlx": fake_mlx, "mlx.core": fake_mlx_core}), \
             patch("backend.utils.platform_detect.is_apple_silicon", return_value=True):
            assert get_backend_type() == "mlx"

    def test_apple_silicon_without_mlx_returns_pytorch(self):
        # Force ImportError on `import mlx.core`
        with patch.dict(sys.modules, {"mlx": None, "mlx.core": None}), \
             patch("backend.utils.platform_detect.is_apple_silicon", return_value=True):
            # Importing None raises ImportError, matching the real-world failure path.
            assert get_backend_type() == "pytorch"

    def test_apple_silicon_mlx_oserror_returns_pytorch(self):
        # Simulate OSError raised when native libraries are missing in a PyInstaller bundle.
        with patch.dict(sys.modules, {}), \
             patch("backend.utils.platform_detect.is_apple_silicon", return_value=True), \
             patch("builtins.__import__", side_effect=OSError("missing .metallib")):
            assert get_backend_type() == "pytorch"
