"""
Platform detection for backend selection.
"""

import logging
import os
import platform
import re
import subprocess
from typing import Literal, Optional

logger = logging.getLogger(__name__)


def configure_rocm_gpu() -> None:
    """Detect AMD GPU and set HSA_OVERRIDE_GFX_VERSION only for older GPUs.

    Uses ``rocminfo`` to collect all GPU versions, finds the oldest (lowest
    gfx number), and only sets the override for RDNA 2 and older (gfx < 1100).
    RDNA 3+ (gfx1100+) and RDNA 4 (gfx1200+) have native ROCm support and the
    override can cause suboptimal performance or errors.

    Safe defaults: on non-AMD systems (no rocminfo, no GPUs found, parse
    errors), no override is set and no exception is raised.
    """
    if os.environ.get("HSA_OVERRIDE_GFX_VERSION"):
        return
    try:
        result = subprocess.run(
            ["rocminfo"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return
        # Collect all GPUs found in rocminfo output
        gfx_versions: list[str] = []
        for line in result.stdout.splitlines():
            line_lower = line.lower()
            if "gfx" in line_lower:
                match = re.search(r"(gfx\d+)", line_lower)
                if match:
                    gfx_versions.append(match.group(1))

        if not gfx_versions:
            return

        # Check if any GPU needs the override (RDNA 2 and older)
        # Use the oldest GPU (lowest gfx number) for the decision
        try:
            gfx_nums: list[int] = []
            for v in gfx_versions:
                m = re.search(r"\d+", v)
                if m:
                    gfx_nums.append(int(m.group()))
            if not gfx_nums:
                return
            oldest_num = min(gfx_nums)
            oldest_gfx = gfx_versions[gfx_nums.index(oldest_num)]
            if oldest_num < 1100:
                os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
                logger.info(
                    "AMD GPU detected (%s), setting HSA_OVERRIDE_GFX_VERSION=10.3.0 for compatibility. All GPUs: %s",
                    oldest_gfx,
                    ", ".join(gfx_versions),
                )
            else:
                logger.info(
                    "AMD GPU detected (%s), native ROCm support available, skipping HSA_OVERRIDE_GFX_VERSION. All GPUs: %s",
                    oldest_gfx,
                    ", ".join(gfx_versions),
                )
        except (ValueError, AttributeError) as e:
            logger.info("Could not parse GPU version from rocminfo output: %s", e)
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
        logger.info(
            "Could not detect AMD GPU via rocminfo, skipping automatic HSA_OVERRIDE_GFX_VERSION configuration: %s",
            e,
        )


def is_apple_silicon() -> bool:
    """
    Check if running on Apple Silicon (arm64 macOS).

    Returns:
        True if on Apple Silicon, False otherwise
    """
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def get_cuda_arch() -> Optional[str]:
    """Return the SM architecture string for the primary CUDA GPU, or None.

    Examples: ``"sm_90"`` for an RTX 4090, ``"sm_120"`` for an RTX 5090
    (Blackwell).  Returns ``None`` when no CUDA GPU is present or torch is
    not installed.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability(0)
        return f"sm_{major}{minor}"
    except Exception:
        return None


def get_backend_type() -> Literal["mlx", "pytorch"]:
    """
    Detect the best backend for the current platform.

    Returns:
        "mlx" on Apple Silicon (if MLX is available and functional), "pytorch" otherwise
    """
    if is_apple_silicon():
        try:
            import mlx.core  # noqa: F401 — triggers native lib loading
            return "mlx"
        except (ImportError, OSError, RuntimeError):
            # MLX not installed, or native libraries failed to load inside a
            # PyInstaller bundle (OSError on missing .dylib / .metallib).
            # Fall through to PyTorch.
            return "pytorch"
    return "pytorch"
