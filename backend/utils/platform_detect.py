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


def get_supported_platforms() -> list[str]:
    """Return which compute platforms the current machine supports.

    Possible values: "cuda", "mps", "xpu", "rocm", "cpu"

    Rules:
    - "cpu" is always included (every machine can run CPU inference).
    - "cuda" is added when PyTorch reports a CUDA device available.
    - "rocm" is added on ROCm builds (torch.version.hip is set).
    - "mps" is added when the Metal Performance Shaders backend is available.
    - "xpu" is added when Intel Extension for PyTorch detects an Arc/XPU device.

    Apple Silicon machines therefore return ["mps", "cpu"], a typical
    CUDA Linux machine returns ["cuda", "cpu"], an Intel Arc machine returns
    ["xpu", "cpu"], and a CPU-only machine returns ["cpu"].
    """
    supported: list[str] = []

    try:
        import torch

        if torch.cuda.is_available():
            # Distinguish ROCm from CUDA — both report via cuda.is_available()
            # on the ROCm PyTorch build, but torch.version.hip is non-None.
            is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
            if is_rocm:
                supported.append("rocm")
            else:
                supported.append("cuda")

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            supported.append("mps")

        try:
            import intel_extension_for_pytorch  # noqa: F401

            if hasattr(torch, "xpu") and torch.xpu.is_available():
                supported.append("xpu")
        except ImportError:
            pass

    except ImportError:
        pass  # torch not available at all — only CPU

    supported.append("cpu")
    return supported
