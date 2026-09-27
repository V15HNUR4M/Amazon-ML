"""
gpu_utils.py — GPU hardware and environment detection with CPU fallback.

Detects CUDA, RAPIDS cuDF, CuPy, and NVIDIA T4 GPU capabilities,
providing clean memory monitoring, VRAM safety checks, and seamless
CPU fallback for environments without CUDA.
"""

from __future__ import annotations

import gc
import logging
import os
import platform
import time
from dataclasses import dataclass
from typing import Any, Optional

import psutil

logger = logging.getLogger(__name__)

# Check torch availability
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore
    _HAS_TORCH = False

# Check RAPIDS cuDF availability
try:
    import cudf
    _HAS_CUDF = True
except ImportError:
    cudf = None  # type: ignore
    _HAS_CUDF = False

# Set up Windows CUDA DLL paths if available
import sys
from pathlib import Path

_NVIDIA_BIN_DIRS = [
    Path(sys.prefix) / "Lib" / "site-packages" / "nvidia" / "cuda_nvrtc" / "bin",
    Path(sys.prefix) / "Lib" / "site-packages" / "nvidia" / "cuda_runtime" / "bin",
]
for _p in _NVIDIA_BIN_DIRS:
    if _p.exists():
        os.environ["PATH"] = str(_p) + ";" + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(str(_p))
            except Exception:
                pass

# Check CuPy availability
try:
    import cupy
    _HAS_CUPY = True
except ImportError:
    cupy = None  # type: ignore
    _HAS_CUPY = False


def is_cuda_available() -> bool:
    """Return True if PyTorch or CuPy detects a functional CUDA device."""
    if _HAS_TORCH:
        try:
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                return True
        except Exception:
            pass
    if _HAS_CUPY:
        try:
            return bool(cupy.cuda.is_available())
        except Exception:
            pass
    return False


def is_cudf_available() -> bool:
    """Return True if RAPIDS cuDF is installed and functioning."""
    return _HAS_CUDF


def is_cupy_available() -> bool:
    """Return True if CuPy is installed and CUDA device is active."""
    if not _HAS_CUPY:
        return False
    try:
        return bool(cupy.cuda.is_available())
    except Exception:
        return False


def get_process_rss_mb() -> float:
    """Return current process resident set size (RSS) in MB."""
    try:
        return round(psutil.Process(os.getpid()).memory_info().rss / (1024**2), 2)
    except Exception:
        return -1.0


def get_gpu_memory_mb(device_index: int = 0) -> dict[str, float]:
    """Return current GPU VRAM statistics in MB.

    Returns zeros if CUDA is not available.
    """
    if not is_cuda_available():
        return {
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "peak_allocated_mb": 0.0,
            "total_vram_mb": 0.0,
            "free_vram_mb": 0.0,
        }

    if _HAS_TORCH:
        try:
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated(device_index) / (1024**2)
                reserved = torch.cuda.memory_reserved(device_index) / (1024**2)
                peak = torch.cuda.max_memory_allocated(device_index) / (1024**2)
                free, total = torch.cuda.mem_get_info(device_index)
                return {
                    "allocated_mb": round(alloc, 2),
                    "reserved_mb": round(reserved, 2),
                    "peak_allocated_mb": round(peak, 2),
                    "total_vram_mb": round(total / (1024**2), 2),
                    "free_vram_mb": round(free / (1024**2), 2),
                }
        except Exception as e:
            logger.debug("PyTorch memory read failed: %s", e)

    if _HAS_CUPY and is_cupy_available():
        try:
            dev = cupy.cuda.Device(device_index)
            free_b, total_b = dev.mem_info
            alloc_b = total_b - free_b
            return {
                "allocated_mb": round(alloc_b / (1024**2), 2),
                "reserved_mb": round(alloc_b / (1024**2), 2),
                "peak_allocated_mb": round(alloc_b / (1024**2), 2),
                "total_vram_mb": round(total_b / (1024**2), 2),
                "free_vram_mb": round(free_b / (1024**2), 2),
            }
        except Exception as e:
            logger.debug("CuPy memory read failed: %s", e)

    return {
        "allocated_mb": 0.0,
        "reserved_mb": 0.0,
        "peak_allocated_mb": 0.0,
        "total_vram_mb": 0.0,
        "free_vram_mb": 0.0,
    }


def clear_gpu_cache() -> None:
    """Garbage collect and release cached GPU memory back to OS/driver."""
    gc.collect()
    if _HAS_TORCH:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
    if _HAS_CUPY and is_cupy_available():
        try:
            cupy.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass


def get_device_info() -> dict[str, Any]:
    """Return comprehensive hardware, CUDA, and environment detection summary."""
    cuda_ok = is_cuda_available()
    device_name = "CPU Fallback"
    device_count = 0
    total_vram = 0.0
    cuda_version = None

    if cuda_ok:
        if _HAS_TORCH and torch.cuda.is_available():
            try:
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0)
                cuda_version = torch.version.cuda
                _, tot = torch.cuda.mem_get_info(0)
                total_vram = round(tot / (1024**2), 2)
            except Exception as e:
                logger.debug("Error querying torch device properties: %s", e)
        elif _HAS_CUPY and is_cupy_available():
            try:
                device_count = cupy.cuda.runtime.getDeviceCount()
                props = cupy.cuda.runtime.getDeviceProperties(0)
                device_name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
                dev = cupy.cuda.Device(0)
                _, total_b = dev.mem_info
                total_vram = round(total_b / (1024**2), 2)
                cuda_version = str(cupy.cuda.runtime.runtimeGetVersion())
            except Exception as e:
                logger.debug("Error querying cupy device properties: %s", e)

    return {
        "cuda_available": cuda_ok,
        "cudf_available": is_cudf_available(),
        "cupy_available": is_cupy_available(),
        "device_count": device_count,
        "device_name": device_name,
        "total_vram_mb": total_vram,
        "cuda_version": cuda_version,
        "torch_version": torch.__version__ if _HAS_TORCH else None,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "process_rss_mb": get_process_rss_mb(),
    }


@dataclass
class DeviceContext:
    """Context manager to measure runtime, peak RAM, and peak VRAM of an operation."""

    name: str = "operation"
    start_time: float = 0.0
    elapsed_s: float = 0.0
    baseline_rss_mb: float = 0.0
    peak_rss_mb: float = 0.0
    baseline_vram_mb: float = 0.0
    peak_vram_mb: float = 0.0

    def __enter__(self) -> DeviceContext:
        clear_gpu_cache()
        self.baseline_rss_mb = get_process_rss_mb()
        self.peak_rss_mb = self.baseline_rss_mb
        vram_info = get_gpu_memory_mb()
        self.baseline_vram_mb = vram_info["allocated_mb"]
        self.peak_vram_mb = self.baseline_vram_mb
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.elapsed_s = round(time.time() - self.start_time, 4)
        self.peak_rss_mb = max(self.peak_rss_mb, get_process_rss_mb())
        vram_info = get_gpu_memory_mb()
        self.peak_vram_mb = max(self.peak_vram_mb, vram_info["peak_allocated_mb"])
        logger.debug(
            "[%s] completed in %.3fs | RSS: %.1f -> %.1f MB | VRAM: %.1f -> %.1f MB",
            self.name,
            self.elapsed_s,
            self.baseline_rss_mb,
            self.peak_rss_mb,
            self.baseline_vram_mb,
            self.peak_vram_mb,
        )
