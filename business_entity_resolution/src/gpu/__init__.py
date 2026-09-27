"""
GPU-accelerated modules for Turn 7 Business Entity Resolution.
Supports NVIDIA CUDA / T4 GPU with seamless CPU fallback.
"""

from src.gpu.gpu_blocking import (
    DiskGPUBlockingIndex,
    GPUBlockingConfig,
    GPUBlockingIndex,
    ShardedReferenceIndex,
    build_gpu_candidates,
    extract_core_domain,
)
from src.gpu.gpu_features import (
    BASELINE_FEATURE_NAMES,
    TURN7_FEATURE_NAMES,
    TokenIDFStore,
    compute_gpu_features,
)
from src.gpu.gpu_pipeline import Turn7Config, Turn7Pipeline
from src.gpu.gpu_utils import (
    DeviceContext,
    clear_gpu_cache,
    get_device_info,
    get_gpu_memory_mb,
    get_process_rss_mb,
    is_cuda_available,
    is_cudf_available,
    is_cupy_available,
)

__all__ = [
    "DeviceContext",
    "DiskGPUBlockingIndex",
    "GPUBlockingConfig",
    "GPUBlockingIndex",
    "ShardedReferenceIndex",
    "Turn7Config",
    "Turn7Pipeline",
    "TokenIDFStore",
    "BASELINE_FEATURE_NAMES",
    "TURN7_FEATURE_NAMES",
    "build_gpu_candidates",
    "compute_gpu_features",
    "extract_core_domain",
    "get_device_info",
    "get_gpu_memory_mb",
    "get_process_rss_mb",
    "clear_gpu_cache",
    "is_cuda_available",
    "is_cudf_available",
    "is_cupy_available",
]
