"""Low-overhead resource measurements emitted by training workers."""

from __future__ import annotations

import os
import resource
import time
from dataclasses import dataclass, field

import torch

RESOURCE_METRICS_SCHEMA_VERSION = 1


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB, while macOS reports bytes.
    return int(value if os.uname().sysname == "Darwin" else value * 1024)


@dataclass
class ResourceTracker:
    """Capture elapsed, CPU, RSS and CUDA peaks for one process interval."""

    role: str
    device: torch.device | str = "cpu"
    client_id: str | int | None = None
    _wall_start: float = field(init=False)
    _usage_start: resource.struct_rusage = field(init=False)

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        self.reset()

    def reset(self) -> None:
        self._wall_start = time.perf_counter()
        self._usage_start = resource.getrusage(resource.RUSAGE_SELF)
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

    def snapshot(
        self,
        *,
        round_idx: int | None,
        phase: str,
        samples: int | None = None,
        batches: int | None = None,
    ) -> dict:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        elapsed = time.perf_counter() - self._wall_start
        cuda_allocated = cuda_reserved = None
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
            cuda_allocated = torch.cuda.max_memory_allocated(self.device)
            cuda_reserved = torch.cuda.max_memory_reserved(self.device)
        return {
            "schema_version": RESOURCE_METRICS_SCHEMA_VERSION,
            "role": self.role,
            "client_id": self.client_id,
            "round": round_idx,
            "phase": phase,
            "wall_time_seconds": elapsed,
            "cpu_user_seconds": usage.ru_utime - self._usage_start.ru_utime,
            "cpu_system_seconds": (
                usage.ru_stime - self._usage_start.ru_stime
            ),
            "peak_rss_bytes": _peak_rss_bytes(),
            "peak_cuda_allocated_bytes": cuda_allocated,
            "peak_cuda_reserved_bytes": cuda_reserved,
            "samples": samples,
            "batches": batches,
            "samples_per_second": (
                samples / elapsed if samples is not None and elapsed else None
            ),
            "batches_per_second": (
                batches / elapsed if batches is not None and elapsed else None
            ),
        }


def publish_resource_metric(queue, metric: dict) -> None:
    """Best-effort publish; metrics must never block model shutdown."""
    if queue is None:
        return
    try:
        queue.put(metric, timeout=1)
    except Exception:
        return
