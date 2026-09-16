"""Low-overhead resource measurements emitted by training workers."""

from __future__ import annotations

import os
import resource
import threading
import time
from dataclasses import dataclass, field

import torch

RESOURCE_METRICS_SCHEMA_VERSION = 2
RESOURCE_MEASUREMENT_POLICY = "resource_metrics_v2"
RSS_SAMPLE_INTERVAL_SECONDS = 0.05


def _tensor_storage_identity(tensor: torch.Tensor) -> tuple:
    storage = tensor.untyped_storage()
    return (
        tensor.device.type,
        tensor.device.index,
        storage.data_ptr(),
        storage.nbytes(),
    )


def tensor_storage_bytes(*values) -> int:
    """Count unique tensor storages reachable from the supplied values."""
    storages: dict[tuple, int] = {}
    visited: set[int] = set()

    def visit(value) -> None:
        if isinstance(value, torch.Tensor):
            identity = _tensor_storage_identity(value)
            storages.setdefault(identity, value.untyped_storage().nbytes())
            return
        if isinstance(value, dict):
            identity = id(value)
            if identity in visited:
                return
            visited.add(identity)
            for item in value.values():
                visit(item)
            return
        if isinstance(value, (list, tuple, set)):
            identity = id(value)
            if identity in visited:
                return
            visited.add(identity)
            for item in value:
                visit(item)

    for value in values:
        visit(value)
    return sum(storages.values())


def owned_resource_bytes(
    model, optimizer=None, dataset=None
) -> dict[str, int]:
    """Return deterministic tensor ownership without double-counting views."""
    parameter_bytes = tensor_storage_bytes(*model.parameters())
    buffer_bytes = tensor_storage_bytes(*model.buffers())
    optimizer_bytes = (
        tensor_storage_bytes(optimizer.state) if optimizer is not None else 0
    )
    dataset_values = _dataset_tensor_values(dataset)
    dataset_bytes = tensor_storage_bytes(*dataset_values)
    return {
        "model_parameter_bytes": parameter_bytes,
        "model_buffer_bytes": buffer_bytes,
        "optimizer_state_bytes": optimizer_bytes,
        "owned_dataset_bytes": dataset_bytes,
        "owned_static_bytes": (
            parameter_bytes + buffer_bytes + optimizer_bytes + dataset_bytes
        ),
    }


def _dataset_tensor_values(dataset) -> list:
    if dataset is None:
        return []
    if isinstance(dataset, (list, tuple)):
        values = []
        for item in dataset:
            values.extend(_dataset_tensor_values(item))
        return values
    nested = getattr(dataset, "datasets", None)
    if nested is not None:
        return _dataset_tensor_values(nested)
    train_dataset = getattr(dataset, "train_dataset", None)
    test_dataset = getattr(dataset, "test_dataset", None)
    if train_dataset is not None or test_dataset is not None:
        return _dataset_tensor_values([train_dataset, test_dataset])
    wrapped = getattr(dataset, "dataset", None)
    if wrapped is not None:
        return _dataset_tensor_values(wrapped)
    return [
        getattr(dataset, field, None)
        for field in ("data", "labels", "mean", "std", "valid_frames")
    ]


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB, while macOS reports bytes.
    return int(value if os.uname().sysname == "Darwin" else value * 1024)


def _current_rss_bytes() -> int | None:
    if os.uname().sysname != "Linux":
        return None
    try:
        with open("/proc/self/status", encoding="ascii") as source:
            for line in source:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


@dataclass
class ResourceTracker:
    """Capture elapsed, CPU, RSS and CUDA peaks for one process interval."""

    role: str
    device: torch.device | str = "cpu"
    client_id: str | int | None = None
    _wall_start: float = field(init=False)
    _usage_start: resource.struct_rusage = field(init=False)
    _phase_start_rss: int | None = field(init=False, default=None)
    _phase_sampled_peak_rss: int | None = field(init=False, default=None)
    _rss_stop: threading.Event | None = field(init=False, default=None)
    _rss_thread: threading.Thread | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        self.reset()

    def reset(self) -> None:
        self._stop_rss_sampler()
        self._wall_start = time.perf_counter()
        self._usage_start = resource.getrusage(resource.RUSAGE_SELF)
        self._phase_start_rss = _current_rss_bytes()
        self._phase_sampled_peak_rss = self._phase_start_rss
        if self._phase_start_rss is not None:
            self._rss_stop = threading.Event()
            self._rss_thread = threading.Thread(
                target=self._sample_rss,
                name=f"rss-sampler-{self.role}",
                daemon=True,
            )
            self._rss_thread.start()
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

    def _sample_rss(self) -> None:
        while self._rss_stop is not None and not self._rss_stop.wait(
            RSS_SAMPLE_INTERVAL_SECONDS
        ):
            current = _current_rss_bytes()
            if current is not None:
                self._phase_sampled_peak_rss = max(
                    self._phase_sampled_peak_rss or 0, current
                )

    def _stop_rss_sampler(self) -> None:
        if self._rss_stop is not None:
            self._rss_stop.set()
        if self._rss_thread is not None:
            self._rss_thread.join(timeout=1)
        self._rss_stop = None
        self._rss_thread = None

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
        self._stop_rss_sampler()
        phase_end_rss = _current_rss_bytes()
        if phase_end_rss is not None:
            self._phase_sampled_peak_rss = max(
                self._phase_sampled_peak_rss or 0, phase_end_rss
            )
        cuda_allocated = cuda_reserved = None
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
            cuda_allocated = torch.cuda.max_memory_allocated(self.device)
            cuda_reserved = torch.cuda.max_memory_reserved(self.device)
        return {
            "schema_version": RESOURCE_METRICS_SCHEMA_VERSION,
            "measurement_policy": RESOURCE_MEASUREMENT_POLICY,
            "role": self.role,
            "client_id": self.client_id,
            "process_id": os.getpid(),
            "round": round_idx,
            "phase": phase,
            "wall_time_seconds": elapsed,
            "cpu_user_seconds": usage.ru_utime - self._usage_start.ru_utime,
            "cpu_system_seconds": (
                usage.ru_stime - self._usage_start.ru_stime
            ),
            "process_peak_rss_bytes": _peak_rss_bytes(),
            "phase_start_rss_bytes": self._phase_start_rss,
            "phase_end_rss_bytes": phase_end_rss,
            "phase_sampled_peak_rss_bytes": self._phase_sampled_peak_rss,
            "rss_measurement_backend": (
                "linux_proc_status_sampler_v1"
                if self._phase_start_rss is not None
                else "process_peak_only_v1"
            ),
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
