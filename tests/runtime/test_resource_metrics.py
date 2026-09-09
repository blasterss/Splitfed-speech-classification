import torch

from src.utils.runtime.resource_metrics import ResourceTracker


def test_resource_tracker_reports_cpu_interval_and_throughput():
    tracker = ResourceTracker("client", torch.device("cpu"), client_id=2)
    metric = tracker.snapshot(round_idx=3, phase="train", samples=8, batches=2)

    assert metric["schema_version"] == 1
    assert metric["role"] == "client"
    assert metric["client_id"] == 2
    assert metric["round"] == 3
    assert metric["wall_time_seconds"] >= 0
    assert metric["peak_rss_bytes"] > 0
    assert metric["samples_per_second"] >= 0
    assert metric["peak_cuda_allocated_bytes"] is None
