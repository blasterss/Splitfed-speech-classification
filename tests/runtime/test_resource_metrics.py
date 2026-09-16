import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset

from src.dataset.dataset import EmotionalDataset
from src.utils.runtime.resource_metrics import (
    RESOURCE_MEASUREMENT_POLICY,
    RESOURCE_METRICS_SCHEMA_VERSION,
    ResourceTracker,
    owned_resource_bytes,
    tensor_storage_bytes,
)


def test_resource_tracker_reports_cpu_interval_and_throughput():
    tracker = ResourceTracker("client", torch.device("cpu"), client_id=2)
    metric = tracker.snapshot(round_idx=3, phase="train", samples=8, batches=2)

    assert metric["schema_version"] == RESOURCE_METRICS_SCHEMA_VERSION
    assert metric["measurement_policy"] == RESOURCE_MEASUREMENT_POLICY
    assert metric["process_id"] > 0
    assert metric["role"] == "client"
    assert metric["client_id"] == 2
    assert metric["round"] == 3
    assert metric["wall_time_seconds"] >= 0
    assert metric["process_peak_rss_bytes"] > 0
    assert metric["phase_start_rss_bytes"] > 0
    assert metric["phase_end_rss_bytes"] > 0
    assert (
        metric["phase_sampled_peak_rss_bytes"]
        >= metric["phase_start_rss_bytes"]
    )
    assert metric["samples_per_second"] >= 0
    assert metric["peak_cuda_allocated_bytes"] is None


def test_tensor_storage_bytes_does_not_double_count_shared_views():
    tensor = torch.zeros(8, dtype=torch.float32)

    assert tensor_storage_bytes(tensor, tensor.view(2, 4)) == 32


def test_owned_resource_bytes_separates_model_optimizer_and_dataset():
    model = nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters())
    dataset = EmotionalDataset(
        np.ones((2, 2, 3), dtype=np.float32),
        np.asarray([0.0, 1.0], dtype=np.float32),
        mean=np.ones(2, dtype=np.float32),
        std=np.ones(2, dtype=np.float32),
        valid_frames=np.asarray([3, 3], dtype=np.int64),
    )

    before = owned_resource_bytes(model, optimizer, dataset)
    loss = model(torch.ones(2, 2)).sum()
    loss.backward()
    optimizer.step()
    after = owned_resource_bytes(model, optimizer, dataset)

    assert before["model_parameter_bytes"] == 12
    assert before["model_buffer_bytes"] == 0
    assert before["optimizer_state_bytes"] == 0
    assert before["owned_dataset_bytes"] == 88
    assert before["owned_static_bytes"] == 100
    assert after["optimizer_state_bytes"] > 0
    assert after["owned_static_bytes"] > before["owned_static_bytes"]

    combined = owned_resource_bytes(
        model, optimizer, ConcatDataset([dataset, dataset])
    )
    assert combined["owned_dataset_bytes"] == 88
