import queue

import pytest

from src.schema import MergeSFLPolicyConfig
from src.splitfed.load_controller import (
    WorkerState,
    WorkerTelemetry,
    bootstrap_telemetry,
    collect_selected_telemetry,
    profiles_from_manifests,
)


def _config():
    return MergeSFLPolicyConfig(
        max_batch_size=8,
        local_steps=2,
        ingress_budget_bytes=1024,
        feature_bytes_per_sample=64,
        min_clients=1,
        max_clients=2,
        initial_worker_states={
            0: {
                "compute_seconds_per_sample": 0.02,
                "transfer_seconds_per_sample": 0.01,
            }
        },
    )


def test_runtime_builds_profiles_and_bootstrap_observations():
    profiles = profiles_from_manifests(
        {
            0: {
                "coverage": {
                    "train": {"class_0": 3, "class_1": 1, "samples": 4}
                }
            }
        },
        {0: 2},
    )
    telemetry = bootstrap_telemetry(_config(), observed_at=100.0)

    assert profiles[0].label_distribution == (0.75, 0.25)
    assert profiles[0].participation_count == 2
    assert profiles[0].train_samples == 4
    assert telemetry[0].state == WorkerState(0.02, 0.01)
    assert telemetry[0].observed_at == 100.0


def test_runtime_collects_exact_planned_telemetry():
    telemetry_queue = queue.Queue()
    observation = WorkerTelemetry(0, WorkerState(0.02, 0.01), 100.0)
    telemetry_queue.put(observation)

    assert collect_selected_telemetry(
        telemetry_queue,
        selected_client_ids={0},
        round_deadline_at=10**12,
    ) == {0: observation}


def test_runtime_rejects_telemetry_outside_cohort():
    telemetry_queue = queue.Queue()
    telemetry_queue.put(WorkerTelemetry(1, WorkerState(0.02, 0.01), 100.0))

    with pytest.raises(ValueError, match="outside planned cohort"):
        collect_selected_telemetry(
            telemetry_queue,
            selected_client_ids={0},
            round_deadline_at=10**12,
        )
