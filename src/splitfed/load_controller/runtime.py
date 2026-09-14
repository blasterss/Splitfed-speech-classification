"""Runtime adapters between controller reports and MergeSFL planning types."""

import queue
import time
from collections.abc import Mapping

from ...schema import MergeSFLPolicyConfig
from .mergesfl import WorkerProfile, WorkerState, WorkerTelemetry


def profiles_from_manifests(
    manifests: Mapping[int, dict],
    participation_counts: Mapping[int, int],
) -> list[WorkerProfile]:
    """Build binary-label worker profiles from validated dataset reports."""
    if set(manifests) != set(participation_counts):
        raise ValueError("manifests and participation counts must match")
    profiles = []
    for client_id in sorted(manifests):
        coverage = manifests[client_id].get("coverage", {}).get("train", {})
        class_0 = coverage.get("class_0")
        class_1 = coverage.get("class_1")
        if not all(
            isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0
            for value in (class_0, class_1)
        ):
            raise ValueError(f"invalid train coverage for client {client_id}")
        total = class_0 + class_1
        if total <= 0:
            raise ValueError(f"empty train coverage for client {client_id}")
        profiles.append(
            WorkerProfile(
                client_id=client_id,
                label_distribution=(class_0 / total, class_1 / total),
                participation_count=participation_counts[client_id],
                train_samples=total,
            )
        )
    return profiles


def bootstrap_telemetry(
    config: MergeSFLPolicyConfig, *, observed_at: float | None = None
) -> dict[int, WorkerTelemetry]:
    """Convert configured initial timings into first-round observations."""
    timestamp = time.time() if observed_at is None else observed_at
    return {
        client_id: WorkerTelemetry(
            client_id=client_id,
            state=WorkerState(
                compute_seconds_per_sample=timing.compute_seconds_per_sample,
                transfer_seconds_per_sample=timing.transfer_seconds_per_sample,
            ),
            observed_at=timestamp,
        )
        for client_id, timing in config.initial_worker_states.items()
    }


def collect_selected_telemetry(
    telemetry_queue,
    *,
    selected_client_ids: set[int],
    round_deadline_at: float,
) -> dict[int, WorkerTelemetry]:
    """Collect exactly one fresh observation from each selected client."""
    observations = {}
    while set(observations) != selected_client_ids:
        remaining = round_deadline_at - time.time()
        if remaining <= 0:
            missing = sorted(selected_client_ids - set(observations))
            raise TimeoutError(f"missing MergeSFL telemetry from {missing}")
        try:
            observation = telemetry_queue.get(timeout=remaining)
        except queue.Empty as exc:
            missing = sorted(selected_client_ids - set(observations))
            raise TimeoutError(
                f"missing MergeSFL telemetry from {missing}"
            ) from exc
        if not isinstance(observation, WorkerTelemetry):
            raise ValueError("invalid MergeSFL telemetry payload")
        if observation.client_id not in selected_client_ids:
            raise ValueError("telemetry sender is outside planned cohort")
        if observation.client_id in observations:
            raise ValueError("duplicate MergeSFL telemetry")
        observations[observation.client_id] = observation
    return observations
