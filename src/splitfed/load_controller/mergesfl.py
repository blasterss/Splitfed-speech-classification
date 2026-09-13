"""Deterministic mathematical primitives for MergeSFL Algorithm 1."""

import math
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass

ClientId = Hashable


@dataclass(frozen=True)
class WorkerProfile:
    client_id: ClientId
    label_distribution: tuple[float, ...]
    participation_count: int = 0

    def __post_init__(self) -> None:
        _validate_distribution(self.label_distribution)
        if self.participation_count < 0:
            raise ValueError("participation_count must be non-negative")


@dataclass(frozen=True)
class WorkerState:
    compute_seconds_per_sample: float
    transfer_seconds_per_sample: float

    def __post_init__(self) -> None:
        if not _is_positive_finite(self.compute_seconds_per_sample):
            raise ValueError("compute_seconds_per_sample must be positive")
        if not _is_positive_finite(self.transfer_seconds_per_sample):
            raise ValueError("transfer_seconds_per_sample must be positive")

    @property
    def cost(self) -> float:
        return (
            self.compute_seconds_per_sample
            + self.transfer_seconds_per_sample
        )


def estimate_worker_state(
    observation: WorkerState,
    previous: WorkerState | None,
    *,
    alpha: float,
) -> WorkerState:
    """Apply the moving averages from Equations 5 and 6."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be within [0, 1]")
    if previous is None:
        return observation
    return WorkerState(
        compute_seconds_per_sample=(
            alpha * previous.compute_seconds_per_sample
            + (1 - alpha) * observation.compute_seconds_per_sample
        ),
        transfer_seconds_per_sample=(
            alpha * previous.transfer_seconds_per_sample
            + (1 - alpha) * observation.transfer_seconds_per_sample
        ),
    )


def initial_batch_sizes(
    states: Mapping[ClientId, WorkerState], max_batch_size: int
) -> dict[ClientId, int]:
    """Calculate the initial regulated batch sizes from Equation 9."""
    if not states:
        raise ValueError("at least one worker state is required")
    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be positive")
    fastest_id = min(states, key=lambda key: (states[key].cost, str(key)))
    fastest_cost = states[fastest_id].cost
    return {
        client_id: max(
            1, math.floor(max_batch_size * fastest_cost / state.cost)
        )
        for client_id, state in states.items()
    }


def bandwidth_usage(
    cohort: Sequence[ClientId],
    batch_sizes: Mapping[ClientId, int],
    feature_bytes_per_sample: int,
) -> int:
    """Return ingress bandwidth occupied per iteration (Equation 10)."""
    if feature_bytes_per_sample <= 0:
        raise ValueError("feature_bytes_per_sample must be positive")
    if not cohort:
        raise ValueError("cohort must contain at least one client")
    try:
        sizes = [batch_sizes[client_id] for client_id in cohort]
    except KeyError as exc:
        raise ValueError(
            f"missing batch size for client {exc.args[0]!r}"
        ) from exc
    if any(size <= 0 for size in sizes):
        raise ValueError("batch sizes must be positive")
    return sum(sizes) * feature_bytes_per_sample


def selection_priorities(
    profiles: Sequence[WorkerProfile],
) -> dict[ClientId, float]:
    """Calculate participation priorities from Equation 13."""
    if not profiles:
        raise ValueError("at least one worker profile is required")
    total = sum(profile.participation_count + 1 for profile in profiles)
    return {
        profile.client_id: total / (profile.participation_count + 1)
        for profile in profiles
    }


def reference_label_distribution(
    profiles: Sequence[WorkerProfile],
) -> tuple[float, ...]:
    """Return the unweighted eligible-worker mean distribution."""
    _validate_profiles(profiles)
    classes = len(profiles[0].label_distribution)
    return tuple(
        sum(profile.label_distribution[index] for profile in profiles)
        / len(profiles)
        for index in range(classes)
    )


def merged_label_distribution(
    profiles: Sequence[WorkerProfile],
    batch_sizes: Mapping[ClientId, int],
) -> tuple[float, ...]:
    """Return the batch-weighted merged distribution from Equation 11."""
    _validate_profiles(profiles)
    try:
        weights = [batch_sizes[profile.client_id] for profile in profiles]
    except KeyError as exc:
        raise ValueError(
            f"missing batch size for client {exc.args[0]!r}"
        ) from exc
    if any(weight <= 0 for weight in weights):
        raise ValueError("batch sizes must be positive")
    total = sum(weights)
    classes = len(profiles[0].label_distribution)
    return tuple(
        sum(
            weight * profile.label_distribution[index]
            for profile, weight in zip(profiles, weights, strict=True)
        )
        / total
        for index in range(classes)
    )


def kl_divergence(
    distribution: Sequence[float],
    reference: Sequence[float],
    *,
    epsilon: float = 1e-12,
) -> float:
    """Calculate KL(distribution || reference) from Equation 12."""
    _validate_distribution(distribution)
    _validate_distribution(reference)
    if len(distribution) != len(reference):
        raise ValueError("distributions must use the same class order")
    if not _is_positive_finite(epsilon):
        raise ValueError("epsilon must be positive")
    return sum(
        value * math.log(value / max(reference_value, epsilon))
        for value, reference_value in zip(
            distribution, reference, strict=True
        )
        if value > 0
    )


def _validate_profiles(profiles: Sequence[WorkerProfile]) -> None:
    if not profiles:
        raise ValueError("at least one worker profile is required")
    classes = len(profiles[0].label_distribution)
    if any(len(profile.label_distribution) != classes for profile in profiles):
        raise ValueError("worker distributions must use the same class order")
    client_ids = [profile.client_id for profile in profiles]
    if len(client_ids) != len(set(client_ids)):
        raise ValueError("worker client IDs must be unique")


def _validate_distribution(distribution: Sequence[float]) -> None:
    if not distribution:
        raise ValueError("label distribution must not be empty")
    if any(not math.isfinite(value) or value < 0 for value in distribution):
        raise ValueError("label probabilities must be finite and non-negative")
    if not math.isclose(sum(distribution), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("label probabilities must sum to one")


def _is_positive_finite(value: float) -> bool:
    return math.isfinite(value) and value > 0
