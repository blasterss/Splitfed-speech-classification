import math

import pytest

from src.splitfed.load_controller import (
    WorkerProfile,
    WorkerState,
    bandwidth_usage,
    estimate_worker_state,
    initial_batch_sizes,
    kl_divergence,
    merged_label_distribution,
    reference_label_distribution,
    selection_priorities,
)


def test_algorithm1_math_matches_hand_calculated_reference():
    previous = WorkerState(0.75, 0.25)
    observed = WorkerState(1.75, 1.25)
    estimate = estimate_worker_state(observed, previous, alpha=0.8)
    assert estimate.compute_seconds_per_sample == pytest.approx(0.95)
    assert estimate.transfer_seconds_per_sample == pytest.approx(0.45)

    states = {"fast": WorkerState(0.5, 0.5), "slow": WorkerState(1.0, 1.0)}
    batches = initial_batch_sizes(states, max_batch_size=8)
    assert batches == {"fast": 8, "slow": 4}
    assert bandwidth_usage(tuple(states), batches, 10) == 120

    profiles = [
        WorkerProfile("fast", (0.8, 0.2), participation_count=0),
        WorkerProfile("slow", (0.2, 0.8), participation_count=1),
    ]
    assert selection_priorities(profiles) == {"fast": 3.0, "slow": 1.5}
    reference = reference_label_distribution(profiles)
    merged = merged_label_distribution(profiles, batches)
    assert reference == pytest.approx((0.5, 0.5))
    assert merged == pytest.approx((0.6, 0.4))
    assert kl_divergence(merged, reference) == pytest.approx(
        0.6 * math.log(1.2) + 0.4 * math.log(0.8)
    )


def test_algorithm1_math_is_independent_of_mapping_insertion_order():
    first = {"slow": WorkerState(1.0, 1.0), "fast": WorkerState(0.5, 0.5)}
    second = dict(reversed(tuple(first.items())))

    assert initial_batch_sizes(first, 8) == initial_batch_sizes(second, 8)


@pytest.mark.parametrize(
    "distribution", [(), (0.2, 0.2), (-0.1, 1.1), (float("nan"), 1.0)]
)
def test_worker_profile_rejects_invalid_label_distribution(distribution):
    with pytest.raises(ValueError, match="label"):
        WorkerProfile("client", distribution)
