from src.schema import MergeSFLPolicyConfig
from src.splitfed.load_controller import (
    WorkerProfile,
    WorkerState,
    select_cohort_binary_ga,
)


def _config():
    return MergeSFLPolicyConfig(
        max_batch_size=8,
        local_steps=42,
        ingress_budget_bytes=120,
        feature_bytes_per_sample=10,
        kl_threshold=0.1,
        min_clients=2,
        max_clients=3,
        ga_seed=9,
        population_size=8,
        generations=5,
        mutation_probability=0.1,
    )


def _inputs():
    profiles = [
        WorkerProfile(0, (0.9, 0.1), 0),
        WorkerProfile(1, (0.1, 0.9), 2),
        WorkerProfile(2, (0.5, 0.5), 1),
    ]
    states = {
        0: WorkerState(0.5, 0.5),
        1: WorkerState(1.0, 1.0),
        2: WorkerState(1.5, 1.5),
    }
    return profiles, states


def test_binary_ga_is_replayable_and_returns_feasible_cohort():
    profiles, states = _inputs()

    first = select_cohort_binary_ga(
        profiles, states, _config(), experiment_seed=42, round_idx=3
    )
    second = select_cohort_binary_ga(
        profiles, states, _config(), experiment_seed=42, round_idx=3
    )

    assert first == second
    assert 2 <= len(first.cohort) <= 3
    assert first.bandwidth_used <= 120
    assert set(first.batch_sizes) == set(first.cohort)
    assert len(first.trace["best_scores"]) == 5


def test_binary_ga_is_independent_of_input_order():
    profiles, states = _inputs()
    reordered_profiles = list(reversed(profiles))
    reordered_states = dict(reversed(tuple(states.items())))

    first = select_cohort_binary_ga(
        profiles, states, _config(), experiment_seed=42, round_idx=3
    )
    second = select_cohort_binary_ga(
        reordered_profiles,
        reordered_states,
        _config(),
        experiment_seed=42,
        round_idx=3,
    )

    assert first == second
