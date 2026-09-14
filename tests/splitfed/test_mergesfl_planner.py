import pytest

from src.schema import MergeSFLPolicyConfig
from src.splitfed.load_controller import (
    MergeSFLPlanner,
    WorkerProfile,
    WorkerState,
    WorkerTelemetry,
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


def test_planner_builds_replayable_round_plan_and_updates_ema():
    profiles, states = _inputs()
    telemetry = [
        WorkerTelemetry(client_id, state, observed_at=90.0)
        for client_id, state in states.items()
    ]
    first_planner = MergeSFLPlanner(_config(), experiment_seed=42)
    second_planner = MergeSFLPlanner(_config(), experiment_seed=42)

    first = first_planner.plan(
        profiles,
        telemetry,
        round_idx=1,
        model_version="model-0",
        now=100.0,
    )
    second = second_planner.plan(
        profiles,
        telemetry,
        round_idx=1,
        model_version="model-0",
        now=100.0,
    )

    assert first == second
    assert first.deadline_at == 400.0
    assert first.bandwidth_used <= _config().ingress_budget_bytes
    assert first.required_quorum == len(first.cohort)
    assert first.decision_trace["refinement_policy"] == (
        "integer_refinement_v1"
    )
    assert set(first.decision_trace["telemetry_inputs"]) == {"0", "1", "2"}


def test_planner_caps_batches_to_each_clients_train_samples():
    profiles = [
        WorkerProfile(0, (0.5, 0.5), train_samples=2),
        WorkerProfile(1, (0.5, 0.5), train_samples=20),
    ]
    telemetry = [
        WorkerTelemetry(0, WorkerState(0.5, 0.5), 90.0),
        WorkerTelemetry(1, WorkerState(1.0, 1.0), 90.0),
    ]

    plan = MergeSFLPlanner(_config(), 42).plan(
        profiles,
        telemetry,
        round_idx=1,
        model_version="model-0",
        now=100.0,
    )

    assert plan.batch_size_by_client[0] <= 2


def test_planner_fails_when_kl_constraint_is_infeasible():
    config = _config().model_copy(
        update={"min_clients": 1, "max_clients": 1, "kl_threshold": 0.01}
    )
    profiles = [
        WorkerProfile(0, (1.0, 0.0)),
        WorkerProfile(1, (0.0, 1.0)),
    ]
    telemetry = [
        WorkerTelemetry(0, WorkerState(1.0, 1.0), 90.0),
        WorkerTelemetry(1, WorkerState(1.0, 1.0), 90.0),
    ]

    with pytest.raises(ValueError, match="KL threshold"):
        MergeSFLPlanner(config, 42).plan(
            profiles,
            telemetry,
            round_idx=1,
            model_version="model-0",
            now=100.0,
        )


def test_planner_rejects_round_without_fresh_quorum():
    profiles, states = _inputs()
    telemetry = [
        WorkerTelemetry(client_id, state, observed_at=1.0)
        for client_id, state in states.items()
    ]
    planner = MergeSFLPlanner(_config(), experiment_seed=42)

    try:
        planner.plan(
            profiles,
            telemetry,
            round_idx=1,
            model_version="model-0",
            now=1000.0,
        )
    except ValueError as exc:
        assert "not enough eligible" in str(exc)
    else:
        raise AssertionError("stale telemetry unexpectedly formed a quorum")


def test_heartbeat_keeps_unselected_client_eligible_without_ema_change():
    profiles, states = _inputs()
    planner = MergeSFLPlanner(_config(), experiment_seed=42)
    planner.plan(
        profiles,
        [
            WorkerTelemetry(client_id, state, 90.0)
            for client_id, state in states.items()
        ],
        round_idx=1,
        model_version="model-0",
        now=100.0,
    )

    plan = planner.plan(
        profiles,
        [
            WorkerTelemetry(client_id, None, 999.0, round=1)
            for client_id in states
        ],
        round_idx=2,
        model_version="model-1",
        now=1000.0,
    )

    assert plan.decision_trace["rejected"] == {}
    assert planner.estimates == states
