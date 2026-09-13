import queue
import time
from types import SimpleNamespace

import pytest
import torch.multiprocessing as mp

from src.schema import TrainingMode, WorkloadPolicy
from src.splitfed.client import _client_worker
from src.splitfed.client.worker import _round_workload_counts
from src.splitfed.common.lifecycle import _cancel_training
from src.splitfed.load_controller import RoundPlan, WorkerState


class FakeStopEvent:
    def __init__(self):
        self.set_called = False

    def set(self):
        self.set_called = True

    def is_set(self):
        return self.set_called


class FakeBarrier:
    def __init__(self):
        self.abort_called = False
        self.wait_timeouts = []

    def wait(self, timeout):
        self.wait_timeouts.append(timeout)

    def abort(self):
        self.abort_called = True


class NoOpClient:
    def __init__(self, **kwargs):
        self.client_id = kwargs["cfg"].client_id

    def evaluate(self, round):
        return {}


class RecordingClient(NoOpClient):
    trained_rounds = []
    evaluated_rounds = []
    events = []

    def train_one_round(self, round):
        self.trained_rounds.append(round)
        self.events.append(("train", round))

    def federative_aggregate(self, round):
        self.events.append(("aggregate", round))

    def evaluate(self, round):
        self.evaluated_rounds.append(round)
        self.events.append(("evaluate", round))
        return {}


class PlannedClient(RecordingClient):
    configured = []

    def configure_round(self, **kwargs):
        self.configured.append(kwargs)

    def skip_round(self, round_idx):
        self.events.append(("skip", round_idx))

    def federative_aggregate(self, round, aggregation_weight=None):
        self.events.append(("aggregate", round, aggregation_weight))


def test_fixed_steps_resource_counts_include_reused_samples():
    assert _round_workload_counts(
        loader_batches=45,
        dataset_samples=360,
        batch_size=8,
        local_steps=100,
        policy=WorkloadPolicy.fixed_steps_v1,
    ) == (100, 800)


def _barrier_waiter(barrier, ready_queue, result_queue):
    ready_queue.put("ready")
    try:
        barrier.wait(timeout=5)
    except Exception as exc:
        result_queue.put(type(exc).__name__)
    else:
        result_queue.put("passed")


def _worker_args(stop_event, ready_barrier, eval_barrier):
    cfg = SimpleNamespace(
        client_id=0,
        runtime=SimpleNamespace(seed=42),
    )
    training_cfg = SimpleNamespace(
        num_rounds=0,
        fed_every=1,
        seed=17,
        mode=TrainingMode.splitfed,
        barrier_timeout_sec=0.25,
    )
    channels = (object(), object(), object(), object())
    return (
        cfg,
        training_cfg,
        *channels,
        stop_event,
        ready_barrier,
        eval_barrier,
    )


def test_client_initialization_failure_cancels_peer_waits(monkeypatch):
    stop_event = FakeStopEvent()
    ready_barrier = FakeBarrier()
    eval_barrier = FakeBarrier()

    def fail_client(**kwargs):
        raise RuntimeError("dataset load failed")

    monkeypatch.setattr("src.splitfed.client.Client", fail_client)

    with pytest.raises(RuntimeError, match="dataset load failed"):
        _client_worker(*_worker_args(stop_event, ready_barrier, eval_barrier))

    assert stop_event.set_called
    assert ready_barrier.abort_called
    assert eval_barrier.abort_called


def test_client_initialization_failure_reports_structured_context(monkeypatch):
    failure_queue = queue.Queue(maxsize=1)

    def fail_client(**kwargs):
        raise RuntimeError("dataset load failed")

    monkeypatch.setattr("src.splitfed.client.Client", fail_client)

    with pytest.raises(RuntimeError, match="dataset load failed"):
        _client_worker(
            *_worker_args(FakeStopEvent(), FakeBarrier(), FakeBarrier()),
            failure_queue=failure_queue,
        )

    failure = failure_queue.get_nowait()
    assert failure["schema_version"] == 1
    assert failure["component"] == "client"
    assert failure["client_id"] == 0
    assert failure["round"] is None
    assert failure["step"] is None
    assert failure["exception_type"] == "RuntimeError"
    assert failure["message"] == "dataset load failed"
    assert "RuntimeError: dataset load failed" in failure["traceback"]


def test_client_barrier_waits_use_configured_timeout(monkeypatch):
    stop_event = FakeStopEvent()
    ready_barrier = FakeBarrier()
    eval_barrier = FakeBarrier()
    monkeypatch.setattr("src.splitfed.client.Client", NoOpClient)

    _client_worker(*_worker_args(stop_event, ready_barrier, eval_barrier))

    assert ready_barrier.wait_timeouts == [0.25]
    assert eval_barrier.wait_timeouts == [0.25, 0.25]
    assert not stop_event.set_called


def test_client_evaluates_on_cadence_and_final_round(monkeypatch):
    RecordingClient.trained_rounds = []
    RecordingClient.evaluated_rounds = []
    RecordingClient.events = []
    stop_event = FakeStopEvent()
    ready_barrier = FakeBarrier()
    eval_barrier = FakeBarrier()
    monkeypatch.setattr("src.splitfed.client.Client", RecordingClient)
    cfg = SimpleNamespace(client_id=0, runtime=SimpleNamespace(seed=42))
    training_cfg = SimpleNamespace(
        num_rounds=3,
        eval_every=2,
        fed_every=1,
        seed=17,
        mode=TrainingMode.split,
        barrier_timeout_sec=0.25,
    )

    _client_worker(
        cfg,
        training_cfg,
        object(),
        object(),
        None,
        None,
        stop_event,
        ready_barrier,
        eval_barrier,
    )

    assert RecordingClient.trained_rounds == [1, 2, 3]
    assert RecordingClient.evaluated_rounds == [2, 3]
    assert eval_barrier.wait_timeouts == [0.25] * 4


@pytest.mark.parametrize(
    "mode,expected_events",
    [
        (
            TrainingMode.splitfed,
            [("train", 1), ("aggregate", 1), ("evaluate", 1)],
        ),
        (
            TrainingMode.federated,
            [("train", 1), ("aggregate", 1), ("evaluate", 1)],
        ),
    ],
)
def test_evaluation_uses_mode_compatible_model_pair(
    monkeypatch, mode, expected_events
):
    RecordingClient.events = []
    stop_event = FakeStopEvent()
    monkeypatch.setattr("src.splitfed.client.Client", RecordingClient)
    cfg = SimpleNamespace(client_id=0, runtime=SimpleNamespace(seed=42))
    training_cfg = SimpleNamespace(
        num_rounds=1,
        eval_every=1,
        fed_every=1,
        seed=17,
        mode=mode,
        barrier_timeout_sec=0.25,
    )

    _client_worker(
        cfg,
        training_cfg,
        object(),
        object(),
        object(),
        object(),
        stop_event,
        FakeBarrier(),
        FakeBarrier(),
    )

    assert RecordingClient.events == expected_events


def test_final_federated_aggregation_can_be_disabled(monkeypatch):
    RecordingClient.events = []
    stop_event = FakeStopEvent()
    monkeypatch.setattr("src.splitfed.client.Client", RecordingClient)
    cfg = SimpleNamespace(client_id=0, runtime=SimpleNamespace(seed=42))
    training_cfg = SimpleNamespace(
        num_rounds=2,
        eval_every=2,
        fed_every=1,
        seed=17,
        aggregate_final=False,
        mode=TrainingMode.federated,
        barrier_timeout_sec=0.25,
    )

    _client_worker(
        cfg,
        training_cfg,
        object(),
        object(),
        object(),
        object(),
        stop_event,
        FakeBarrier(),
        FakeBarrier(),
    )

    assert RecordingClient.events == [
        ("train", 1),
        ("aggregate", 1),
        ("train", 2),
        ("evaluate", 2),
    ]


def _round_plan(*, cohort=(0,), batch_size=4):
    return RoundPlan(
        round=1,
        seed=17,
        cohort=cohort,
        batch_size_by_client={client_id: batch_size for client_id in cohort},
        local_steps=3,
        required_quorum=len(cohort),
        deadline_at=time.time() + 10,
        model_version="initial",
        estimates={
            client_id: WorkerState(0.01, 0.001) for client_id in cohort
        },
        bandwidth_used=batch_size * len(cohort),
        reference_distribution=(0.5, 0.5),
        merged_distribution=(0.5, 0.5),
        kl_divergence=0.0,
        decision_trace={},
    )


def test_client_executes_controller_round_plan(monkeypatch):
    PlannedClient.events = []
    PlannedClient.configured = []
    monkeypatch.setattr("src.splitfed.client.Client", PlannedClient)
    plan_queue = queue.Queue()
    plan_queue.put(_round_plan())
    cfg = SimpleNamespace(client_id=0, runtime=SimpleNamespace(seed=42))
    training_cfg = SimpleNamespace(
        num_rounds=1,
        eval_every=1,
        fed_every=1,
        seed=17,
        mode=TrainingMode.splitfed,
        barrier_timeout_sec=0.25,
    )

    _client_worker(
        cfg,
        training_cfg,
        object(),
        object(),
        object(),
        object(),
        FakeStopEvent(),
        FakeBarrier(),
        FakeBarrier(),
        round_plan_queue=plan_queue,
    )

    assert PlannedClient.configured == [
        {"round_idx": 1, "batch_size": 4, "local_steps": 3}
    ]
    assert PlannedClient.events[:2] == [
        ("train", 1),
        ("aggregate", 1, 12),
    ]


def test_client_skips_round_outside_controller_cohort(monkeypatch):
    PlannedClient.events = []
    PlannedClient.configured = []
    monkeypatch.setattr("src.splitfed.client.Client", PlannedClient)
    plan_queue = queue.Queue()
    plan_queue.put(_round_plan(cohort=(1,)))
    cfg = SimpleNamespace(client_id=0, runtime=SimpleNamespace(seed=42))
    training_cfg = SimpleNamespace(
        num_rounds=1,
        eval_every=1,
        fed_every=1,
        seed=17,
        mode=TrainingMode.splitfed,
        barrier_timeout_sec=0.25,
    )

    _client_worker(
        cfg,
        training_cfg,
        object(),
        object(),
        object(),
        object(),
        FakeStopEvent(),
        FakeBarrier(),
        FakeBarrier(),
        round_plan_queue=plan_queue,
    )

    assert PlannedClient.configured == []
    assert PlannedClient.events == [("skip", 1), ("evaluate", 1)]


@pytest.mark.parametrize(
    ("mode", "expected_seed"),
    [
        (TrainingMode.federated, 17),
        (TrainingMode.splitfed, 17),
        (TrainingMode.split, 45),
    ],
)
def test_aggregated_clients_share_training_initialization_seed(
    monkeypatch, mode, expected_seed
):
    observed = []
    monkeypatch.setattr("src.splitfed.client.worker.set_seed", observed.append)
    monkeypatch.setattr("src.splitfed.client.Client", NoOpClient)
    cfg = SimpleNamespace(client_id=3, runtime=SimpleNamespace(seed=42))
    training_cfg = SimpleNamespace(
        num_rounds=0,
        fed_every=1,
        seed=17,
        mode=mode,
        barrier_timeout_sec=0.25,
    )

    _client_worker(
        cfg,
        training_cfg,
        object(),
        object(),
        object(),
        object(),
        FakeStopEvent(),
        FakeBarrier(),
        FakeBarrier(),
    )

    assert observed == [expected_seed]


def test_spawned_barrier_waiter_exits_after_cancellation():
    context = mp.get_context("spawn")
    barrier = context.Barrier(2)
    stop_event = context.Event()
    ready_queue = context.Queue()
    result_queue = context.Queue()
    process = context.Process(
        target=_barrier_waiter,
        args=(barrier, ready_queue, result_queue),
    )

    process.start()
    assert ready_queue.get(timeout=5) == "ready"

    _cancel_training(stop_event, (barrier,))
    process.join(timeout=5)

    assert stop_event.is_set()
    assert not process.is_alive()
    assert process.exitcode == 0
    assert result_queue.get(timeout=1) == "BrokenBarrierError"
