import queue
import time

import pytest
import torch
from torch import nn

from src.schema import (
    FedServerConfig,
    SplitServerConfig,
    TrainingConfig,
    TrainingMode,
)
from src.splitfed.load_controller import RoundPlan, WorkerState
from src.splitfed.split_server import (
    SplitServer,
    _batch_is_ready,
    _build_personalized_models,
    _evict_stale_batches,
    _forward_concat,
    _forward_mergesfl,
    _split_server_worker_concat,
    _split_server_worker_personalized,
    _store_pending_batch,
    _validate_message,
    validate_message_against_plan,
)
from src.transport.message import Message, MessageType

FUTURE_DEADLINE = time.time() + 3600


def _train_message(sender, step):
    return Message(
        type=MessageType.TRAIN_STEP,
        sender=sender,
        round=1,
        step=step,
        payload={},
        deadline_at=FUTURE_DEADLINE,
    )


def test_round_end_is_a_typed_protocol_message():
    message = Message(
        type="round_end",
        sender="client-0",
        round=1,
        step=2,
        payload={},
        deadline_at=FUTURE_DEADLINE,
    )

    assert message.type is MessageType.ROUND_END
    assert _validate_message(message, "client-0")


def _split_message(**overrides):
    values = {
        "type": "train_step",
        "sender": "client-0",
        "round": 1,
        "step": 1,
        "deadline_at": FUTURE_DEADLINE,
        "payload": {
            "activations": torch.ones(2, 3),
            "labels": torch.ones(2),
        },
    }
    values.update(overrides)
    return Message(**values)


def test_mergesfl_rescales_dispatched_gradients_by_merged_batch_size():
    model = nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(0.5)
    criterion = nn.BCEWithLogitsLoss()
    messages = {
        "client-0": _split_message(
            sender="client-0",
            payload={
                "activations": torch.tensor([[1.0]]),
                "labels": torch.tensor([1.0]),
            },
        ),
        "client-1": _split_message(
            sender="client-1",
            payload={
                "activations": torch.tensor([[1.0], [2.0], [3.0]]),
                "labels": torch.tensor([0.0, 1.0, 0.0]),
            },
        ),
    }

    concat_grads, _ = _forward_concat(
        messages, model, criterion, torch.device("cpu")
    )
    mergesfl_grads, _ = _forward_mergesfl(
        messages, model, criterion, torch.device("cpu")
    )

    assert torch.allclose(
        mergesfl_grads["client-0"], concat_grads["client-0"] * 4
    )
    assert torch.allclose(
        mergesfl_grads["client-1"], concat_grads["client-1"] * (4 / 3)
    )


def test_split_message_validates_channel_sender_and_correlation():
    assert _validate_message(_split_message(), "client-0")
    assert not _validate_message(_split_message(sender="client-1"), "client-0")
    assert not _validate_message(_split_message(round=0), "client-0")
    assert not _validate_message(_split_message(step=0), "client-0")
    assert not _validate_message(
        _split_message(deadline_at=time.time() - 1), "client-0"
    )


def test_split_message_requires_labels_with_matching_batch_size():
    assert not _validate_message(
        _split_message(payload={"activations": torch.ones(2, 3)}),
        "client-0",
    )
    assert not _validate_message(
        _split_message(
            payload={
                "activations": torch.ones(2, 3),
                "labels": torch.ones(3),
            }
        ),
        "client-0",
    )


def _controller_plan():
    return RoundPlan(
        round=1,
        seed=42,
        cohort=("client-0",),
        batch_size_by_client={"client-0": 2},
        local_steps=3,
        required_quorum=1,
        deadline_at=FUTURE_DEADLINE,
        model_version="initial",
        estimates={"client-0": WorkerState(0.01, 0.01)},
        bandwidth_used=2,
        reference_distribution=(0.5, 0.5),
        merged_distribution=(0.5, 0.5),
        kl_divergence=0.0,
        decision_trace={},
    )


def test_split_message_must_match_planned_batch_and_cohort():
    plan = _controller_plan()
    validate_message_against_plan(_split_message(), "client-0", plan)

    with pytest.raises(ValueError, match="batch size"):
        validate_message_against_plan(
            _split_message(
                payload={
                    "activations": torch.ones(1, 3),
                    "labels": torch.ones(1),
                }
            ),
            "client-0",
            plan,
        )
    with pytest.raises(ValueError, match="outside planned cohort"):
        validate_message_against_plan(
            _split_message(sender="client-1"), "client-1", plan
        )


def test_split_round_end_distinguishes_selected_and_skipped_clients():
    plan = _controller_plan()
    validate_message_against_plan(
        Message(
            type="round_end",
            sender="client-0",
            round=1,
            step=4,
            deadline_at=FUTURE_DEADLINE,
        ),
        "client-0",
        plan,
    )
    validate_message_against_plan(
        Message(
            type="round_end",
            sender="client-1",
            round=1,
            step=1,
            deadline_at=FUTURE_DEADLINE,
        ),
        "client-1",
        plan,
    )


def test_split_message_rejects_expired_round_plan():
    plan = _controller_plan()
    expired = RoundPlan(
        **{
            **plan.__dict__,
            "deadline_at": time.time() - 1,
        }
    )

    with pytest.raises(TimeoutError, match="deadline"):
        validate_message_against_plan(_split_message(), "client-0", expired)


def test_duplicate_split_step_is_rejected_as_replay():
    pending = {}
    message = _split_message()

    _store_pending_batch(pending, message, "client-0")

    with pytest.raises(ValueError, match="Duplicate split step"):
        _store_pending_batch(pending, message, "client-0")


class WorkerStopEvent:
    def __init__(self):
        self.stopped = False

    def is_set(self):
        return self.stopped

    def set(self):
        self.stopped = True

    def wait(self, timeout):
        return self.stopped


class OneMessageUplink:
    def __init__(self, message):
        self.message = message

    def recv_nowait(self):
        message, self.message = self.message, None
        return message


class DiscardResultQueue:
    def put_nowait(self, value):
        pass


def test_split_worker_reports_original_failure_context():
    stop_event = WorkerStopEvent()
    failure_queue = queue.Queue(maxsize=1)
    config = SplitServerConfig(
        seed=42,
        model_scope="shared",
        split_uplink_channel="split_uplink",
        split_downlink_channel="split_downlink",
        model={
            "pos_weight": 1,
            "optimizer": "adam",
            "lr": 0.001,
            "device": "cpu",
            "gradient_accumulation_steps": 1,
            "batch_timeout_sec": 1,
        },
    )
    invalid = _split_message(sender="wrong-client")
    channels = {
        "client-0": {
            "uplink": OneMessageUplink(invalid),
            "downlink": RecordingDownlink(),
        }
    }

    with pytest.raises(ValueError, match="Invalid split message"):
        _split_server_worker_concat(
            config,
            channels,
            stop_event,
            DiscardResultQueue(),
            failure_queue,
        )

    failure = failure_queue.get_nowait()
    assert failure["component"] == "split_server"
    assert failure["client_id"] == "client-0"
    assert failure["round"] == 1
    assert failure["step"] == 1
    assert failure["exception_type"] == "ValueError"


class RecordingDownlink:
    def __init__(self):
        self.messages = []

    def send(self, message):
        self.messages.append(message)


class ScriptedUplink:
    def __init__(self, messages):
        self.messages = list(messages)

    def recv_nowait(self):
        return self.messages.pop(0) if self.messages else None


class StoppingDownlink(RecordingDownlink):
    all_messages = []
    stop_event = None

    def send(self, message):
        super().send(message)
        self.all_messages.append(message)
        if len(self.all_messages) == 4:
            self.stop_event.set()


def test_personalized_splitfed_aggregates_on_cadence_and_acks_rounds(
    monkeypatch,
):
    stop_event = WorkerStopEvent()
    StoppingDownlink.all_messages = []
    StoppingDownlink.stop_event = stop_event
    client_ids = ("client-0", "client-1")
    models = {
        client_id: nn.Linear(1, 1, bias=False) for client_id in client_ids
    }
    models["client-0"].weight.data.zero_()
    models["client-1"].weight.data.fill_(2.0)
    optimizers = {
        client_id: torch.optim.SGD(model.parameters(), lr=0.1)
        for client_id, model in models.items()
    }
    monkeypatch.setattr(
        "src.splitfed.split_server.worker.personalized.build_personalized_models",
        lambda *_: (models, optimizers),
    )
    channels = {}
    for client_id, size in zip(client_ids, (1, 3), strict=True):
        messages = [
            Message(
                type="round_end",
                sender=client_id,
                round=round_idx,
                step=2,
                deadline_at=FUTURE_DEADLINE,
                payload={"dataset_size": size},
            )
            for round_idx in (1, 2)
        ]
        channels[client_id] = {
            "uplink": ScriptedUplink(messages),
            "downlink": StoppingDownlink(),
        }
    split_config = SplitServerConfig(
        model={"lr": 0.001, "device": "cpu"},
        model_scope="personalized",
        seed=42,
        split_uplink_channel="split_uplink",
        split_downlink_channel="split_downlink",
    )
    training_config = TrainingConfig(
        mode="splitfed",
        num_rounds=2,
        seed=42,
        eval_every=2,
        fed_every=2,
        aggregate_final=True,
    )
    fed_config = FedServerConfig(
        strategy="weighted_fedavg",
        seed=42,
        device="cpu",
        aggregation_freq=2,
        min_clients=2,
        federated_uplink_channel="federated_uplink",
        federated_downlink_channel="federated_downlink",
    )

    _split_server_worker_personalized(
        split_config,
        channels,
        stop_event,
        DiscardResultQueue(),
        training_mode=TrainingMode.splitfed,
        training_config=training_config,
        fed_server_config=fed_config,
    )

    assert [
        message.payload["server_aggregated"]
        for message in StoppingDownlink.all_messages
    ] == [False, False, True, True]
    assert torch.equal(models["client-0"].weight, torch.tensor([[1.5]]))
    assert torch.equal(models["client-1"].weight, torch.tensor([[1.5]]))


def test_stale_batch_sends_correlated_error_to_waiting_clients(monkeypatch):
    message = _split_message(round=2, step=3)
    pending = {(2, 3): {"client-0": message}}
    timestamps = {(2, 3): 0.0}
    downlink = RecordingDownlink()
    channels = {"client-0": {"downlink": downlink}}
    monkeypatch.setattr(
        "src.splitfed.split_server.protocol.time.monotonic", lambda: 10.0
    )

    _evict_stale_batches(
        pending,
        timestamps,
        channels,
        timeout_seconds=5.0,
    )

    assert pending == {}
    assert timestamps == {}
    assert len(downlink.messages) == 1
    error = downlink.messages[0]
    assert error.type is MessageType.ERROR
    assert (error.round, error.step) == (2, 3)
    assert error.request_id == message.request_id


def test_batch_becomes_ready_when_missing_client_finished_round():
    batch = {"client-1": _train_message("client-1", step=2)}

    assert not _batch_is_ready(batch, {"client-0", "client-1"}, set())
    assert _batch_is_ready(
        batch,
        {"client-0", "client-1"},
        {"client-0"},
    )


def test_personalized_server_models_and_optimizers_are_isolated():
    config = SplitServerConfig(
        model={
            "pos_weight": 1,
            "optimizer": "adam",
            "lr": 0.001,
            "device": "cpu",
            "gradient_accumulation_steps": 1,
            "batch_timeout_sec": 5,
        },
        model_scope="personalized",
        seed=42,
        split_uplink_channel="split_uplink",
        split_downlink_channel="split_downlink",
    )
    models, optimizers = _build_personalized_models(
        ["client-0", "client-1"], config, torch.device("cpu")
    )
    other_before = {
        key: value.detach().clone()
        for key, value in models["client-1"].state_dict().items()
    }

    loss = sum(
        parameter.sum() for parameter in models["client-0"].parameters()
    )
    loss.backward()
    optimizers["client-0"].step()

    assert models["client-0"] is not models["client-1"]
    assert optimizers["client-0"] is not optimizers["client-1"]
    assert all(
        torch.equal(other_before[key], value)
        for key, value in models["client-1"].state_dict().items()
    )


def test_personalized_server_saves_one_checkpoint_per_client(tmp_path):
    server = SplitServer.__new__(SplitServer)
    server.config = SplitServerConfig(
        model={"lr": 0.001, "device": "cpu"},
        model_scope="personalized",
        seed=42,
        split_uplink_channel="split_uplink",
        split_downlink_channel="split_downlink",
    )
    server.get_state_dict = lambda: {
        "client-0": {"weight": torch.tensor([0.0])},
        "client-1": {"weight": torch.tensor([1.0])},
    }

    server.save(tmp_path)

    first = torch.load(
        tmp_path / "split_server_client_client-0.pt", weights_only=True
    )
    second = torch.load(
        tmp_path / "split_server_client_client-1.pt", weights_only=True
    )
    assert first["mode"] == "split"
    assert first["server_model_scope"] == "personalized"
    assert first["client_id"] == "client-0"
    assert second["client_id"] == "client-1"
    assert torch.equal(
        first["model_state_dict"]["weight"], torch.tensor([0.0])
    )
    assert torch.equal(
        second["model_state_dict"]["weight"], torch.tensor([1.0])
    )


def test_concat_client_gradient_uses_full_batch_loss():
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1.0)
    criterion = torch.nn.BCEWithLogitsLoss()
    activations = torch.tensor([[0.5]])
    labels = torch.tensor([[1.0]])
    message = Message(
        type="train_step",
        sender="client-0",
        round=1,
        step=1,
        payload={"activations": activations, "labels": labels},
    )

    gradients, _ = _forward_concat(
        {"client-0": message}, model, criterion, torch.device("cpu")
    )

    expected = torch.sigmoid(activations) - labels
    assert torch.allclose(gradients["client-0"], expected)
