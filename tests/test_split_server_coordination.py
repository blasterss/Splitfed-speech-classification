import pytest
import torch

from src.schema import SplitServerConfig
from src.splitfed.split_server import (
    SplitServer,
    _batch_is_ready,
    _build_personalized_models,
    _evict_stale_batches,
    _forward_parallel,
    _step_accumulated_gradients,
    _store_pending_batch,
    _validate_message,
)
from src.transport.message import Message, MessageType


def _train_message(sender, step):
    return Message(
        type=MessageType.TRAIN_STEP,
        sender=sender,
        round=1,
        step=step,
        payload={},
    )


def test_round_end_is_a_typed_protocol_message():
    message = Message(
        type="round_end", sender="client-0", round=1, step=2, payload={}
    )

    assert message.type is MessageType.ROUND_END
    assert _validate_message(message, "client-0")


def _split_message(**overrides):
    values = {
        "type": "train_step",
        "sender": "client-0",
        "round": 1,
        "step": 1,
        "payload": {
            "activations": torch.ones(2, 3),
            "labels": torch.ones(2),
        },
    }
    values.update(overrides)
    return Message(**values)


def test_split_message_validates_channel_sender_and_correlation():
    assert _validate_message(_split_message(), "client-0")
    assert not _validate_message(_split_message(sender="client-1"), "client-0")
    assert not _validate_message(_split_message(round=0), "client-0")
    assert not _validate_message(_split_message(step=0), "client-0")


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


def test_duplicate_split_step_is_rejected_as_replay():
    pending = {}
    message = _split_message()

    _store_pending_batch(pending, message, "client-0")

    with pytest.raises(ValueError, match="Duplicate split step"):
        _store_pending_batch(pending, message, "client-0")


class RecordingDownlink:
    def __init__(self):
        self.messages = []

    def send(self, message):
        self.messages.append(message)


def test_stale_batch_sends_correlated_error_to_waiting_clients(monkeypatch):
    message = _split_message(round=2, step=3)
    pending = {(2, 3): {"client-0": message}}
    timestamps = {(2, 3): 0.0}
    downlink = RecordingDownlink()
    channels = {"client-0": {"downlink": downlink}}
    monkeypatch.setattr(
        "src.splitfed.split_server.time.monotonic", lambda: 10.0
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


def test_batch_becomes_ready_when_missing_client_finished_round():
    batch = {"client-1": _train_message("client-1", step=2)}

    assert not _batch_is_ready(batch, {"client-0", "client-1"}, set())
    assert _batch_is_ready(
        batch,
        {"client-0", "client-1"},
        {"client-0"},
    )


def test_accumulated_server_gradients_are_averaged_before_step():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    parameter.grad = torch.tensor([6.0])

    _step_accumulated_gradients([parameter], optimizer, batch_count=3)

    assert torch.equal(parameter.detach(), torch.tensor([-1.0]))
    assert parameter.grad is None


def test_remainder_gradient_uses_actual_batch_count():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    parameter.grad = torch.tensor([6.0])

    _step_accumulated_gradients([parameter], optimizer, batch_count=2)

    assert torch.equal(parameter.detach(), torch.tensor([-2.0]))


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


def test_parallel_client_gradient_uses_full_batch_loss():
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

    gradients, _ = _forward_parallel(
        {"client-0": message}, model, criterion, torch.device("cpu")
    )

    expected = torch.sigmoid(activations) - labels
    assert torch.allclose(gradients["client-0"], expected)
