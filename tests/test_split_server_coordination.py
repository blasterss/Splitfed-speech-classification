import torch

from src.splitfed.split_server import (
    _batch_is_ready,
    _forward_parallel,
    _step_accumulated_gradients,
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
