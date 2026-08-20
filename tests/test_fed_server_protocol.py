from types import SimpleNamespace

import pytest
import torch

from src.splitfed.fed_server import (
    _fed_server_worker,
    _quorum_decision,
    _validate_client_update,
)
from src.transport.message import Message


def _update(**overrides):
    values = {
        "type": "client_update",
        "sender": "client-0",
        "round": 2,
        "step": 1,
        "payload": {
            "state_dict": {"weight": torch.ones(2)},
            "dataset_size": 3,
        },
    }
    values.update(overrides)
    return Message(**values)


def test_client_update_accepts_matching_round_and_schema():
    message = _update()

    state, size = _validate_client_update(
        message,
        expected_client_id="client-0",
        expected_round=2,
        expected_schema={"weight": (torch.Size([2]), torch.float32)},
    )

    assert state is message.payload["state_dict"]
    assert size == 3


@pytest.mark.parametrize(
    "message,match",
    [
        (_update(sender="client-1"), "sender"),
        (_update(round=1), "round"),
        (
            _update(
                payload={
                    "state_dict": {"weight": torch.ones(2)},
                    "dataset_size": 0,
                }
            ),
            "dataset_size",
        ),
        (
            _update(
                payload={
                    "state_dict": {"weight": torch.ones(3)},
                    "dataset_size": 3,
                }
            ),
            "shape",
        ),
    ],
)
def test_client_update_rejects_invalid_protocol(message, match):
    with pytest.raises(ValueError, match=match):
        _validate_client_update(
            message,
            expected_client_id="client-0",
            expected_round=2,
            expected_schema={"weight": (torch.Size([2]), torch.float32)},
        )


@pytest.mark.parametrize(
    ("updates", "clients", "minimum", "elapsed", "expected"),
    [
        (3, 3, 2, 0.1, "aggregate"),
        (2, 3, 2, 0.5, "wait"),
        (2, 3, 2, 1.0, "aggregate"),
        (1, 3, 2, 1.0, "fail"),
    ],
)
def test_partial_quorum_decision(updates, clients, minimum, elapsed, expected):
    assert (
        _quorum_decision(
            update_count=updates,
            client_count=clients,
            min_clients=minimum,
            elapsed=elapsed,
            timeout=1.0,
        )
        == expected
    )


class FakeUplink:
    def __init__(self, message=None):
        self.message = message

    def recv_nowait(self):
        message, self.message = self.message, None
        return message


class FakeStopEvent:
    def __init__(self):
        self.stopped = False

    def is_set(self):
        return self.stopped

    def wait(self, timeout):
        pass


class FakeDownlink:
    def __init__(self, messages, stop_event):
        self.messages = messages
        self.stop_event = stop_event

    def send(self, message):
        self.messages.append(message)
        if len(self.messages) == 3:
            self.stop_event.stopped = True


class FakeResultQueue:
    def __init__(self):
        self.value = None

    def put_nowait(self, value):
        self.value = value


def test_worker_aggregates_partial_quorum_and_broadcasts_to_all_clients():
    stop_event = FakeStopEvent()
    broadcasts = []
    updates = {
        "client-0": _update(
            sender="client-0",
            payload={
                "state_dict": {"weight": torch.tensor([0.0])},
                "dataset_size": 1,
            },
        ),
        "client-1": _update(
            sender="client-1",
            payload={
                "state_dict": {"weight": torch.tensor([10.0])},
                "dataset_size": 3,
            },
        ),
        "client-2": None,
    }
    channels = {
        client_id: {
            "uplink": FakeUplink(message),
            "downlink": FakeDownlink(broadcasts, stop_event),
        }
        for client_id, message in updates.items()
    }
    config = SimpleNamespace(seed=42, min_clients=2, quorum_timeout_sec=0)
    result_queue = FakeResultQueue()

    _fed_server_worker(config, channels, 3, stop_event, result_queue)

    assert len(broadcasts) == 3
    assert all(message.round == 2 for message in broadcasts)
    assert all(
        torch.equal(message.payload["weight"], torch.tensor([7.5]))
        for message in broadcasts
    )
    assert torch.equal(result_queue.value["weight"], torch.tensor([7.5]))
