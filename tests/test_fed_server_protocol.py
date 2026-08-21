import queue
import time
from types import SimpleNamespace

import pytest
import torch

from src.schema import AggregationStrategy
from src.splitfed.fed_server import _fed_server_worker
from src.splitfed.fed_server.protocol import (
    quorum_decision,
    validate_client_update,
)
from src.transport.message import Message
from src.utils.persistence import deserialize_state_dict

FUTURE_DEADLINE = time.time() + 3600


def _update(**overrides):
    values = {
        "type": "client_update",
        "sender": "client-0",
        "round": 2,
        "step": 1,
        "deadline_at": FUTURE_DEADLINE,
        "payload": {
            "state_dict": {"weight": torch.ones(2)},
            "dataset_size": 3,
        },
    }
    values.update(overrides)
    return Message(**values)


def test_client_update_accepts_matching_round_and_schema():
    message = _update()

    state, size = validate_client_update(
        message,
        expected_client_id="client-0",
        expected_round=2,
        expected_schema={"weight": (torch.Size([2]), torch.float32)},
    )

    assert state is message.payload["state_dict"]
    assert size == 3


def test_client_update_rejects_expired_deadline():
    with pytest.raises(TimeoutError, match="deadline"):
        validate_client_update(
            _update(deadline_at=time.time() - 1),
            expected_client_id="client-0",
            expected_round=2,
            expected_schema={"weight": (torch.Size([2]), torch.float32)},
        )


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
        validate_client_update(
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
        quorum_decision(
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


class SequenceUplink:
    def __init__(self, messages):
        self.messages = iter(messages)

    def recv_nowait(self):
        return next(self.messages, None)


class FakeStopEvent:
    def __init__(self):
        self.stopped = False

    def is_set(self):
        return self.stopped

    def set(self):
        self.stopped = True

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


def test_worker_aggregates_partial_quorum_and_catches_up_late_client():
    stop_event = FakeStopEvent()
    broadcasts = []
    late_update = _update(
        sender="client-2",
        payload={
            "state_dict": {"weight": torch.tensor([100.0])},
            "dataset_size": 5,
        },
    )
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
        "client-2": late_update,
    }
    channels = {
        client_id: {
            "uplink": (
                SequenceUplink([None, message])
                if client_id == "client-2"
                else FakeUplink(message)
            ),
            "downlink": FakeDownlink(broadcasts, stop_event),
        }
        for client_id, message in updates.items()
    }
    config = SimpleNamespace(
        seed=42,
        device="cpu",
        min_clients=2,
        quorum_timeout_sec=0,
        strategy=AggregationStrategy.weighted_fedavg,
    )
    result_queue = FakeResultQueue()

    _fed_server_worker(config, channels, 3, stop_event, result_queue)

    assert len(broadcasts) == 3
    assert all(message.round == 2 for message in broadcasts)
    assert broadcasts[0].request_id == updates["client-0"].request_id
    assert broadcasts[1].request_id == updates["client-1"].request_id
    assert broadcasts[2].request_id == late_update.request_id
    assert all(
        torch.equal(message.payload["weight"], torch.tensor([7.5]))
        for message in broadcasts
    )
    saved_state = deserialize_state_dict(result_queue.value)
    assert torch.equal(saved_state["weight"], torch.tensor([7.5]))


def test_worker_rejects_replayed_federated_request():
    stop_event = FakeStopEvent()
    broadcasts = []
    update = _update(sender="client-0")
    channels = {
        "client-0": {
            "uplink": SequenceUplink([update, update]),
            "downlink": FakeDownlink(broadcasts, stop_event),
        }
    }
    config = SimpleNamespace(
        seed=42,
        device="cpu",
        min_clients=1,
        quorum_timeout_sec=0,
        strategy=AggregationStrategy.fedavg,
    )

    failure_queue = queue.Queue(maxsize=1)

    with pytest.raises(ValueError, match="Replay"):
        _fed_server_worker(
            config,
            channels,
            1,
            stop_event,
            FakeResultQueue(),
            failure_queue,
        )

    assert len(broadcasts) == 1
    failure = failure_queue.get_nowait()
    assert failure["component"] == "fed_server"
    assert failure["client_id"] == "client-0"
    assert failure["round"] == 2
    assert failure["step"] == 1
    assert failure["exception_type"] == "ValueError"
