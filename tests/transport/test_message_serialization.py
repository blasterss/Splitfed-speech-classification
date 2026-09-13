import time

import pytest
import torch

from src.transport.message import Message


def test_message_protobuf_round_trip_preserves_supported_payloads():
    message = Message(
        type="client_update",
        sender="client-1",
        round=2,
        step=3,
        request_id="request-1",
        deadline_at=time.time() + 30,
        payload={
            "state_dict": {
                "weight": torch.tensor([[1.5, -2.0]], dtype=torch.float32),
                "counter": torch.tensor(4, dtype=torch.int64),
                "empty": torch.empty((0, 2), dtype=torch.float32),
            },
            "dataset_size": 17,
            "accepted": True,
            "reason": "ok",
            "ratio": 0.5,
            "opaque": b"bytes",
        },
    )

    restored = Message.from_bytes(message.to_bytes())

    assert restored.type == message.type
    assert restored.sender == message.sender
    assert restored.round == message.round
    assert restored.step == message.step
    assert restored.request_id == message.request_id
    assert restored.deadline_at == message.deadline_at
    assert restored.payload["dataset_size"] == 17
    assert restored.payload["accepted"] is True
    assert restored.payload["reason"] == "ok"
    assert restored.payload["ratio"] == 0.5
    assert restored.payload["opaque"] == b"bytes"
    assert torch.equal(
        restored.payload["state_dict"]["weight"],
        message.payload["state_dict"]["weight"],
    )
    assert torch.equal(
        restored.payload["state_dict"]["counter"],
        message.payload["state_dict"]["counter"],
    )
    assert restored.payload["state_dict"]["empty"].shape == (0, 2)


def test_message_serialization_requires_a_wire_deadline():
    message = Message(type="ack", sender="server", round=1, step=1)

    with pytest.raises(ValueError, match="deadline"):
        message.to_bytes()


def test_message_serialization_rejects_unsupported_payload_type():
    message = Message(
        type="ack",
        sender="server",
        round=1,
        step=1,
        deadline_at=time.time() + 30,
        payload={"unsafe": object()},
    )

    with pytest.raises(TypeError, match="Unsupported"):
        message.to_bytes()


def test_message_deserialization_rejects_malformed_protobuf():
    with pytest.raises(ValueError, match="protobuf"):
        Message.from_bytes(b"not-protobuf")
