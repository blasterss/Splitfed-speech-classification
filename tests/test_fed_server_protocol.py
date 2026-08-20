import pytest
import torch

from src.splitfed.fed_server import _validate_client_update
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
