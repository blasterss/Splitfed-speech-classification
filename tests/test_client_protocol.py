import pytest
import torch

from src.splitfed.client import _extract_payload
from src.transport.message import Message


def _response(**overrides):
    values = {
        "type": "gradients",
        "sender": "split_server",
        "round": 2,
        "step": 3,
        "payload": {"gradients": torch.ones(1)},
    }
    values.update(overrides)
    return Message(**values)


def test_extract_payload_accepts_correlated_split_response():
    tensor = _extract_payload(
        _response(),
        "gradients",
        "gradients",
        "client-0",
        2,
        3,
    )

    assert torch.equal(tensor, torch.ones(1))


@pytest.mark.parametrize(
    "overrides",
    [
        {"sender": "client-1"},
        {"type": "logits"},
        {"round": 1},
        {"step": 4},
    ],
)
def test_extract_payload_rejects_uncorrelated_response(overrides):
    assert (
        _extract_payload(
            _response(**overrides),
            "gradients",
            "gradients",
            "client-0",
            2,
            3,
        )
        is None
    )
