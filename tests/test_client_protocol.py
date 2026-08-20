import pytest
import torch

from src.splitfed.client import _extract_payload, _validate_global_update
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


def _global_response(state_dict, **overrides):
    values = {
        "type": "global_update",
        "sender": "fed_server",
        "round": 2,
        "step": 1,
        "payload": state_dict,
    }
    values.update(overrides)
    return Message(**values)


def test_global_update_accepts_matching_state_schema():
    expected = {"weight": torch.ones(2, dtype=torch.float32)}
    received = {"weight": torch.zeros(2, dtype=torch.float32)}

    assert (
        _validate_global_update(_global_response(received), expected, 2)
        is received
    )


@pytest.mark.parametrize(
    "response,match",
    [
        (_global_response({}, sender="client-1"), "sender"),
        (_global_response({}, round=1), "round"),
        (_global_response({"other": torch.ones(2)}), "keys"),
        (_global_response({"weight": torch.ones(3)}), "shape"),
        (
            _global_response({"weight": torch.ones(2, dtype=torch.float64)}),
            "dtype",
        ),
    ],
)
def test_global_update_rejects_invalid_envelope_or_state(response, match):
    expected = {"weight": torch.ones(2, dtype=torch.float32)}

    with pytest.raises(ValueError, match=match):
        _validate_global_update(response, expected, 2)
