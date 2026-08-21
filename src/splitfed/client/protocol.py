"""Validation helpers for client-side split and federated responses."""

import torch

from ...logger import logger
from ...transport.base import Message

logger = logger.getChild("Client")


def _extract_payload(
    response: Message | None,
    key: str,
    expected_type: str,
    client_id: str,
    round: int,
    step: int,
    expected_request_id: str,
) -> object | None:
    """Safely extract a correlated tensor from a split-server response."""
    if response is None:
        logger.error(
            "Client %s: received None response from server (r=%d s=%d).",
            client_id,
            round,
            step,
        )
        return None

    try:
        response.validate_for_receive()
    except (ValueError, TimeoutError) as exc:
        logger.error(
            "Client %s: invalid response deadline: %s", client_id, exc
        )
        return None

    if (
        response.sender != "split_server"
        or response.type != expected_type
        or response.round != round
        or response.step != step
        or response.request_id != expected_request_id
    ):
        logger.error(
            "Client %s: uncorrelated server response "
            "(expected type=%s r=%d s=%d, got sender=%s type=%s r=%d s=%d).",
            client_id,
            expected_type,
            round,
            step,
            response.sender,
            response.type,
            response.round,
            response.step,
        )
        return None

    if not isinstance(getattr(response, "payload", None), dict):
        logger.error(
            "Client %s: invalid response payload type (r=%d s=%d, type=%s).",
            client_id,
            round,
            step,
            response.type,
        )
        return None

    if key not in response.payload:
        logger.error(
            "Client %s: missing key '%s' in server response (r=%d s=%d).",
            client_id,
            key,
            round,
            step,
        )
        return None

    tensor = response.payload[key]
    if not isinstance(tensor, torch.Tensor):
        logger.error(
            "Client %s: payload['%s'] is not a Tensor (r=%d s=%d, got %s).",
            client_id,
            key,
            round,
            step,
            type(tensor).__name__,
        )
        return None

    return tensor


def _validate_global_update(
    response: Message,
    expected_state: dict,
    round: int,
    expected_request_id: str,
) -> dict:
    response.validate_for_receive()
    if response.sender != "fed_server":
        raise ValueError("Invalid global update sender")
    if response.type != "global_update":
        raise ValueError("Invalid global update type")
    if response.round != round:
        raise ValueError("Invalid global update round")
    if response.step != 1:
        raise ValueError("Invalid global update step")
    if response.request_id != expected_request_id:
        raise ValueError("Invalid global update request_id")
    if not isinstance(response.payload, dict):
        raise ValueError("Invalid global update payload")
    if response.payload.keys() != expected_state.keys():
        raise ValueError("Invalid global update state keys")

    for key, expected in expected_state.items():
        value = response.payload[key]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Invalid global update tensor for {key}")
        if value.shape != expected.shape:
            raise ValueError(f"Invalid global update shape for {key}")
        if value.dtype != expected.dtype:
            raise ValueError(f"Invalid global update dtype for {key}")

    return response.payload
