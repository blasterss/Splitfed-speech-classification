"""Federated update validation and quorum decision contracts."""

import torch

from ...transport.base import Message


def state_schema(state_dict: dict) -> dict:
    return {
        key: (value.shape, value.dtype) for key, value in state_dict.items()
    }


def quorum_decision(
    update_count: int,
    client_count: int,
    min_clients: int,
    elapsed: float,
    timeout: float,
) -> str:
    if update_count >= client_count:
        return "aggregate"
    if elapsed < timeout:
        return "wait"
    if update_count >= min_clients:
        return "aggregate"
    return "fail"


def validate_client_update(
    message: Message,
    expected_client_id,
    expected_round: int | None,
    expected_schema: dict | None,
    require_aggregation_weight: bool = False,
) -> tuple[dict, int, int | None]:
    message.validate_for_receive()
    if message.sender != expected_client_id:
        raise ValueError("Invalid client update sender")
    if message.type != "client_update":
        raise ValueError("Invalid client update type")
    if expected_round is not None and message.round != expected_round:
        raise ValueError("Invalid client update round")
    if message.round <= 0 or message.step != 1:
        raise ValueError("Invalid client update round or step")
    if not isinstance(message.payload, dict):
        raise ValueError("Invalid client update payload")
    state_dict = message.payload.get("state_dict")
    dataset_size = message.payload.get("dataset_size")
    aggregation_weight = message.payload.get("aggregation_weight")
    if not isinstance(dataset_size, int) or isinstance(dataset_size, bool):
        raise ValueError("Invalid client update dataset_size")
    if dataset_size <= 0:
        raise ValueError("Invalid client update dataset_size")
    if aggregation_weight is not None and (
        not isinstance(aggregation_weight, int)
        or isinstance(aggregation_weight, bool)
        or aggregation_weight <= 0
    ):
        raise ValueError("Invalid client update aggregation_weight")
    if require_aggregation_weight and aggregation_weight is None:
        raise ValueError("Missing client update aggregation_weight")
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("Invalid client update state_dict")
    if not all(
        isinstance(value, torch.Tensor) for value in state_dict.values()
    ):
        raise ValueError("Invalid client update state tensor")
    if expected_schema is not None:
        if state_dict.keys() != expected_schema.keys():
            raise ValueError("Invalid client update state keys")
        for key, value in state_dict.items():
            shape, dtype = expected_schema[key]
            if value.shape != shape:
                raise ValueError(f"Invalid client update shape for {key}")
            if value.dtype != dtype:
                raise ValueError(f"Invalid client update dtype for {key}")
    return state_dict, dataset_size, aggregation_weight
