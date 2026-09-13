"""Strict protobuf serialization for transport messages."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from google.protobuf.message import DecodeError

from .proto import message_pb2

_DTYPES = {
    str(dtype): dtype
    for dtype in (
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    )
}


def serialize_message(message) -> bytes:
    """Serialize a validated Message without pickle or executable objects."""
    if message.deadline_at is None:
        raise ValueError("Cannot serialize a message without a deadline")
    wire = message_pb2.Message(
        type=message.type.value,
        sender=message.sender,
        round=message.round,
        step=message.step,
        payload=_encode_mapping(message.payload),
        protocol=message.protocol,
        protocol_version=message.protocol_version,
        request_id=message.request_id,
        deadline_at=message.deadline_at,
    )
    return wire.SerializeToString(deterministic=True)


def deserialize_message(data: bytes, message_class):
    """Deserialize protobuf bytes and re-run the Message envelope checks."""
    if not isinstance(data, bytes):
        raise TypeError("Serialized message must be bytes")
    wire = message_pb2.Message()
    try:
        wire.ParseFromString(data)
    except DecodeError as exc:
        raise ValueError("Invalid protobuf message") from exc
    return message_class(
        type=wire.type,
        sender=wire.sender,
        round=wire.round,
        step=wire.step,
        payload=_decode_mapping(wire.payload),
        protocol=wire.protocol,
        protocol_version=wire.protocol_version,
        request_id=wire.request_id,
        deadline_at=wire.deadline_at,
    )


def _encode_mapping(value: Mapping[str, Any]) -> message_pb2.Mapping:
    if not isinstance(value, Mapping):
        raise TypeError("Message payload mappings must implement Mapping")
    result = message_pb2.Mapping()
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise TypeError("Message payload keys must be non-empty strings")
        entry = result.entries.add(key=key)
        _encode_value(item, entry.value)
    return result


def _encode_value(value: Any, result: message_pb2.Value) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        if str(tensor.dtype) not in _DTYPES:
            raise TypeError(f"Unsupported tensor dtype {tensor.dtype}")
        result.tensor_value.shape.extend(tensor.shape)
        result.tensor_value.dtype = str(tensor.dtype)
        result.tensor_value.data = (
            tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        )
    elif isinstance(value, Mapping):
        result.mapping_value.CopyFrom(_encode_mapping(value))
    elif isinstance(value, bool):
        result.bool_value = value
    elif isinstance(value, int):
        result.int_value = value
    elif isinstance(value, float):
        result.float_value = value
    elif isinstance(value, str):
        result.string_value = value
    elif isinstance(value, bytes):
        result.bytes_value = value
    else:
        raise TypeError(
            f"Unsupported message payload type {type(value).__name__}"
        )


def _decode_mapping(value: message_pb2.Mapping) -> dict[str, Any]:
    result = {}
    for entry in value.entries:
        if not entry.key or entry.key in result:
            raise ValueError("Payload keys must be unique non-empty strings")
        result[entry.key] = _decode_value(entry.value)
    return result


def _decode_value(value: message_pb2.Value) -> Any:
    kind = value.WhichOneof("kind")
    if kind == "tensor_value":
        return _decode_tensor(value.tensor_value)
    if kind == "mapping_value":
        return _decode_mapping(value.mapping_value)
    if kind is None:
        raise ValueError("Payload value has no protobuf kind")
    return getattr(value, kind)


def _decode_tensor(value: message_pb2.Tensor) -> torch.Tensor:
    dtype = _DTYPES.get(value.dtype)
    if dtype is None:
        raise ValueError(f"Unsupported tensor dtype {value.dtype!r}")
    if any(dimension < 0 for dimension in value.shape):
        raise ValueError("Tensor shape dimensions must be non-negative")
    expected = 1
    for dimension in value.shape:
        expected *= dimension
    if expected == 0:
        if value.data:
            raise ValueError("Empty tensor shape must not carry tensor bytes")
        return torch.empty(tuple(value.shape), dtype=dtype)
    tensor = torch.frombuffer(bytearray(value.data), dtype=dtype).clone()
    if tensor.numel() != expected:
        raise ValueError(
            "Tensor byte count does not match its declared shape and dtype"
        )
    return tensor.reshape(tuple(value.shape))
