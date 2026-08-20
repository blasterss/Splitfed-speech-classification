from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4

MESSAGE_PROTOCOL = "secureasr.queue"
MESSAGE_PROTOCOL_VERSION = 2


class MessageType(str, Enum):
    # split-learning
    TRAIN_STEP = "train_step"
    GRADIENTS = "gradients"
    EVAL_STEP = "eval_step"
    LOGITS = "logits"
    ROUND_END = "round_end"

    # federated
    CLIENT_UPDATE = "client_update"
    GLOBAL_UPDATE = "global_update"

    # control
    ACK = "ack"
    ERROR = "error"


@dataclass
class Message:
    type: MessageType
    sender: str
    round: int
    step: int
    payload: dict[str, Any] = field(default_factory=dict)
    protocol: str = MESSAGE_PROTOCOL
    protocol_version: int = MESSAGE_PROTOCOL_VERSION
    request_id: str = field(default_factory=lambda: uuid4().hex)

    def __post_init__(self) -> None:
        if not isinstance(self.type, MessageType):
            self.type = MessageType(self.type)
        if self.protocol != MESSAGE_PROTOCOL:
            raise ValueError(f"Unsupported message protocol {self.protocol!r}")
        if self.protocol_version != MESSAGE_PROTOCOL_VERSION:
            raise ValueError(
                "Unsupported message protocol version "
                f"{self.protocol_version!r}"
            )
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("Message request_id must be a non-empty string")
        if len(self.request_id) > 128:
            raise ValueError("Message request_id exceeds 128 characters")

    # ------------------------------------------------------------------
    # Serialization (stubs for future gRPC transport)
    # ------------------------------------------------------------------
    def to_bytes(self) -> bytes:
        raise NotImplementedError

    @classmethod
    def from_bytes(cls, data: bytes) -> Message:
        raise NotImplementedError

    def __repr__(self) -> str:
        payload_keys = list(self.payload.keys())
        return (
            f"Message(type={self.type.value!r}, sender={self.sender!r}, "
            f"round={self.round}, step={self.step}, "
            f"request_id={self.request_id!r}, "
            f"payload_keys={payload_keys})"
        )
