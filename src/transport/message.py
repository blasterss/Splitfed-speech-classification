from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4

MESSAGE_PROTOCOL = "secureasr.transport"
MESSAGE_PROTOCOL_VERSION = 4


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
    deadline_at: float | None = None

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
        if self.deadline_at is not None and (
            not isinstance(self.deadline_at, (int, float))
            or not math.isfinite(self.deadline_at)
            or self.deadline_at <= 0
        ):
            raise ValueError(
                "Message deadline_at must be a positive timestamp"
            )

    def ensure_deadline(self, timeout: float) -> None:
        """Stamp an unset deadline and reject messages already expired."""
        now = time.time()
        if self.deadline_at is None:
            self.deadline_at = now + timeout
        if self.deadline_at <= now:
            raise TimeoutError(
                f"Message {self.request_id} deadline has already expired"
            )

    def is_expired(self, now: float | None = None) -> bool:
        """Return true for missing or elapsed wire deadlines."""
        return self.deadline_at is None or self.deadline_at <= (
            time.time() if now is None else now
        )

    def validate_for_receive(self, now: float | None = None) -> None:
        """Reject envelopes that reached a consumer without live budget."""
        if self.deadline_at is None:
            raise ValueError("Received message has no deadline")
        if self.is_expired(now):
            raise TimeoutError(
                f"Message {self.request_id} deadline expired before receive"
            )

    def to_bytes(self) -> bytes:
        from .serialization import serialize_message

        return serialize_message(self)

    @classmethod
    def from_bytes(cls, data: bytes) -> Message:
        from .serialization import deserialize_message

        return deserialize_message(data, cls)

    def __repr__(self) -> str:
        payload_keys = list(self.payload.keys())
        return (
            f"Message(type={self.type.value!r}, sender={self.sender!r}, "
            f"round={self.round}, step={self.step}, "
            f"request_id={self.request_id!r}, "
            f"deadline_at={self.deadline_at!r}, "
            f"payload_keys={payload_keys})"
        )
