from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from enum import Enum


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
    payload: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.type, MessageType):
            self.type = MessageType(self.type)

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
            f"payload_keys={payload_keys})"
        )
