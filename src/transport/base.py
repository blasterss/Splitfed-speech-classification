from __future__ import annotations

import multiprocessing as mp
import queue
import time
from abc import ABC, abstractmethod

from ..schema import GRPCChannelConfig, QueueChannelConfig
from .message import Message


def _payload_nbytes(value) -> int:
    """Estimate payload bytes without serializing or copying tensors."""
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.numel() * value.element_size()
    except ImportError:
        pass
    if isinstance(value, dict):
        return sum(
            len(str(key).encode("utf-8")) + _payload_nbytes(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return sum(_payload_nbytes(item) for item in value)
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    if isinstance(value, bytes):
        return len(value)
    return 8


def estimate_message_bytes(message: Message) -> int:
    """Return a stable logical-size estimate for a local transport message."""
    envelope = (
        str(message.sender),
        message.type.value,
        message.protocol,
        message.request_id,
    )
    return (
        sum(len(item.encode("utf-8")) for item in envelope)
        + 32
        + (_payload_nbytes(message.payload))
    )


class ChannelCancelled(RuntimeError):
    """Raised when a shared cancellation event interrupts a channel wait."""


class Channel(ABC):
    @abstractmethod
    def send(self, msg: Message) -> None:
        """Send a message. Blocks if the channel is full."""

    @abstractmethod
    def recv(self) -> Message:
        """Receive a message. Blocks until one is available."""

    @abstractmethod
    def recv_nowait(self) -> Message | None:
        """
        Non-blocking receive.

        Returns the next Message if one is immediately available,
        or None if the channel is empty.
        """


class QueueChannel(Channel):
    def __init__(
        self,
        maxsize: int = 0,
        timeout: float = 60,
        mp_context=None,
        stop_event=None,
    ):
        context = mp_context or mp.get_context("spawn")
        self.queue = context.Queue(maxsize=maxsize)
        self.timeout = timeout
        self.stop_event = stop_event
        self._bytes_sent = context.Value("Q", 0)
        self._messages_sent = context.Value("Q", 0)

    def send(self, msg: Message) -> None:
        msg.ensure_deadline(self.timeout)
        self.queue.put(msg, timeout=self.timeout)
        with self._bytes_sent.get_lock():
            self._bytes_sent.value += estimate_message_bytes(msg)
        with self._messages_sent.get_lock():
            self._messages_sent.value += 1

    def statistics(self) -> dict[str, int]:
        return {
            "messages_sent": self._messages_sent.value,
            "bytes_sent": self._bytes_sent.value,
        }

    def recv(self) -> Message:
        deadline = time.monotonic() + self.timeout
        while True:
            if self.stop_event is not None and self.stop_event.is_set():
                raise ChannelCancelled("Channel receive cancelled")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise queue.Empty
            try:
                message = self.queue.get(timeout=min(remaining, 0.1))
            except queue.Empty:
                continue
            message.validate_for_receive()
            return message

    def recv_nowait(self) -> Message | None:
        try:
            message = self.queue.get_nowait()
        except queue.Empty:
            return None
        message.validate_for_receive()
        return message


class GrpcChannel(Channel):
    def __init__(self) -> None:
        raise NotImplementedError(
            "GrpcChannel is not yet implemented. "
            "Implement send/recv/recv_nowait using a gRPC stub "
            "and wire Message via Message.to_bytes() / Message.from_bytes()."
        )

    def send(self, msg: Message) -> None:
        raise NotImplementedError

    def recv(self) -> Message:
        raise NotImplementedError

    def recv_nowait(self) -> Message | None:
        raise NotImplementedError


class ChannelFactory:
    @staticmethod
    def create(
        channel_params: QueueChannelConfig | GRPCChannelConfig,
        mp_context=None,
        stop_event=None,
    ) -> Channel:
        transport = channel_params.transport

        if transport == "queue":
            return QueueChannel(
                maxsize=channel_params.maxsize,
                timeout=channel_params.timeout,
                mp_context=mp_context,
                stop_event=stop_event,
            )
        elif transport == "grpc":
            return GrpcChannel()
        else:
            raise ValueError(f"Unknown channel transport: {transport!r}")
