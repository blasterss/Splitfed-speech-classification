from __future__ import annotations

import queue
from abc import ABC, abstractmethod
from multiprocessing import Queue
from typing import Optional, Union

from .message import Message
from ..schema import QueueChannelConfig, GRPCChannelConfig


class Channel(ABC):
    @abstractmethod
    def send(self, msg: Message) -> None:
        """Send a message. Blocks if the channel is full."""

    @abstractmethod
    def recv(self) -> Message:
        """Receive a message. Blocks until one is available."""

    @abstractmethod
    def recv_nowait(self) -> Optional[Message]:
        """
        Non-blocking receive.

        Returns the next Message if one is immediately available,
        or None if the channel is empty.
        """


class QueueChannel(Channel):
    def __init__(
        self,
        maxsize: int = 0,
        timeout: int = 60,
    ):
        self.queue: Queue = Queue(maxsize=maxsize)
        self.timeout: int = timeout

    def send(self, msg: Message) -> None:
        self.queue.put(msg, timeout=self.timeout)

    def recv(self) -> Message:
        return self.queue.get(timeout=self.timeout)

    def recv_nowait(self) -> Optional[Message]:
        try:
            return self.queue.get_nowait()
        except queue.Empty:
            return None


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

    def recv_nowait(self) -> Optional[Message]:
        raise NotImplementedError


class ChannelFactory:
    @staticmethod
    def create(
        channel_params: Union[QueueChannelConfig, GRPCChannelConfig],
    ) -> Channel:
        transport = channel_params.transport

        if transport == "queue":
            return QueueChannel(
                maxsize=channel_params.maxsize,
                timeout=channel_params.timeout,
            )
        elif transport == "grpc":
            return GrpcChannel()
        else:
            raise ValueError(f"Unknown channel transport: {transport!r}")
