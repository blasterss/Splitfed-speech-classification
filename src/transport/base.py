from __future__ import annotations

import multiprocessing as mp
import queue
from abc import ABC, abstractmethod

from ..schema import GRPCChannelConfig, QueueChannelConfig
from .message import Message


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
        timeout: int = 60,
        mp_context=None,
    ):
        context = mp_context or mp.get_context("spawn")
        self.queue = context.Queue(maxsize=maxsize)
        self.timeout: int = timeout

    def send(self, msg: Message) -> None:
        self.queue.put(msg, timeout=self.timeout)

    def recv(self) -> Message:
        return self.queue.get(timeout=self.timeout)

    def recv_nowait(self) -> Message | None:
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

    def recv_nowait(self) -> Message | None:
        raise NotImplementedError


class ChannelFactory:
    @staticmethod
    def create(
        channel_params: QueueChannelConfig | GRPCChannelConfig,
        mp_context=None,
    ) -> Channel:
        transport = channel_params.transport

        if transport == "queue":
            return QueueChannel(
                maxsize=channel_params.maxsize,
                timeout=channel_params.timeout,
                mp_context=mp_context,
            )
        elif transport == "grpc":
            return GrpcChannel()
        else:
            raise ValueError(f"Unknown channel transport: {transport!r}")
