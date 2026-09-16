from __future__ import annotations

import multiprocessing as mp
import queue
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

import grpc

from ..schema import GRPCChannelConfig, QueueChannelConfig
from .message import Message, MessageType
from .proto import message_pb2, message_pb2_grpc


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


def _message_type_counters(context) -> dict[str, dict[str, object]]:
    return {
        message_type.value: {
            "messages": context.Value("Q", 0),
            "bytes": context.Value("Q", 0),
        }
        for message_type in MessageType
    }


def _record_message_type(
    counters, message_type: MessageType, size: int
) -> None:
    counter = counters[message_type.value]
    with counter["messages"].get_lock():
        counter["messages"].value += 1
    with counter["bytes"].get_lock():
        counter["bytes"].value += size


def _message_type_statistics(counters) -> dict[str, dict[str, int]]:
    return {
        message_type: {
            "messages": counter["messages"].value,
            "bytes": counter["bytes"].value,
        }
        for message_type, counter in counters.items()
        if counter["messages"].value or counter["bytes"].value
    }


class Channel(ABC):
    @abstractmethod
    def send(self, msg: Message) -> None:
        """Send a message. Blocks if the channel is full."""

    @abstractmethod
    def recv(self) -> Message:
        """Receive a message. Blocks until one is available."""

    @abstractmethod
    def recv_nowait(self) -> Message | None:
        """Non-blocking receive.

        Returns the next Message if one is immediately available,
        or None if the channel is empty.
        """

    def close(self) -> None:
        """Release transport-owned resources, if any."""
        return None

    def statistics(self) -> dict[str, int]:
        """Return transport counters owned by this logical channel."""
        return {"messages_sent": 0, "bytes_sent": 0}


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
        self._by_message_type = _message_type_counters(context)

    def send(self, msg: Message) -> None:
        msg.ensure_deadline(self.timeout)
        self.queue.put(msg, timeout=self.timeout)
        size = estimate_message_bytes(msg)
        with self._bytes_sent.get_lock():
            self._bytes_sent.value += size
        with self._messages_sent.get_lock():
            self._messages_sent.value += 1
        _record_message_type(self._by_message_type, msg.type, size)

    def statistics(self) -> dict[str, int]:
        return {
            "messages_sent": self._messages_sent.value,
            "bytes_sent": self._bytes_sent.value,
            "by_message_type": _message_type_statistics(self._by_message_type),
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


class _GrpcReceiver(message_pb2_grpc.TransportServicer):
    def __init__(self, messages: queue.Queue):
        self.messages = messages

    def Send(self, request, context):
        try:
            message = Message.from_bytes(request.SerializeToString())
            message.validate_for_receive()
            timeout = max(0.0, context.time_remaining())
            self.messages.put(message, timeout=timeout)
        except queue.Full:
            context.abort(
                grpc.StatusCode.RESOURCE_EXHAUSTED, "channel is full"
            )
        except (TypeError, ValueError, TimeoutError) as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        return message_pb2.SendReply()


class GrpcChannel(Channel):
    """A synchronous, pickle-safe gRPC-backed directional channel."""

    def __init__(
        self,
        address: str,
        timeout: float = 30,
        maxsize: int = 0,
        max_message_bytes: int = 64 * 1024 * 1024,
        use_tls: bool = False,
        mp_context=None,
        stop_event=None,
    ) -> None:
        if use_tls:
            raise ValueError("TLS credentials are not implemented")
        context = mp_context or mp.get_context("spawn")
        self.address = address
        self.timeout = timeout
        self.maxsize = maxsize
        self.max_message_bytes = max_message_bytes
        self.stop_event = stop_event
        self._bytes_sent = context.Value("Q", 0)
        self._messages_sent = context.Value("Q", 0)
        self._by_message_type = _message_type_counters(context)
        self._server = None
        self._messages = None
        self._client_channel = None
        self._stub = None

    def send(self, msg: Message) -> None:
        msg.ensure_deadline(self.timeout)
        request = message_pb2.Message.FromString(msg.to_bytes())
        size = len(request.SerializeToString())
        if size > self.max_message_bytes:
            raise ValueError(
                f"Serialized message exceeds {self.max_message_bytes} bytes"
            )
        self._ensure_client()
        while True:
            if self.stop_event is not None and self.stop_event.is_set():
                raise ChannelCancelled("Channel send cancelled")
            remaining = msg.deadline_at - time.time()
            if remaining <= 0:
                raise TimeoutError(
                    f"Message {msg.request_id} deadline expired during send"
                )
            try:
                self._stub.Send(request, timeout=min(remaining, 0.1))
                break
            except grpc.RpcError as exc:
                if exc.code() in (
                    grpc.StatusCode.UNAVAILABLE,
                    grpc.StatusCode.DEADLINE_EXCEEDED,
                ):
                    continue
                if exc.code() is grpc.StatusCode.INVALID_ARGUMENT:
                    raise ValueError(exc.details()) from exc
                raise RuntimeError(
                    f"gRPC send failed with {exc.code().name}: {exc.details()}"
                ) from exc
        with self._bytes_sent.get_lock():
            self._bytes_sent.value += size
        with self._messages_sent.get_lock():
            self._messages_sent.value += 1
        _record_message_type(self._by_message_type, msg.type, size)

    def recv(self) -> Message:
        self._ensure_server()
        deadline = time.monotonic() + self.timeout
        while True:
            if self.stop_event is not None and self.stop_event.is_set():
                raise ChannelCancelled("Channel receive cancelled")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise queue.Empty
            try:
                message = self._messages.get(timeout=min(remaining, 0.1))
            except queue.Empty:
                continue
            message.validate_for_receive()
            return message

    def recv_nowait(self) -> Message | None:
        self._ensure_server()
        try:
            message = self._messages.get_nowait()
        except queue.Empty:
            return None
        message.validate_for_receive()
        return message

    def statistics(self) -> dict[str, int]:
        return {
            "messages_sent": self._messages_sent.value,
            "bytes_sent": self._bytes_sent.value,
            "by_message_type": _message_type_statistics(self._by_message_type),
        }

    def close(self) -> None:
        if self._client_channel is not None:
            self._client_channel.close()
            self._client_channel = None
            self._stub = None
        if self._server is not None:
            self._server.stop(grace=0).wait(timeout=self.timeout)
            self._server = None
            self._messages = None

    def _ensure_client(self) -> None:
        if self._stub is None:
            self._client_channel = grpc.insecure_channel(
                self.address,
                options=(
                    (
                        "grpc.max_send_message_length",
                        self.max_message_bytes,
                    ),
                    (
                        "grpc.max_receive_message_length",
                        self.max_message_bytes,
                    ),
                ),
            )
            self._stub = message_pb2_grpc.TransportStub(self._client_channel)

    def _ensure_server(self) -> None:
        if self._server is not None:
            return
        self._messages = queue.Queue(maxsize=self.maxsize)
        self._server = grpc.server(
            ThreadPoolExecutor(max_workers=1),
            options=(
                ("grpc.max_receive_message_length", self.max_message_bytes),
                ("grpc.max_send_message_length", self.max_message_bytes),
            ),
        )
        message_pb2_grpc.add_TransportServicer_to_server(
            _GrpcReceiver(self._messages), self._server
        )
        if self._server.add_insecure_port(self.address) == 0:
            self._server = None
            self._messages = None
            raise RuntimeError(
                f"Could not bind gRPC channel to {self.address}"
            )
        self._server.start()


class ChannelFactory:
    @staticmethod
    def create(
        channel_params: QueueChannelConfig | GRPCChannelConfig,
        client_id: int | None = None,
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
            if client_id is None:
                raise ValueError("client_id is required for a gRPC channel")
            return GrpcChannel(
                address=channel_params.addresses[client_id],
                timeout=channel_params.timeout_sec,
                maxsize=channel_params.buffer_size,
                max_message_bytes=channel_params.max_message_bytes,
                use_tls=channel_params.use_tls,
                mp_context=mp_context,
                stop_event=stop_event,
            )
        else:
            raise ValueError(f"Unknown channel transport: {transport!r}")
