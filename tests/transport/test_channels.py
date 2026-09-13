import multiprocessing as mp
import queue
import socket
import time

import pytest

from src.schema import GRPCChannelConfig, QueueChannelConfig, TransportType
from src.transport.base import (
    ChannelCancelled,
    ChannelFactory,
    GrpcChannel,
    QueueChannel,
)
from src.transport.message import Message, MessageType


def test_message_coerces_string_type_to_enum():
    message = Message(
        type="client_update",
        sender="client-1",
        round=2,
        step=3,
        payload={"dataset_size": 10},
    )

    assert message.type is MessageType.CLIENT_UPDATE
    assert message.payload == {"dataset_size": 10}


def test_queue_channel_round_trips_messages():
    channel = QueueChannel(timeout=1)
    message = Message(
        type=MessageType.ACK,
        sender="server",
        round=1,
        step=1,
    )

    assert channel.recv_nowait() is None
    channel.send(message)

    received = channel.recv()

    assert received == message
    assert received.deadline_at is not None
    assert received.deadline_at > time.time()


def test_queue_channel_rejects_expired_message_before_enqueue():
    channel = QueueChannel(timeout=1)
    message = Message(
        type=MessageType.ACK,
        sender="server",
        round=1,
        step=1,
        deadline_at=time.time() - 1,
    )

    with pytest.raises(TimeoutError, match="deadline"):
        channel.send(message)

    assert channel.recv_nowait() is None


def test_queue_channel_rejects_expired_message_at_receive_boundary():
    channel = QueueChannel(timeout=1)
    message = Message(
        type=MessageType.ACK,
        sender="server",
        round=1,
        step=1,
        deadline_at=time.time() - 1,
    )
    channel.queue.put(message, timeout=1)

    with pytest.raises(TimeoutError, match="deadline"):
        channel.recv()


def test_queue_channel_recv_raises_timeout_when_empty():
    channel = QueueChannel(timeout=0.01)

    with pytest.raises(queue.Empty):
        channel.recv()


def test_queue_channel_recv_is_interrupted_by_cancellation():
    class SetEvent:
        @staticmethod
        def is_set():
            return True

    channel = QueueChannel(timeout=60, stop_event=SetEvent())

    started = time.monotonic()
    with pytest.raises(ChannelCancelled, match="cancelled"):
        channel.recv()
    assert time.monotonic() - started < 0.5


def test_channel_factory_creates_queue_channel():
    config = QueueChannelConfig(
        transport=TransportType.queue,
        name="test",
        maxsize=1,
        timeout=1,
    )

    channel = ChannelFactory.create(config)

    assert isinstance(channel, QueueChannel)
    assert channel.timeout == 1


def _free_address():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return f"127.0.0.1:{listener.getsockname()[1]}"


def _receive_in_spawned_process(channel, result_queue):
    try:
        result_queue.put(channel.recv())
    finally:
        channel.close()


def test_grpc_channel_round_trips_through_spawned_receiver():
    context = mp.get_context("spawn")
    channel = GrpcChannel(
        address=_free_address(),
        timeout=5,
        mp_context=context,
    )
    result_queue = context.Queue()
    receiver = context.Process(
        target=_receive_in_spawned_process,
        args=(channel, result_queue),
    )
    receiver.start()
    message = Message(
        type="ack",
        sender="server",
        round=1,
        step=1,
    )

    try:
        channel.send(message)
        received = result_queue.get(timeout=5)
        receiver.join(timeout=5)
    finally:
        channel.close()
        if receiver.is_alive():
            receiver.terminate()
            receiver.join(timeout=5)

    assert receiver.exitcode == 0
    assert received == message
    assert channel.statistics()["messages_sent"] == 1
    assert channel.statistics()["bytes_sent"] > 0


def test_channel_factory_creates_insecure_grpc_channel():
    config = GRPCChannelConfig(
        transport=TransportType.grpc,
        name="test",
        addresses={1: "127.0.0.1:50051"},
        use_tls=False,
        timeout_sec=2,
    )

    channel = ChannelFactory.create(config, client_id=1)

    assert isinstance(channel, GrpcChannel)
    assert channel.address == "127.0.0.1:50051"
    assert channel.timeout == 2


def test_grpc_channel_rejects_unconfigured_tls():
    with pytest.raises(ValueError, match="TLS credentials"):
        GrpcChannel(address="127.0.0.1:50051", use_tls=True)
