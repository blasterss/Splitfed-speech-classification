import queue
import time

import pytest

from src.schema import QueueChannelConfig, TransportType
from src.transport.base import ChannelCancelled, ChannelFactory, QueueChannel
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
