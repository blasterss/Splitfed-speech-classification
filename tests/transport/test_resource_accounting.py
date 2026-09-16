import torch

from src.transport.base import QueueChannel, estimate_message_bytes
from src.transport.message import Message


def test_queue_channel_counts_logical_tensor_payload_bytes():
    channel = QueueChannel(timeout=1)
    message = Message(
        type="train_step",
        sender="client-0",
        round=1,
        step=1,
        payload={"encoded": b"123456"},
    )

    expected = estimate_message_bytes(message)
    channel.send(message)
    channel.recv()

    assert channel.statistics() == {
        "messages_sent": 1,
        "bytes_sent": expected,
        "by_message_type": {"train_step": {"messages": 1, "bytes": expected}},
    }
    channel.queue.close()
    channel.queue.join_thread()

    tensor_message = Message(
        type="train_step",
        sender="client-0",
        round=1,
        step=1,
        payload={"activations": torch.zeros(2, 3, dtype=torch.float32)},
    )
    assert estimate_message_bytes(tensor_message) >= 24
