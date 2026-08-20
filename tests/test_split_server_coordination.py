from src.splitfed.split_server import _batch_is_ready, _validate_message
from src.transport.message import Message, MessageType


def _train_message(sender, step):
    return Message(
        type=MessageType.TRAIN_STEP,
        sender=sender,
        round=1,
        step=step,
        payload={},
    )


def test_round_end_is_a_typed_protocol_message():
    message = Message(
        type="round_end", sender="client-0", round=1, step=2, payload={}
    )

    assert message.type is MessageType.ROUND_END
    assert _validate_message(message, "client-0")


def test_batch_becomes_ready_when_missing_client_finished_round():
    batch = {"client-1": _train_message("client-1", step=2)}

    assert not _batch_is_ready(batch, {"client-0", "client-1"}, set())
    assert _batch_is_ready(
        batch,
        {"client-0", "client-1"},
        {"client-0"},
    )
