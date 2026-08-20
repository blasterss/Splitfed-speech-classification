import time

import pytest

from src.transport.message import (
    MESSAGE_PROTOCOL,
    MESSAGE_PROTOCOL_VERSION,
    Message,
)


def test_message_uses_current_protocol_identity_by_default():
    message = Message(type="ack", sender="controller", round=1, step=0)

    assert message.protocol == MESSAGE_PROTOCOL
    assert message.protocol_version == MESSAGE_PROTOCOL_VERSION
    assert message.request_id


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"protocol": "other"}, "protocol"),
        ({"protocol_version": MESSAGE_PROTOCOL_VERSION + 1}, "version"),
        ({"request_id": ""}, "request_id"),
    ],
)
def test_message_rejects_unsupported_protocol_identity(overrides, match):
    with pytest.raises(ValueError, match=match):
        Message(
            type="ack",
            sender="controller",
            round=1,
            step=0,
            **overrides,
        )


def test_message_reports_expired_deadline():
    expired = Message(
        type="ack",
        sender="controller",
        round=1,
        step=0,
        deadline_at=time.time() - 1,
    )

    assert expired.is_expired()
