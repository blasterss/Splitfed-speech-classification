"""Structured failure records shared by training processes."""

import queue
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

MAX_FAILURE_MESSAGE_CHARS = 4096
MAX_FAILURE_TRACEBACK_CHARS = 32768


def _bounded(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "...[truncated]"


@dataclass(frozen=True)
class FailureRecord:
    """Serializable first-failure diagnostic without model or data payloads."""

    schema_version: int
    timestamp_utc: str
    component: str
    client_id: int | None
    round: int | None
    step: int | None
    exception_type: str
    message: str
    traceback: str

    @classmethod
    def from_exception(
        cls,
        *,
        component: str,
        exception: BaseException,
        client_id: int | None = None,
        round: int | None = None,
        step: int | None = None,
    ) -> "FailureRecord":
        return cls(
            schema_version=1,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            component=component,
            client_id=client_id,
            round=round,
            step=step,
            exception_type=type(exception).__name__,
            message=_bounded(str(exception), MAX_FAILURE_MESSAGE_CHARS),
            traceback=_bounded(
                "".join(
                    traceback.format_exception(
                        type(exception), exception, exception.__traceback__
                    )
                ),
                MAX_FAILURE_TRACEBACK_CHARS,
            ),
        )

    def as_dict(self) -> dict:
        return asdict(self)


def publish_failure(failure_queue, record: FailureRecord) -> bool:
    """Best-effort bounded publish that never masks the original failure."""
    if failure_queue is None:
        return False
    try:
        failure_queue.put_nowait(record.as_dict())
    except (queue.Full, OSError, ValueError):
        return False
    return True
