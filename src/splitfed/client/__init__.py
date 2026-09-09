"""SplitFed client component."""

from .client import Client, _cpu_state_dict_snapshot
from .evaluation import _empty_metrics
from .protocol import (
    _extract_payload,
    _validate_global_update,
    _validate_round_ack,
)
from .worker import (
    _abort_barriers,
    _client_worker,
    _evaluate_at_barrier,
    _should_evaluate,
)

__all__ = [
    "Client",
    "_cpu_state_dict_snapshot",
    "_abort_barriers",
    "_client_worker",
    "_empty_metrics",
    "_evaluate_at_barrier",
    "_extract_payload",
    "_should_evaluate",
    "_validate_global_update",
    "_validate_round_ack",
]
