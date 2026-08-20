"""Split-learning message and pending-batch coordination contracts."""

import time

import torch

from ...logger import logger
from ...transport.base import Channel, Message


def _validate_message(msg: Message, client_id: str) -> bool:
    try:
        msg.validate_for_receive()
    except (ValueError, TimeoutError) as exc:
        logger.warning("SplitServer: invalid message deadline: %s", exc)
        return False
    if msg.sender != client_id:
        logger.warning(
            "SplitServer: sender mismatch on client %s channel (got %s)",
            client_id,
            msg.sender,
        )
        return False
    if msg.type not in ("train_step", "eval_step", "round_end"):
        logger.warning(
            "SplitServer: unknown message type '%s' from client %s — "
            "discarding.",
            msg.type,
            client_id,
        )
        return False
    if msg.round <= 0 or msg.step <= 0:
        logger.warning(
            "SplitServer: invalid correlation from client %s "
            "(round=%d step=%d)",
            client_id,
            msg.round,
            msg.step,
        )
        return False
    if not isinstance(msg.payload, dict):
        logger.warning(
            "SplitServer: payload is not a dict (client %s, type %s) — "
            "discarding.",
            client_id,
            msg.type,
        )
        return False
    if msg.type == "round_end":
        return True
    if "activations" not in msg.payload or "labels" not in msg.payload:
        logger.warning(
            "SplitServer: missing activation/label payload (client %s)",
            client_id,
        )
        return False
    activations = msg.payload["activations"]
    labels = msg.payload["labels"]
    if not isinstance(activations, torch.Tensor) or activations.dim() < 1:
        logger.warning(
            "SplitServer: invalid activations tensor (client %s)", client_id
        )
        return False
    if (
        not isinstance(labels, torch.Tensor)
        or labels.dim() < 1
        or labels.shape[0] != activations.shape[0]
    ):
        logger.warning(
            "SplitServer: invalid labels for activation batch (client %s)",
            client_id,
        )
        return False
    return True


def _batch_is_ready(
    batch: dict[str, Message], client_ids: set, completed_clients: set
) -> bool:
    return client_ids - set(batch) <= completed_clients


def _store_pending_batch(
    pending_batches: dict[tuple, dict[str, Message]],
    message: Message,
    client_id: str,
) -> None:
    key = (message.round, message.step)
    if client_id in pending_batches.setdefault(key, {}):
        raise ValueError(
            f"Duplicate split step from client {client_id} for key {key}"
        )
    pending_batches[key][client_id] = message


def _evict_stale_batches(
    pending_batches: dict[tuple, dict[str, Message]],
    pending_timestamps: dict[tuple, float],
    client_channels: dict[str, dict[str, Channel]],
    timeout_seconds: float,
) -> None:
    now = time.monotonic()
    stale_keys = [
        key
        for key, timestamp in pending_timestamps.items()
        if now - timestamp > timeout_seconds
    ]
    for key in stale_keys:
        batch = pending_batches.pop(key, {})
        pending_timestamps.pop(key, None)
        for client_id, message in batch.items():
            client_channels[client_id]["downlink"].send(
                Message(
                    type="error",
                    sender="split_server",
                    round=message.round,
                    step=message.step,
                    request_id=message.request_id,
                    payload={"reason": "split_batch_timeout"},
                )
            )
        logger.warning(
            "SplitServer: evicting stale batch key=%s "
            "(received from %d client(s): %s, timeout=%gs).",
            key,
            len(batch),
            list(batch),
            timeout_seconds,
        )
