"""Shared-model SplitServer child-process topology."""

import queue
import time
from collections import defaultdict

import torch
import torch.multiprocessing as mp
import torch.nn as nn

from ....logger import logger
from ....model.server_side_model import ServerSideModel
from ....schema import SplitServerConfig
from ....transport.base import Channel, Message
from ....transport.replay import ReplayGuard
from ....utils.persistence import serialize_state_dict
from ....utils.runtime import (
    FailureRecord,
    ignore_parent_interrupts,
    publish_failure,
)
from ....utils.runtime.resource_metrics import (
    ResourceTracker,
    publish_resource_metric,
)
from ....utils.training import _RoundStats, build_optimizer, set_seed
from ..operations import (
    _handle_eval_single,
    _handle_train_concat,
    _handle_train_mergesfl,
)
from ..protocol import (
    _batch_is_ready,
    _evict_stale_batches,
    _store_pending_batch,
    _validate_message,
)

logger = logger.getChild("SplitServer")


def _split_server_worker_concat(
    config: SplitServerConfig,
    client_channels: dict[str, dict[str, Channel]],
    stop_event,
    result_queue: mp.Queue,
    failure_queue=None,
    resource_metrics_queue=None,
) -> None:
    ignore_parent_interrupts()
    set_seed(config.seed)  # Ensure deterministic behavior in server process,

    device = torch.device(config.model.device)
    tracker = ResourceTracker("split_server", device)
    model = ServerSideModel(model_type="cnn_birnn").to(device)
    pos_weight = torch.tensor(config.model.pos_weight, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
    optimizer = build_optimizer(model.parameters(), config.model)

    client_ids = list(client_channels.keys())
    pending_batches: dict[tuple, dict[str, Message]] = defaultdict(dict)
    pending_timestamps: dict[tuple, float] = {}
    completed_rounds = defaultdict(set)

    logger.info("SplitServer worker ready, serving clients: %s", client_ids)

    current_round = 1
    stats = _RoundStats()
    replay_guard = ReplayGuard()
    current_client_id = None
    current_step = None

    all_eval_probs = []
    all_eval_labels = []
    train_handler = (
        _handle_train_mergesfl
        if config.training_strategy.value == "mergesfl_v1"
        else _handle_train_concat
    )
    try:
        while not stop_event.is_set():
            served_any = False

            for client_id in client_ids:
                uplink: Channel = client_channels[client_id]["uplink"]

                msg = uplink.recv_nowait()
                if msg is None:
                    continue
                current_client_id = client_id
                current_round = msg.round
                current_step = msg.step

                if not _validate_message(msg, client_id):
                    raise ValueError(
                        f"Invalid split message from client {client_id}"
                    )

                replay_guard.accept(msg.request_id)

                served_any = True

                if msg.type == "eval_step":
                    _handle_eval_single(
                        msg,
                        client_id,
                        model,
                        device,
                        client_channels,
                        all_eval_probs,
                        all_eval_labels,
                    )
                elif msg.type == "round_end":
                    if client_id in completed_rounds[msg.round]:
                        raise ValueError(
                            f"Duplicate round_end from client {client_id} "
                            f"for round {msg.round}"
                        )
                    completed_rounds[msg.round].add(client_id)
                elif msg.type == "train_step":
                    if client_id in completed_rounds[msg.round]:
                        raise ValueError(
                            "Train step after round_end from client "
                            f"{client_id}"
                        )
                    key = (msg.round, msg.step)

                    if key not in pending_timestamps:
                        pending_timestamps[key] = time.monotonic()

                    _store_pending_batch(pending_batches, msg, client_id)

                    logger.info(
                        "Received '%s' from %s (round=%d step=%d) [%d/%d]",
                        msg.type,
                        client_id,
                        msg.round,
                        msg.step,
                        len(pending_batches[key]),
                        len(client_ids),
                    )

                else:
                    logger.warning(
                        "SplitServer: unknown msg type '%s' from %s - "
                        "discarding.",
                        msg.type,
                        client_id,
                    )

                ready_keys = [
                    key
                    for key, batch in pending_batches.items()
                    if _batch_is_ready(
                        batch,
                        set(client_ids),
                        completed_rounds[key[0]],
                    )
                ]
                for key in ready_keys:
                    batch_msgs = pending_batches.pop(key)
                    pending_timestamps.pop(key, None)

                    batch_round = next(iter(batch_msgs.values())).round
                    if batch_round != current_round:
                        stats.log_and_reset(current_round)
                        current_round = batch_round

                    optimizer.zero_grad()
                    batch_loss = train_handler(
                        batch_msgs,
                        model,
                        criterion,
                        device,
                        client_channels,
                    )
                    if batch_loss is not None:
                        stats.update(batch_loss)
                        optimizer.step()

                completed_round = msg.round
                round_is_complete = completed_rounds[completed_round] == set(
                    client_ids
                )
                round_has_pending = any(
                    key[0] == completed_round for key in pending_batches
                )
                if round_is_complete and not round_has_pending:
                    completed_rounds.pop(completed_round, None)

            _evict_stale_batches(
                pending_batches,
                pending_timestamps,
                client_channels,
                config.model.batch_timeout_sec,
            )

            if not served_any:
                stop_event.wait(timeout=0.001)

    except BaseException as exc:
        publish_failure(
            failure_queue,
            FailureRecord.from_exception(
                component="split_server",
                client_id=current_client_id,
                round=current_round,
                step=current_step,
                exception=exc,
            ),
        )
        stop_event.set()
        raise

    finally:
        publish_resource_metric(
            resource_metrics_queue,
            tracker.snapshot(round_idx=None, phase="lifetime"),
        )
        stats.log_and_reset(current_round)
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        try:
            result_queue.put_nowait(serialize_state_dict(state_dict))
            logger.info(
                "SplitServer worker exiting — state_dict pushed to "
                "result queue"
            )
        except queue.Full:
            logger.warning(
                "SplitServer result queue was already full — state_dict "
                "NOT pushed."
            )
