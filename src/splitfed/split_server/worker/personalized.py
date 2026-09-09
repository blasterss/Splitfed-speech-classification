"""Personalized SplitServer child-process topology."""

import queue

import torch
import torch.multiprocessing as mp
import torch.nn as nn

from ....logger import logger
from ....schema import SplitServerConfig
from ....transport.base import Channel
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
from ....utils.training import _RoundStats, set_seed
from ..operations import _handle_eval_single, _handle_train_concat
from ..optimization import build_personalized_models
from ..protocol import _validate_message

logger = logger.getChild("SplitServer")


def _split_server_worker_personalized(
    config: SplitServerConfig,
    client_channels: dict[str, dict[str, Channel]],
    stop_event,
    result_queue: mp.Queue,
    failure_queue=None,
    resource_metrics_queue=None,
) -> None:
    """Serve each client with an isolated model and optimizer."""
    ignore_parent_interrupts()
    set_seed(config.seed)
    device = torch.device(config.model.device)
    tracker = ResourceTracker("split_server", device)
    client_ids = list(client_channels)
    models, optimizers = build_personalized_models(client_ids, config, device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(config.model.pos_weight, device=device)
    ).to(device)
    completed_steps = {client_id: set() for client_id in client_ids}
    stats = {client_id: _RoundStats() for client_id in client_ids}
    replay_guard = ReplayGuard()
    current_client_id = current_round = current_step = None
    logger.info(
        "Personalized SplitServer worker ready, serving clients: %s",
        client_ids,
    )
    try:
        while not stop_event.is_set():
            served_any = False
            for client_id in client_ids:
                message = client_channels[client_id]["uplink"].recv_nowait()
                if message is None:
                    continue
                current_client_id = client_id
                current_round = message.round
                current_step = message.step
                if not _validate_message(message, client_id):
                    raise ValueError(
                        f"Invalid split message from client {client_id}"
                    )
                replay_guard.accept(message.request_id)
                served_any = True
                correlation = (message.type, message.round, message.step)
                if correlation in completed_steps[client_id]:
                    raise ValueError(
                        f"Duplicate split message from client {client_id} "
                        f"for key {correlation}"
                    )
                completed_steps[client_id].add(correlation)
                if message.type == "train_step":
                    optimizers[client_id].zero_grad()
                    loss = _handle_train_concat(
                        {client_id: message},
                        models[client_id],
                        criterion,
                        device,
                        client_channels,
                    )
                    if loss is not None:
                        stats[client_id].update(loss)
                        optimizers[client_id].step()
                elif message.type == "eval_step":
                    _handle_eval_single(
                        message,
                        client_id,
                        models[client_id],
                        device,
                        client_channels,
                        [],
                        [],
                    )
                elif message.type == "round_end":
                    stats[client_id].log_and_reset(message.round)
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
        state = {
            client_id: {
                key: value.cpu()
                for key, value in models[client_id].state_dict().items()
            }
            for client_id in client_ids
        }
        try:
            result_queue.put_nowait(serialize_state_dict(state))
        except queue.Full:
            logger.warning(
                "Personalized SplitServer result queue was already full"
            )
