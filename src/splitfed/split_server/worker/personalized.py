"""Personalized SplitServer child-process topology."""

import queue

import torch
import torch.multiprocessing as mp
import torch.nn as nn

from ....logger import logger
from ....schema import SplitServerConfig
from ....transport.base import Channel
from ....transport.replay import ReplayGuard
from ....utils.failures import FailureRecord, publish_failure
from ....utils.persistence import serialize_state_dict
from ....utils.process import ignore_parent_interrupts
from ....utils.training import set_seed
from ....utils.training_stats import _RoundStats
from ..operations import _handle_eval_single, _handle_train_batch
from ..optimization import (
    build_personalized_models,
    step_accumulated_gradients,
)
from ..protocol import _validate_message

logger = logger.getChild("SplitServer")


def _split_server_worker_personalized(
    config: SplitServerConfig,
    client_channels: dict[str, dict[str, Channel]],
    stop_event,
    result_queue: mp.Queue,
    failure_queue=None,
) -> None:
    """Serve each client with an isolated model and optimizer."""
    ignore_parent_interrupts()
    set_seed(config.seed)
    device = torch.device(config.model.device)
    client_ids = list(client_channels)
    models, optimizers = build_personalized_models(client_ids, config, device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(config.model.pos_weight, device=device)
    ).to(device)
    accumulated = {client_id: 0 for client_id in client_ids}
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
                    loss = _handle_train_batch(
                        {client_id: message},
                        models[client_id],
                        criterion,
                        device,
                        client_channels,
                    )
                    if loss is not None:
                        stats[client_id].update(loss)
                        accumulated[client_id] += 1
                    if (
                        accumulated[client_id]
                        == config.model.gradient_accumulation_steps
                    ):
                        step_accumulated_gradients(
                            models[client_id].parameters(),
                            optimizers[client_id],
                            accumulated[client_id],
                        )
                        accumulated[client_id] = 0
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
                    if accumulated[client_id]:
                        step_accumulated_gradients(
                            models[client_id].parameters(),
                            optimizers[client_id],
                            accumulated[client_id],
                        )
                        accumulated[client_id] = 0
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
        for client_id in client_ids:
            if accumulated[client_id]:
                step_accumulated_gradients(
                    models[client_id].parameters(),
                    optimizers[client_id],
                    accumulated[client_id],
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
