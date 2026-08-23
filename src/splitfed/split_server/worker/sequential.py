"""Classic SFLv2 shared-server worker with client-by-client updates."""

import queue

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

from ....logger import logger
from ....model.server_side_model import ServerSideModel
from ....schema import SplitServerConfig
from ....transport.base import Channel
from ....transport.replay import ReplayGuard
from ....utils.persistence import serialize_state_dict
from ....utils.runtime import (
    FailureRecord,
    ignore_parent_interrupts,
    publish_failure,
)
from ....utils.training import _RoundStats, set_seed
from ..operations import _handle_eval_single, _handle_train_concat
from ..protocol import _validate_message

logger = logger.getChild("SplitServer.Sequential")


def _split_server_worker_sequential(
    config: SplitServerConfig,
    client_channels: dict[str, dict[str, Channel]],
    stop_event,
    result_queue: mp.Queue,
    failure_queue=None,
) -> None:
    """Train one client to round completion before serving the next client."""
    ignore_parent_interrupts()
    set_seed(config.seed)
    device = torch.device(config.model.device)
    model = ServerSideModel(model_type="cnn_birnn").to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(config.model.pos_weight, device=device)
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.model.learning_rate)
    client_ids = list(client_channels)
    active_index = 0
    current_round = 1
    current_client_id = client_ids[0] if client_ids else None
    current_step = None
    stats = _RoundStats()
    replay_guard = ReplayGuard()
    deferred = {}

    logger.info("SFLv2 sequential client order: %s", client_ids)
    try:
        while not stop_event.is_set():
            if not client_ids:
                stop_event.wait(timeout=0.001)
                continue

            active_client_id = client_ids[active_index]
            message = deferred.pop(active_client_id, None)
            if message is None:
                for candidate_id in client_ids:
                    candidate = client_channels[candidate_id][
                        "uplink"
                    ].recv_nowait()
                    if candidate is None:
                        continue
                    if not _validate_message(candidate, candidate_id):
                        raise ValueError(
                            f"Invalid split message from client {candidate_id}"
                        )
                    replay_guard.accept(candidate.request_id)
                    if candidate.type == "eval_step":
                        _handle_eval_single(
                            candidate,
                            candidate_id,
                            model,
                            device,
                            client_channels,
                            [],
                            [],
                        )
                    elif candidate_id == active_client_id:
                        message = candidate
                    else:
                        deferred[candidate_id] = candidate
            if message is None:
                stop_event.wait(timeout=0.001)
                continue

            current_client_id = active_client_id
            current_step = message.step

            if message.type == "train_step":
                if message.round != current_round:
                    raise ValueError(
                        "Sequential split round mismatch: expected "
                        f"{current_round}, got {message.round}"
                    )
                optimizer.zero_grad()
                loss = _handle_train_concat(
                    {current_client_id: message},
                    model,
                    criterion,
                    device,
                    client_channels,
                )
                if loss is not None:
                    optimizer.step()
                    stats.update(loss)
            elif message.type == "round_end":
                if message.round != current_round:
                    raise ValueError(
                        "Sequential split round_end mismatch: expected "
                        f"{current_round}, got {message.round}"
                    )
                active_index += 1
                if active_index == len(client_ids):
                    stats.log_and_reset(current_round)
                    active_index = 0
                    current_round += 1
            else:
                raise ValueError(
                    f"Unexpected sequential split message {message.type.value}"
                )
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
        state = {key: value.cpu() for key, value in model.state_dict().items()}
        try:
            result_queue.put_nowait(serialize_state_dict(state))
        except queue.Full:
            logger.warning("SplitServer result queue was already full")
