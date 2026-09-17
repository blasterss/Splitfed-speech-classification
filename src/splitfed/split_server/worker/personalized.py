"""Personalized SplitServer child-process topology."""

import queue

import torch
import torch.multiprocessing as mp
import torch.nn as nn

from ....logger import logger
from ....schema import (
    FedServerConfig,
    SplitServerConfig,
    TrainingConfig,
    TrainingMode,
)
from ....transport.base import Channel, Message
from ....transport.replay import ReplayGuard
from ....utils.persistence import deserialize_state_dict, serialize_state_dict
from ....utils.runtime import (
    FailureRecord,
    ignore_parent_interrupts,
    publish_failure,
)
from ....utils.runtime.resource_metrics import (
    ResourceTracker,
    owned_resource_bytes,
    publish_resource_metric,
)
from ....utils.training import _RoundStats, set_seed
from ...fed_server.aggregation import aggregate_states
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
    training_mode: TrainingMode = TrainingMode.split,
    training_config: TrainingConfig | None = None,
    fed_server_config: FedServerConfig | None = None,
    coordination_queue=None,
    response_queue=None,
) -> None:
    """Serve each client with an isolated model and optimizer."""
    ignore_parent_interrupts()
    set_seed(config.seed)
    device = torch.device(config.model.device)
    client_ids = list(client_channels)
    tracker = ResourceTracker(
        "split_server",
        device,
        client_id=client_ids[0] if len(client_ids) == 1 else None,
    )
    models, optimizers = build_personalized_models(client_ids, config, device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(config.model.pos_weight, device=device)
    ).to(device)
    completed_steps = {client_id: set() for client_id in client_ids}
    stats = {client_id: _RoundStats() for client_id in client_ids}
    round_end_clients: dict[int, set[str]] = {}
    round_sizes: dict[int, dict[str, int]] = {}
    round_end_messages: dict[int, dict[str, Message]] = {}
    replay_guard = ReplayGuard()
    current_client_id = current_round = current_step = None
    logger.info(
        "Personalized SplitServer worker ready, serving clients: %s",
        client_ids,
    )
    if training_mode is TrainingMode.splitfed and (
        training_config is None or fed_server_config is None
    ):
        raise ValueError(
            "Personalized SplitFed requires training and FedServer configs"
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
                    if coordination_queue is not None:
                        dataset_size = message.payload.get("dataset_size")
                        if (
                            not isinstance(dataset_size, int)
                            or dataset_size <= 0
                        ):
                            raise ValueError(
                                "Invalid personalized split dataset_size"
                            )
                        assert training_config is not None
                        should_aggregate = (
                            message.round % training_config.fed_every == 0
                            and (
                                training_config.aggregate_final
                                or message.round < training_config.num_rounds
                            )
                        )
                        state = None
                        if should_aggregate:
                            state = serialize_state_dict(
                                {
                                    key: value.detach().cpu()
                                    for key, value in models[client_id]
                                    .state_dict()
                                    .items()
                                }
                            )
                        coordination_queue.put(
                            {
                                "client_id": client_id,
                                "round": message.round,
                                "dataset_size": dataset_size,
                                "state": state,
                            },
                            timeout=config.model.batch_timeout_sec,
                        )
                        response = response_queue.get(
                            timeout=config.model.batch_timeout_sec
                        )
                        if response["state"] is not None:
                            models[client_id].load_state_dict(
                                deserialize_state_dict(response["state"])
                            )
                        client_channels[client_id]["downlink"].send(
                            Message(
                                type="ack",
                                sender="split_server",
                                round=message.round,
                                step=message.step,
                                request_id=message.request_id,
                                payload={
                                    "server_aggregated": response["aggregated"]
                                },
                            )
                        )
                        continue
                    if training_mode is TrainingMode.splitfed:
                        dataset_size = message.payload.get("dataset_size")
                        if not isinstance(dataset_size, int) or (
                            dataset_size <= 0
                        ):
                            raise ValueError(
                                "Invalid personalized split dataset_size"
                            )
                        completed = round_end_clients.setdefault(
                            message.round, set()
                        )
                        if client_id in completed:
                            raise ValueError(
                                "Duplicate personalized split round_end from "
                                f"client {client_id} for round {message.round}"
                            )
                        completed.add(client_id)
                        round_sizes.setdefault(message.round, {})[
                            client_id
                        ] = dataset_size
                        round_end_messages.setdefault(message.round, {})[
                            client_id
                        ] = message
                    if training_mode is TrainingMode.splitfed and len(
                        round_end_clients[message.round]
                    ) == len(client_ids):
                        assert training_config is not None
                        assert fed_server_config is not None
                        should_aggregate = (
                            message.round % training_config.fed_every == 0
                            and (
                                training_config.aggregate_final
                                or message.round < training_config.num_rounds
                            )
                        )
                        if should_aggregate:
                            aggregated = aggregate_states(
                                [
                                    {
                                        key: value.detach().cpu()
                                        for key, value in models[cid]
                                        .state_dict()
                                        .items()
                                    }
                                    for cid in client_ids
                                ],
                                [
                                    round_sizes[message.round][cid]
                                    for cid in client_ids
                                ],
                                strategy=fed_server_config.strategy,
                                device=device,
                            )
                            for model in models.values():
                                model.load_state_dict(aggregated)
                        for cid in client_ids:
                            round_end = round_end_messages[message.round][cid]
                            client_channels[cid]["downlink"].send(
                                Message(
                                    type="ack",
                                    sender="split_server",
                                    round=message.round,
                                    step=round_end.step,
                                    request_id=round_end.request_id,
                                    payload={
                                        "server_aggregated": should_aggregate
                                    },
                                )
                            )
                        del round_end_clients[message.round]
                        del round_sizes[message.round]
                        del round_end_messages[message.round]
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
        metric = tracker.snapshot(round_idx=None, phase="train")
        ownership = [
            owned_resource_bytes(models[client_id], optimizers[client_id])
            for client_id in client_ids
        ]
        for field in ownership[0] if ownership else ():
            metric[field] = sum(item[field] for item in ownership)
        publish_resource_metric(resource_metrics_queue, metric)
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


def _split_server_worker_personalized_processes(
    config: SplitServerConfig,
    client_channels: dict[str, dict[str, Channel]],
    stop_event,
    result_queue: mp.Queue,
    failure_queue=None,
    resource_metrics_queue=None,
    training_mode: TrainingMode = TrainingMode.split,
    training_config: TrainingConfig | None = None,
    fed_server_config: FedServerConfig | None = None,
) -> None:
    """Coordinate one isolated personalized model process per client."""
    ignore_parent_interrupts()
    context = mp.get_context("spawn")
    client_ids = list(client_channels)
    coordination_queue = context.Queue()
    response_queues = {cid: context.Queue(maxsize=1) for cid in client_ids}
    state_queues = {cid: context.Queue(maxsize=1) for cid in client_ids}
    processes = []
    final_states = {}
    pending = {}
    try:
        for client_id in client_ids:
            coordinates_rounds = training_mode is TrainingMode.splitfed
            process = context.Process(
                target=_split_server_worker_personalized,
                args=(
                    config,
                    {client_id: client_channels[client_id]},
                    stop_event,
                    state_queues[client_id],
                    failure_queue,
                    resource_metrics_queue,
                    training_mode,
                    training_config,
                    fed_server_config,
                    coordination_queue if coordinates_rounds else None,
                    response_queues[client_id] if coordinates_rounds else None,
                ),
                name=f"SplitServer-{client_id}",
            )
            process.start()
            processes.append(process)
            logger.info(
                "Personalized SplitServer client %s process started (pid=%d)",
                client_id,
                process.pid,
            )

        while any(process.is_alive() for process in processes):
            for client_id, state_queue in state_queues.items():
                if client_id in final_states:
                    continue
                try:
                    payload = state_queue.get_nowait()
                except queue.Empty:
                    continue
                final_states.update(deserialize_state_dict(payload))
            try:
                report = coordination_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            reports = pending.setdefault(report["round"], {})
            reports[report["client_id"]] = report
            if len(reports) != len(client_ids):
                continue
            assert training_config is not None
            assert fed_server_config is not None
            should_aggregate = report[
                "round"
            ] % training_config.fed_every == 0 and (
                training_config.aggregate_final
                or report["round"] < training_config.num_rounds
            )
            state = None
            if should_aggregate:
                aggregated = aggregate_states(
                    [
                        deserialize_state_dict(reports[cid]["state"])
                        for cid in client_ids
                    ],
                    [reports[cid]["dataset_size"] for cid in client_ids],
                    strategy=fed_server_config.strategy,
                    device=config.model.device,
                )
                state = serialize_state_dict(
                    {key: value.cpu() for key, value in aggregated.items()}
                )
            for client_id in client_ids:
                response_queues[client_id].put(
                    {"aggregated": should_aggregate, "state": state}
                )
            del pending[report["round"]]

        for process in processes:
            process.join(timeout=5)
        failed = [p for p in processes if p.exitcode != 0]
        if failed:
            raise RuntimeError(
                "Personalized SplitServer model process failure: "
                + ", ".join(f"{p.name}={p.exitcode}" for p in failed)
            )
        for client_id, state_queue in state_queues.items():
            if client_id not in final_states:
                final_states.update(
                    deserialize_state_dict(state_queue.get(timeout=5))
                )
        result_queue.put(serialize_state_dict(final_states), timeout=5)
    except BaseException as exc:
        stop_event.set()
        publish_failure(
            failure_queue,
            FailureRecord.from_exception(
                component="split_server_coordinator", exception=exc
            ),
        )
        raise
    finally:
        stop_event.set()
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
