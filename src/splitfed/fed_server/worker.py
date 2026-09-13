"""Child-process event loop for the federated server component."""

import queue
import time

import torch.multiprocessing as mp

from ...logger import logger
from ...schema import AggregationStrategy, FedServerConfig
from ...transport.base import Channel, Message
from ...transport.replay import ReplayGuard
from ...utils.persistence import serialize_state_dict
from ...utils.runtime import (
    FailureRecord,
    ignore_parent_interrupts,
    publish_failure,
)
from ...utils.runtime.resource_metrics import (
    ResourceTracker,
    publish_resource_metric,
)
from ...utils.training import set_seed
from .aggregation import aggregate_states
from .protocol import quorum_decision, state_schema, validate_client_update


def _fed_server_worker(
    config: FedServerConfig,
    client_channels: dict[str, dict[str, Channel]],
    num_clients: int,
    stop_event,
    result_queue: mp.Queue,
    failure_queue=None,
    resource_metrics_queue=None,
) -> None:
    """Collect, validate, aggregate and broadcast federated updates."""
    ignore_parent_interrupts()
    set_seed(config.seed)
    tracker = ResourceTracker("fed_server", config.device)
    client_ids = list(client_channels)
    logger.info("FedServer worker ready, serving clients: %s", client_ids)

    latest_params: dict | None = None
    latest_round = 1
    updates: dict[str, dict] = {}
    sizes: dict[str, int] = {}
    aggregation_weights: dict[str, int] = {}
    request_ids: dict[str, str] = {}
    active_round: int | None = None
    last_completed_round = 0
    expected_schema = None
    round_started_at: float | None = None
    replay_guard = ReplayGuard()
    current_client_id = None
    current_round = None
    current_step = None

    try:
        while not stop_event.is_set():
            served_any = False
            for client_id in client_ids:
                if client_id in updates:
                    continue

                msg = client_channels[client_id]["uplink"].recv_nowait()
                if msg is None:
                    continue
                current_client_id = client_id
                current_round = msg.round
                current_step = msg.step
                served_any = True
                replay_guard.accept(msg.request_id)

                if msg.round <= last_completed_round:
                    validate_client_update(
                        msg,
                        expected_client_id=client_id,
                        expected_round=msg.round,
                        expected_schema=None,
                        require_aggregation_weight=(
                            config.strategy
                            is AggregationStrategy.mergesfl_batch_weighted_v1
                        ),
                    )
                    if (
                        msg.round == last_completed_round
                        and latest_params is not None
                    ):
                        client_channels[client_id]["downlink"].send(
                            Message(
                                type="global_update",
                                sender="fed_server",
                                round=msg.round,
                                step=1,
                                request_id=msg.request_id,
                                payload=latest_params,
                            )
                        )
                        logger.info(
                            "Sent correlated catch-up model to %s for "
                            "completed round %d",
                            client_id,
                            msg.round,
                        )
                        continue
                    logger.info(
                        "Discarding late update from %s for completed "
                        "round %d",
                        client_id,
                        msg.round,
                    )
                    continue

                if active_round is None:
                    active_round = msg.round
                    round_started_at = time.monotonic()

                state_dict, dataset_size, aggregation_weight = (
                    validate_client_update(
                    msg,
                    expected_client_id=client_id,
                    expected_round=active_round,
                    expected_schema=expected_schema,
                    require_aggregation_weight=(
                        config.strategy
                        is AggregationStrategy.mergesfl_batch_weighted_v1
                    ),
                )
                )
                if expected_schema is None:
                    expected_schema = state_schema(state_dict)
                updates[client_id] = state_dict
                sizes[client_id] = dataset_size
                if aggregation_weight is not None:
                    aggregation_weights[client_id] = aggregation_weight
                request_ids[client_id] = msg.request_id
                latest_round = active_round
                logger.info(
                    "Received update from client %s (round=%d, samples=%d)",
                    client_id,
                    msg.round,
                    dataset_size,
                )

            if active_round is None:
                decision = "wait"
            else:
                assert round_started_at is not None
                decision = quorum_decision(
                    update_count=len(updates),
                    client_count=num_clients,
                    min_clients=config.min_clients,
                    elapsed=time.monotonic() - round_started_at,
                    timeout=config.quorum_timeout_sec,
                )

            if decision == "fail":
                raise RuntimeError(
                    f"Federated quorum timeout for round {active_round}: "
                    f"received {len(updates)}/{config.min_clients} required"
                )
            if decision == "aggregate":
                participant_ids = [cid for cid in client_ids if cid in updates]
                try:
                    latest_params = aggregate_states(
                        [updates[cid] for cid in participant_ids],
                        [
                            aggregation_weights[cid]
                            if config.strategy
                            is AggregationStrategy.mergesfl_batch_weighted_v1
                            else sizes[cid]
                            for cid in participant_ids
                        ],
                        config.strategy,
                        device=config.device,
                    )
                except Exception as exc:
                    logger.error(
                        "Aggregation failed (round=%d): %s",
                        latest_round,
                        exc,
                        exc_info=True,
                    )
                    raise

                logger.info(
                    "Aggregated %d client updates (round=%d)",
                    len(updates),
                    latest_round,
                )
                for client_id in participant_ids:
                    client_channels[client_id]["downlink"].send(
                        Message(
                            type="global_update",
                            sender="fed_server",
                            round=latest_round,
                            step=1,
                            request_id=request_ids[client_id],
                            payload=latest_params,
                        )
                    )
                updates.clear()
                sizes.clear()
                aggregation_weights.clear()
                request_ids.clear()
                last_completed_round = active_round
                active_round = None
                expected_schema = None
                round_started_at = None

            if not served_any:
                stop_event.wait(timeout=0.001)
    except BaseException as exc:
        publish_failure(
            failure_queue,
            FailureRecord.from_exception(
                component="fed_server",
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
        if latest_params is None:
            try:
                result_queue.put_nowait(None)
            except queue.Full:
                pass
            logger.warning(
                "FedServer worker exiting — no aggregation completed"
            )
        else:
            try:
                state_dict = {
                    key: value.cpu() for key, value in latest_params.items()
                }
                result_queue.put_nowait(serialize_state_dict(state_dict))
                logger.info(
                    "FedServer worker exiting — final state_dict saved"
                )
            except queue.Full:
                logger.warning(
                    "Result queue full — final state_dict not stored"
                )
