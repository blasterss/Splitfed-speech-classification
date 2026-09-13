"""Plan-aware federated aggregation for MergeSFL Algorithm 1."""

import queue
import time

from ...schema import AggregationStrategy, FedServerConfig
from ...transport.base import Message
from ...transport.replay import ReplayGuard
from ..load_controller import RoundPlan
from .aggregation import aggregate_states
from .protocol import (
    state_schema,
    validate_client_update,
    validate_model_sync_request,
)


def run_planned_federated_rounds(
    config: FedServerConfig,
    client_channels,
    stop_event,
    round_plan_queue,
):
    """Execute controller plans until cancellation and return latest state."""
    latest_params = None
    replay_guard = ReplayGuard()
    expected_round = 1
    while not stop_event.is_set():
        plan = _receive_plan(
            round_plan_queue,
            expected_round=expected_round,
            stop_event=stop_event,
        )
        if plan is None:
            break
        latest_params = _execute_plan(
            config,
            client_channels,
            stop_event,
            plan,
            replay_guard,
        )
        expected_round += 1
    return latest_params


def _receive_plan(round_plan_queue, *, expected_round: int, stop_event):
    while not stop_event.is_set():
        try:
            plan = round_plan_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if not isinstance(plan, RoundPlan) or plan.round != expected_round:
            raise ValueError(
                f"Invalid federated RoundPlan for round {expected_round}"
            )
        if plan.deadline_at <= time.time():
            raise TimeoutError(f"Federated RoundPlan {expected_round} expired")
        return plan
    return None


def _execute_plan(config, client_channels, stop_event, plan, replay_guard):
    if config.strategy is not AggregationStrategy.mergesfl_batch_weighted_v1:
        raise ValueError("planned federation requires MergeSFL aggregation")
    all_clients = set(client_channels)
    cohort = set(plan.cohort)
    if not cohort <= all_clients:
        raise ValueError("federated RoundPlan contains unknown clients")
    if plan.required_quorum != len(cohort):
        raise ValueError("MergeSFL requires the complete planned cohort")

    updates = {}
    weights = {}
    update_requests = {}
    sync_requests = {}
    synced = set()
    expected_schema = None
    latest_params = None

    while not stop_event.is_set():
        if time.time() >= plan.deadline_at:
            missing_updates = sorted(cohort - set(updates), key=str)
            missing_sync = sorted(
                all_clients - cohort - set(sync_requests), key=str
            )
            raise TimeoutError(
                "MergeSFL federated round deadline exceeded; "
                f"updates={missing_updates}, sync={missing_sync}"
            )

        served_any = False
        for client_id in sorted(all_clients, key=str):
            if client_id in updates or client_id in sync_requests:
                continue
            message = client_channels[client_id]["uplink"].recv_nowait()
            if message is None:
                continue
            served_any = True
            replay_guard.accept(message.request_id)
            if client_id in cohort:
                state, _dataset_size, weight = validate_client_update(
                    message,
                    expected_client_id=client_id,
                    expected_round=plan.round,
                    expected_schema=expected_schema,
                    require_aggregation_weight=True,
                )
                expected_weight = (
                    plan.batch_size_by_client[client_id] * plan.local_steps
                )
                if weight != expected_weight:
                    raise ValueError(
                        "client update weight differs from RoundPlan"
                    )
                if expected_schema is None:
                    expected_schema = state_schema(state)
                updates[client_id] = state
                weights[client_id] = weight
                update_requests[client_id] = message.request_id
            else:
                validate_model_sync_request(
                    message,
                    expected_client_id=client_id,
                    expected_round=plan.round,
                )
                sync_requests[client_id] = message.request_id

        if latest_params is None and set(updates) == cohort:
            participant_ids = sorted(cohort, key=str)
            latest_params = aggregate_states(
                [updates[client_id] for client_id in participant_ids],
                [weights[client_id] for client_id in participant_ids],
                AggregationStrategy.mergesfl_batch_weighted_v1,
                device=config.device,
            )
            for client_id in participant_ids:
                _send_global(
                    client_channels,
                    client_id,
                    plan.round,
                    update_requests[client_id],
                    latest_params,
                )

        if latest_params is not None:
            for client_id, request_id in tuple(sync_requests.items()):
                if client_id in synced:
                    continue
                _send_global(
                    client_channels,
                    client_id,
                    plan.round,
                    request_id,
                    latest_params,
                )
                synced.add(client_id)
            if synced == all_clients - cohort:
                return latest_params

        if not served_any:
            stop_event.wait(timeout=0.001)

    return latest_params


def _send_global(channels, client_id, round_idx, request_id, state_dict):
    channels[client_id]["downlink"].send(
        Message(
            type="global_update",
            sender="fed_server",
            round=round_idx,
            step=1,
            request_id=request_id,
            payload=state_dict,
        )
    )
