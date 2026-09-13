"""Client child-process lifecycle and round orchestration."""

import queue
import time

from ...logger import logger
from ...schema import (
    ClientConfig,
    ServerModelScope,
    TrainingConfig,
    TrainingMode,
    WorkloadPolicy,
)
from ...transport.base import Channel, ChannelCancelled
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
from ..load_controller import RoundPlan, WorkerState, WorkerTelemetry

logger = logger.getChild("Client")


def _round_workload_counts(
    *,
    loader_batches: int,
    dataset_samples: int,
    batch_size: int,
    local_steps: int,
    policy: WorkloadPolicy,
) -> tuple[int, int]:
    """Return batches and effective samples processed in one round."""
    if policy is WorkloadPolicy.full_epoch_v1:
        return loader_batches, dataset_samples
    if policy is WorkloadPolicy.fixed_steps_v1:
        if loader_batches <= 0:
            return 0, 0
        full_cycles, remaining_batches = divmod(local_steps, loader_batches)
        samples = full_cycles * dataset_samples + min(
            dataset_samples,
            remaining_batches * batch_size,
        )
        return local_steps, samples

    batches = min(loader_batches, local_steps)
    return batches, min(dataset_samples, batches * batch_size)


def _client_worker(
    cfg: ClientConfig,
    training_cfg: TrainingConfig,
    split_uplink: Channel,
    split_downlink: Channel,
    fed_uplink: Channel,
    fed_downlink: Channel,
    stop_event,
    ready_barrier,
    eval_barrier,
    metrics_path=None,
    dataset_report_queue=None,
    failure_queue=None,
    resource_metrics_queue=None,
    split_server_scope: ServerModelScope | None = None,
    round_plan_queue=None,
    telemetry_queue=None,
) -> None:
    """Run a persistent client process across configured training rounds."""
    ignore_parent_interrupts()

    current_round = None
    try:
        initialization_seed = (
            training_cfg.seed
            if training_cfg.mode
            in (TrainingMode.federated, TrainingMode.splitfed)
            else cfg.runtime.seed + int(cfg.client_id)
        )
        set_seed(initialization_seed)

        # Resolve through the package at runtime so the stable public Client
        # seam remains patchable by lifecycle tests and downstream callers.
        from . import Client

        client = Client(
            cfg=cfg,
            split_uplink_channel=split_uplink,
            split_downlink_channel=split_downlink,
            fed_uplink_channel=fed_uplink,
            fed_downlink_channel=fed_downlink,
            mode=training_cfg.mode,
            metrics_path=metrics_path,
            split_server_scope=split_server_scope,
        )
        if dataset_report_queue is not None:
            dataset_report_queue.put(
                {"client_id": cfg.client_id, **client.dataset_manifest},
                timeout=5,
            )

        logger.info(
            "Client %s: dataset loaded — waiting at ready_barrier.",
            cfg.client_id,
        )
        ready_barrier.wait(timeout=training_cfg.barrier_timeout_sec)

        logger.info(
            "Client %s: ready_barrier passed — starting training.",
            cfg.client_id,
        )

        last_round = 0
        for round_idx in range(1, training_cfg.num_rounds + 1):
            if stop_event.is_set():
                return

            last_round = round_idx
            current_round = round_idx
            plan = _receive_round_plan(
                round_plan_queue,
                round_idx=round_idx,
                timeout=training_cfg.barrier_timeout_sec,
            )
            selected = plan is None or cfg.client_id in plan.cohort
            if plan is not None and selected:
                client.configure_round(
                    round_idx=round_idx,
                    batch_size=plan.batch_size_by_client[cfg.client_id],
                    local_steps=plan.local_steps,
                )
            tracker = None
            if resource_metrics_queue is not None:
                tracker = ResourceTracker(
                    "client", client.device, client_id=client.client_id
                )
            if selected:
                client.train_one_round(round_idx)
            else:
                client.skip_round(round_idx)
            if tracker is not None:
                batches, samples = _round_workload_counts(
                    loader_batches=len(client.train_loader),
                    dataset_samples=len(client.dataset.train_dataset),
                    batch_size=cfg.runtime.batch_size,
                    local_steps=cfg.runtime.local_steps,
                    policy=cfg.runtime.workload_policy,
                )
                publish_resource_metric(
                    resource_metrics_queue,
                    tracker.snapshot(
                        round_idx=round_idx,
                        phase="train",
                        samples=samples,
                        batches=batches,
                    ),
                )

            should_aggregate = (
                training_cfg.mode
                in (TrainingMode.federated, TrainingMode.splitfed)
                and round_idx % training_cfg.fed_every == 0
                and (
                    getattr(training_cfg, "aggregate_final", True)
                    or round_idx < training_cfg.num_rounds
                )
            )
            should_evaluate = _should_evaluate(
                round_idx,
                training_cfg.num_rounds,
                training_cfg.eval_every,
            )

            if should_aggregate and selected:
                if plan is None:
                    client.federative_aggregate(round_idx)
                else:
                    aggregation_weight = (
                        plan.batch_size_by_client[cfg.client_id]
                        * plan.local_steps
                    )
                    client.federative_aggregate(
                        round_idx, aggregation_weight=aggregation_weight
                    )
            if plan is not None and telemetry_queue is not None and selected:
                samples = (
                    plan.batch_size_by_client[cfg.client_id]
                    * plan.local_steps
                )
                telemetry_queue.put(
                    _round_telemetry(client, plan, samples), timeout=5
                )
            if should_evaluate:
                if tracker is not None:
                    tracker.reset()
                _evaluate_at_barrier(
                    client,
                    round_idx,
                    eval_barrier,
                    training_cfg.barrier_timeout_sec,
                )
                if tracker is not None:
                    publish_resource_metric(
                        resource_metrics_queue,
                        tracker.snapshot(
                            round_idx=round_idx,
                            phase="evaluation",
                            samples=len(client.dataset.test_dataset),
                            batches=len(client.test_loader),
                        ),
                    )

        if last_round == 0:
            _evaluate_at_barrier(
                client,
                0,
                eval_barrier,
                training_cfg.barrier_timeout_sec,
            )

    except ChannelCancelled:
        if stop_event.is_set():
            logger.info("Client %s cancelled", cfg.client_id)
            return
        raise
    except BaseException as exc:
        publish_failure(
            failure_queue,
            FailureRecord.from_exception(
                component="client",
                client_id=cfg.client_id,
                round=current_round,
                exception=exc,
            ),
        )
        stop_event.set()
        _abort_barriers((ready_barrier, eval_barrier))
        logger.error(
            "Client %s crashed: %s", cfg.client_id, exc, exc_info=True
        )
        raise


def _should_evaluate(round_idx: int, num_rounds: int, eval_every: int) -> bool:
    return round_idx % eval_every == 0 or round_idx == num_rounds


def _receive_round_plan(plan_queue, *, round_idx: int, timeout: float):
    if plan_queue is None:
        return None
    try:
        plan = plan_queue.get(timeout=timeout)
    except queue.Empty as exc:
        raise TimeoutError(f"No MergeSFL plan for round {round_idx}") from exc
    if not isinstance(plan, RoundPlan) or plan.round != round_idx:
        raise ValueError(f"Invalid MergeSFL plan for round {round_idx}")
    if plan.deadline_at <= time.time():
        raise TimeoutError(f"MergeSFL plan for round {round_idx} expired")
    return plan


def _round_telemetry(client, plan: RoundPlan, samples: int) -> WorkerTelemetry:
    if samples <= 0:
        raise ValueError("telemetry requires a positive sample count")
    state = WorkerState(
        compute_seconds_per_sample=max(
            client.last_compute_seconds / samples, 1e-12
        ),
        transfer_seconds_per_sample=max(
            client.last_transfer_seconds / samples, 1e-12
        ),
    )
    return WorkerTelemetry(
        client_id=client.client_id,
        state=state,
        observed_at=time.time(),
    )


def _evaluate_at_barrier(
    client,
    round_idx: int,
    eval_barrier,
    timeout: float,
) -> None:
    logger.info(
        "Client %s: waiting to evaluate round %d", client.client_id, round_idx
    )
    eval_barrier.wait(timeout=timeout)
    metrics = client.evaluate(round=round_idx)
    logger.info(
        "Client %s round %d metrics: %s",
        client.client_id,
        round_idx,
        metrics,
    )
    eval_barrier.wait(timeout=timeout)


def _abort_barriers(barriers: tuple[object, ...]) -> None:
    for barrier in barriers:
        try:
            barrier.abort()
        except Exception as exc:
            logger.debug("Client could not abort lifecycle barrier: %s", exc)
