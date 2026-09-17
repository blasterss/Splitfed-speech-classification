import queue
import time

import torch.multiprocessing as mp

from ...logger import get_logger
from ...schema import ConfigSchema, TrainingMode
from ...utils.config import save_yaml
from ...utils.persistence import ArtifactPaths
from ...utils.runtime.resource_metrics import (
    RESOURCE_MEASUREMENT_POLICY,
    RESOURCE_METRICS_SCHEMA_VERSION,
    ResourceTracker,
)
from ..centralized import CentralizedTrainer
from ..client import _client_worker
from ..common.lifecycle import (
    _cancel_training,
    _shutdown_processes,
    _wait_for_training_processes,
)
from ..fed_server import FedServer
from ..load_controller import (
    MergeSFLPlanner,
    bootstrap_telemetry,
    collect_selected_telemetry,
    profiles_from_manifests,
)
from ..split_server import SplitServer
from .reporting import (
    capture_first_failure,
    close_process_queue,
    collect_dataset_reports,
    drain_dataset_reports,
)
from .topology import (
    FED_DOWNLINK,
    FED_UPLINK,
    SPLIT_DOWNLINK,
    SPLIT_UPLINK,
    build_channels,
    build_servers,
)

logger = get_logger(__name__)


class TrainingController:
    """
    Orchestrates the full Split Learning / Federated Learning pipeline.

    Responsibilities:
        - Channel initialization per client
        - Server lifecycle management (split + federated)
        - Client process spawning
        - Synchronization via barriers
        - Training supervision and teardown
    """

    SPLIT_UPLINK = SPLIT_UPLINK
    SPLIT_DOWNLINK = SPLIT_DOWNLINK
    FED_UPLINK = FED_UPLINK
    FED_DOWNLINK = FED_DOWNLINK
    PROCESS_POLL_TIMEOUT = 0.5
    PROCESS_SHUTDOWN_TIMEOUT = 10

    def __init__(self, config: ConfigSchema):
        self.cfg = config
        self.client_cfgs = config.clients
        self._mp_context = mp.get_context("spawn")

        self.split_server: SplitServer | None = None
        self.fed_server: FedServer | None = None
        self.centralized_trainer: CentralizedTrainer | None = None

        self.channels: dict[str, dict] = {}
        self._client_processes: list[mp.Process] = []
        self._manager: mp.managers.SyncManager | None = None
        self._stop_event = None
        self._stop_events: dict[str, mp.Event] = {}
        self._dataset_report_queue = None
        self._failure_queue = None
        self._resource_metrics_queue = None
        self._telemetry_queue = None
        self._client_round_plan_queues: dict[int, object] = {}
        self._split_round_plan_queue = None
        self._fed_round_plan_queue = None
        self.dataset_manifests: dict[int, dict] = {}
        self.resource_metrics: list[dict] = []
        self.first_failure: dict | None = None
        self.run_started_at: float | None = None
        self.mergesfl_round_plans: list[dict] = []

    def setup(self) -> None:
        """
        Initializes multiprocessing manager, channels, and servers.
        """
        logger.info("=== TRAINING CONTROLLER SETUP ===")

        self._manager = self._mp_context.Manager()
        self._stop_event = self._manager.Event()
        self._dataset_report_queue = self._mp_context.Queue()
        self._failure_queue = self._mp_context.Queue(
            maxsize=len(self.client_cfgs) + 3
        )
        self._resource_metrics_queue = self._mp_context.Queue()
        if self.cfg.load_controller is not None:
            self._telemetry_queue = self._mp_context.Queue()
            self._client_round_plan_queues = {
                client.client_id: self._mp_context.Queue(maxsize=1)
                for client in self.client_cfgs
            }
            self._split_round_plan_queue = self._mp_context.Queue(maxsize=1)
            self._fed_round_plan_queue = self._mp_context.Queue(maxsize=1)

        logger.info("Initialising channels...")
        self._init_channels()

        logger.info("Initialising servers...")
        self._init_servers()

        if self.cfg.training.mode is TrainingMode.centralized:
            self.centralized_trainer = CentralizedTrainer(
                self.cfg,
                stop_event=self._stop_event,
                mp_context=self._mp_context,
                dataset_report_queue=self._dataset_report_queue,
                resource_metrics_queue=self._resource_metrics_queue,
            )

        logger.info("=== SETUP COMPLETE ===")

    def teardown(self) -> None:
        """
        Gracefully shuts down multiprocessing manager.
        """
        close_process_queue(getattr(self, "_dataset_report_queue", None))
        self._dataset_report_queue = None
        close_process_queue(getattr(self, "_failure_queue", None))
        self._failure_queue = None
        close_process_queue(getattr(self, "_resource_metrics_queue", None))
        self._resource_metrics_queue = None
        close_process_queue(getattr(self, "_telemetry_queue", None))
        self._telemetry_queue = None
        client_plan_queues = getattr(self, "_client_round_plan_queues", {})
        for plan_queue in client_plan_queues.values():
            close_process_queue(plan_queue)
        self._client_round_plan_queues = {}
        close_process_queue(getattr(self, "_split_round_plan_queue", None))
        self._split_round_plan_queue = None
        close_process_queue(getattr(self, "_fed_round_plan_queue", None))
        self._fed_round_plan_queue = None

        if self._manager is not None:
            if self._stop_event is not None:
                self._stop_event.set()
            self._manager.shutdown()
            self._manager = None
            self._stop_event = None

    def _init_channels(self) -> None:
        """Create per-client communication channels for the selected mode."""
        self.channels = build_channels(
            self.cfg,
            self._mp_context,
            self._stop_event,
        )

        logger.info(
            "Channels initialised for %d client(s)", len(self.cfg.clients)
        )

    def _init_servers(self) -> None:
        """Initialize the server instances required by the selected mode."""
        self.split_server, self.fed_server = build_servers(
            self.cfg,
            self.channels,
            self._stop_event,
            self._mp_context,
            self._failure_queue,
            self._resource_metrics_queue,
            getattr(self, "_split_round_plan_queue", None),
            getattr(self, "_fed_round_plan_queue", None),
        )
        if self.split_server is not None:
            logger.info("SplitServer initialised")
        if self.fed_server is not None:
            logger.info("FedServer initialised")

    def start_training(self) -> None:
        """
        Starts distributed training across all clients and servers.
        """

        if not hasattr(self, "resource_metrics"):
            self.resource_metrics = []
        self.run_started_at = time.time()
        run_tracker = ResourceTracker("controller", "cpu")
        if self.cfg.training.mode is TrainingMode.centralized:
            try:
                self._start_centralized_training()
            except BaseException as exc:
                self._capture_first_failure(exc)
                raise
            finally:
                self._drain_dataset_reports()
                self.resource_metrics.append(
                    run_tracker.snapshot(round_idx=None, phase="experiment")
                )
            return
        if (
            self.cfg.training.mode
            in (TrainingMode.split, TrainingMode.splitfed)
            and self.split_server is None
        ):
            raise RuntimeError(
                "SplitServer was not initialized for split mode"
            )
        if (
            self.cfg.training.mode
            in (TrainingMode.federated, TrainingMode.splitfed)
            and self.fed_server is None
        ):
            raise RuntimeError(
                "FedServer was not initialized for federated mode"
            )

        if self.split_server is None and self.fed_server is None:
            raise RuntimeError("Call setup() before start_training().")

        if self._manager is None:
            raise RuntimeError("Manager is not running — was setup() called?")

        if self._stop_event is None:
            raise RuntimeError("Stop event is not initialized — call setup().")

        logger.info("=== STARTING TRAINING ===")

        num_clients = len(self.client_cfgs)
        metrics_path = None
        if getattr(self.cfg, "models_save_path", None):
            metrics_path = ArtifactPaths.from_root(
                self.cfg.models_save_path,
                self.cfg.experiment.name,
            ).metrics

        # Barrier: ensures all clients are ready before training begins
        controller_joins_ready = (
            getattr(self.cfg, "load_controller", None) is not None
        )
        ready_barrier = self._manager.Barrier(
            num_clients + int(controller_joins_ready)
        )

        # Barrier: ensures all clients finish training before evaluation
        eval_barrier = self._manager.Barrier(num_clients)

        training_error: BaseException | None = None
        split_server_config = getattr(self.cfg, "split_server", None)

        try:
            if self.split_server is not None:
                self.split_server.start()
            if self.fed_server is not None:
                self.fed_server.start()

            for client_cfg in self.client_cfgs:
                cid = client_cfg.client_id
                ch = self.channels[cid]

                self._stop_events[cid] = self._stop_event

                process_factory = getattr(self, "_mp_context", mp).Process
                p = process_factory(
                    target=_client_worker,
                    args=(
                        client_cfg,
                        self.cfg.training,
                        ch.get(self.SPLIT_UPLINK),
                        ch.get(self.SPLIT_DOWNLINK),
                        ch.get(self.FED_UPLINK),
                        ch.get(self.FED_DOWNLINK),
                        self._stop_event,
                        ready_barrier,
                        eval_barrier,
                        metrics_path,
                        getattr(self, "_dataset_report_queue", None),
                        getattr(self, "_failure_queue", None),
                        getattr(self, "_resource_metrics_queue", None),
                        (
                            split_server_config.model_scope
                            if split_server_config is not None
                            else None
                        ),
                        getattr(self, "_client_round_plan_queues", {}).get(
                            cid
                        ),
                        getattr(self, "_telemetry_queue", None),
                    ),
                    daemon=False,
                    name=f"Client-{cid}",
                )

                p.start()
                self._client_processes.append(p)

                logger.info("Client '%s' process started (pid=%d)", cid, p.pid)

            if controller_joins_ready:
                self._run_mergesfl_control_loop(ready_barrier)

            _wait_for_training_processes(
                self._client_processes,
                tuple(
                    server
                    for server in (self.split_server, self.fed_server)
                    if server is not None
                ),
                self.PROCESS_POLL_TIMEOUT,
                on_poll=self._drain_resource_metrics,
            )

        except KeyboardInterrupt as exc:
            training_error = exc
            logger.info("Training interrupted — cancelling workers")
            _cancel_training(self._stop_event, (ready_barrier, eval_barrier))
            _shutdown_processes(
                self._client_processes, self.PROCESS_SHUTDOWN_TIMEOUT
            )

        except BaseException as exc:
            training_error = exc

            logger.error(
                "Exception while waiting for clients: %s", exc, exc_info=True
            )

            _cancel_training(self._stop_event, (ready_barrier, eval_barrier))
            _shutdown_processes(
                self._client_processes, self.PROCESS_SHUTDOWN_TIMEOUT
            )
            self._capture_first_failure(exc)

        finally:
            logger.info("=== TRAINING COMPLETE — stopping servers ===")

            if self.split_server is not None:
                self.split_server.stop()
            if self.fed_server is not None:
                self.fed_server.stop()

            self._drain_dataset_reports()
            self._collect_transport_metrics()
            self._drain_resource_metrics()
            self.resource_metrics.append(
                run_tracker.snapshot(round_idx=None, phase="experiment")
            )
            self._client_processes.clear()

        if training_error is not None:
            raise training_error

    def _run_mergesfl_control_loop(self, ready_barrier) -> None:
        policy = self.cfg.load_controller
        if policy is None:
            return
        expected_clients = {client.client_id for client in self.client_cfgs}
        collect_dataset_reports(
            self._dataset_report_queue,
            self.dataset_manifests,
            expected_client_ids=expected_clients,
            timeout=self.cfg.training.barrier_timeout_sec,
        )
        participation = {client_id: 0 for client_id in expected_clients}
        latest_telemetry = bootstrap_telemetry(policy)
        planner = MergeSFLPlanner(policy, self.cfg.training.seed)

        for round_idx in range(1, self.cfg.training.num_rounds + 1):
            profiles = profiles_from_manifests(
                self.dataset_manifests, participation
            )
            plan = planner.plan(
                profiles,
                list(latest_telemetry.values()),
                round_idx=round_idx,
                model_version=f"round-{round_idx - 1}",
            )
            self._dispatch_round_plan(plan)
            self.mergesfl_round_plans.append(plan.to_dict())
            self._persist_mergesfl_plans()
            if round_idx == 1:
                ready_barrier.wait(
                    timeout=self.cfg.training.barrier_timeout_sec
                )
            observations = collect_selected_telemetry(
                self._telemetry_queue,
                expected_client_ids=expected_clients,
                expected_round=round_idx,
                round_deadline_at=plan.deadline_at,
            )
            latest_telemetry.update(observations)
            for client_id in plan.cohort:
                participation[client_id] += 1

    def _dispatch_round_plan(self, plan) -> None:
        self._split_round_plan_queue.put(plan, timeout=5)
        self._fed_round_plan_queue.put(plan, timeout=5)
        for plan_queue in self._client_round_plan_queues.values():
            plan_queue.put(plan, timeout=5)

    def _persist_mergesfl_plans(self) -> None:
        if not getattr(self.cfg, "models_save_path", None):
            return
        paths = ArtifactPaths.from_root(
            self.cfg.models_save_path, self.cfg.experiment.name
        )
        paths.mkdir()
        save_yaml(
            paths.metadata / "mergesfl_round_plans.yaml",
            {
                "policy": self.cfg.load_controller.name,
                "round_plans": self.mergesfl_round_plans,
            },
            verbose=False,
        )

    def _start_centralized_training(self) -> None:
        if self._manager is None or self._stop_event is None:
            raise RuntimeError("Call setup() before start_training().")
        if self.centralized_trainer is None:
            raise RuntimeError("Centralized trainer was not initialized")
        try:
            self.centralized_trainer.start()
            self.centralized_trainer.wait(self.PROCESS_POLL_TIMEOUT)
        except BaseException:
            self._stop_event.set()
            raise
        finally:
            self.centralized_trainer.stop()
            self._drain_resource_metrics()

    def _drain_dataset_reports(self) -> None:
        report_queue = getattr(self, "_dataset_report_queue", None)
        if report_queue is None:
            return
        drain_dataset_reports(
            report_queue,
            self.dataset_manifests,
        )

    def _capture_first_failure(self, fallback: BaseException) -> None:
        self.first_failure = capture_first_failure(
            getattr(self, "first_failure", None),
            getattr(self, "_failure_queue", None),
            fallback,
        )

    def _drain_resource_metrics(self) -> None:
        report_queue = getattr(self, "_resource_metrics_queue", None)
        if report_queue is None:
            return
        while True:
            try:
                self.resource_metrics.append(report_queue.get_nowait())
            except queue.Empty:
                break

    def _collect_transport_metrics(self) -> None:
        for client_id, client_channels in self.channels.items():
            for channel_name, channel in client_channels.items():
                statistics = getattr(channel, "statistics", None)
                if statistics is None:
                    continue
                self.resource_metrics.append(
                    {
                        "schema_version": RESOURCE_METRICS_SCHEMA_VERSION,
                        "measurement_policy": RESOURCE_MEASUREMENT_POLICY,
                        "role": "transport",
                        "client_id": client_id,
                        "process_id": None,
                        "round": None,
                        "phase": channel_name,
                        "byte_accounting": (
                            "serialized_protobuf_v1"
                            if self.cfg.experiment.transport.value == "grpc"
                            else "logical_payload_estimate_v1"
                        ),
                        "wall_time_seconds": 0.0,
                        "cpu_user_seconds": 0.0,
                        "cpu_system_seconds": 0.0,
                        "process_peak_rss_bytes": None,
                        "peak_cuda_allocated_bytes": None,
                        "peak_cuda_reserved_bytes": None,
                        "samples": None,
                        "batches": None,
                        "samples_per_second": None,
                        "batches_per_second": None,
                        **statistics(),
                    }
                )
