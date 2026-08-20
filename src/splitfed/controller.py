import queue

import torch.multiprocessing as mp

from ..logger import get_logger
from ..schema import ConfigSchema, TrainingMode
from ..transport.base import ChannelFactory
from ..utils.artifacts import ArtifactPaths
from ..utils.failures import FailureRecord
from .centralized import CentralizedTrainer
from .client import _client_worker
from .common.lifecycle import (
    _cancel_training,
    _shutdown_processes,
    _wait_for_training_processes,
)
from .fed_server import FedServer
from .split_server import SplitServer

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

    SPLIT_UPLINK = "split_uplink"
    SPLIT_DOWNLINK = "split_downlink"
    FED_UPLINK = "federated_uplink"
    FED_DOWNLINK = "federated_downlink"
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
        self.dataset_manifests: dict[int, dict] = {}
        self.first_failure: dict | None = None

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
            )

        logger.info("=== SETUP COMPLETE ===")

    def teardown(self) -> None:
        """
        Gracefully shuts down multiprocessing manager.
        """
        report_queue = getattr(self, "_dataset_report_queue", None)
        if report_queue is not None:
            close = getattr(report_queue, "close", None)
            if close is not None:
                close()
            join_thread = getattr(report_queue, "join_thread", None)
            if join_thread is not None:
                join_thread()
            self._dataset_report_queue = None

        failure_queue = getattr(self, "_failure_queue", None)
        if failure_queue is not None:
            close = getattr(failure_queue, "close", None)
            if close is not None:
                close()
            join_thread = getattr(failure_queue, "join_thread", None)
            if join_thread is not None:
                join_thread()
            self._failure_queue = None

        if self._manager is not None:
            if self._stop_event is not None:
                self._stop_event.set()
            self._manager.shutdown()
            self._manager = None
            self._stop_event = None

    def _init_channels(self) -> None:
        """
        Creates per-client communication channels.
        """

        required = set()
        if self.cfg.training.mode in (
            TrainingMode.split,
            TrainingMode.splitfed,
        ):
            required.update({self.SPLIT_UPLINK, self.SPLIT_DOWNLINK})
        if self.cfg.training.mode in (
            TrainingMode.federated,
            TrainingMode.splitfed,
        ):
            required.update({self.FED_UPLINK, self.FED_DOWNLINK})

        for client_cfg in self.cfg.clients:
            cid = client_cfg.client_id
            self.channels[cid] = {}

            for name, params in self.cfg.channels.items():
                if name not in required:
                    logger.warning(
                        "Unexpected channel name '%s' in config – skipping",
                        name,
                    )
                    continue

                self.channels[cid][name] = ChannelFactory.create(
                    params,
                    mp_context=self._mp_context,
                    stop_event=self._stop_event,
                )

            missing = required - self.channels[cid].keys()

            if missing:
                raise ValueError(
                    f"Client '{cid}' is missing channel definitions: {missing}"
                )

        logger.info(
            "Channels initialised for %d client(s)", len(self.cfg.clients)
        )

    def _init_servers(self) -> None:
        """
        Initializes SplitServer and FedServer instances.
        """

        if self.cfg.split_server is not None:
            split_channels = {
                cid: {
                    "uplink": self.channels[cid][self.SPLIT_UPLINK],
                    "downlink": self.channels[cid][self.SPLIT_DOWNLINK],
                }
                for cid in self.channels
            }
            self.split_server = SplitServer(
                config=self.cfg.split_server,
                client_channels=split_channels,
                stop_event=self._stop_event,
                mp_context=self._mp_context,
                failure_queue=self._failure_queue,
            )
            self.split_server.training_mode = self.cfg.training.mode.value

            logger.info("SplitServer initialised")

        if self.cfg.fed_server is not None:
            fed_channels = {
                cid: {
                    "uplink": self.channels[cid][self.FED_UPLINK],
                    "downlink": self.channels[cid][self.FED_DOWNLINK],
                }
                for cid in self.channels
            }
            self.fed_server = FedServer(
                config=self.cfg.fed_server,
                client_channels=fed_channels,
                num_clients=len(self.cfg.clients),
                stop_event=self._stop_event,
                mp_context=self._mp_context,
                failure_queue=self._failure_queue,
            )
            self.fed_server.training_mode = self.cfg.training.mode.value

            logger.info("FedServer initialised")

    def start_training(self) -> None:
        """
        Starts distributed training across all clients and servers.
        """

        if self.cfg.training.mode is TrainingMode.centralized:
            try:
                self._start_centralized_training()
            except BaseException as exc:
                self._capture_first_failure(exc)
                raise
            finally:
                self._drain_dataset_reports()
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
        ready_barrier = self._manager.Barrier(num_clients)

        # Barrier: ensures all clients finish training before evaluation
        eval_barrier = self._manager.Barrier(num_clients)

        training_error: BaseException | None = None

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
                    ),
                    daemon=False,
                    name=f"Client-{cid}",
                )

                p.start()
                self._client_processes.append(p)

                logger.info("Client '%s' process started (pid=%d)", cid, p.pid)

            _wait_for_training_processes(
                self._client_processes,
                tuple(
                    server
                    for server in (self.split_server, self.fed_server)
                    if server is not None
                ),
                self.PROCESS_POLL_TIMEOUT,
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
            self._client_processes.clear()

        if training_error is not None:
            raise training_error

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

    def _drain_dataset_reports(self) -> None:
        report_queue = getattr(self, "_dataset_report_queue", None)
        if report_queue is None:
            return
        while True:
            try:
                report = report_queue.get_nowait()
            except queue.Empty:
                break
            client_id = report.get("client_id")
            if not isinstance(client_id, int):
                raise ValueError("Dataset report has invalid client_id")
            if client_id in self.dataset_manifests:
                raise ValueError(
                    f"Duplicate dataset report for client {client_id}"
                )
            self.dataset_manifests[client_id] = report

    def _capture_first_failure(self, fallback: BaseException) -> None:
        if getattr(self, "first_failure", None) is not None:
            return
        failure_queue = getattr(self, "_failure_queue", None)
        if failure_queue is not None:
            try:
                self.first_failure = failure_queue.get_nowait()
                return
            except queue.Empty:
                pass
        self.first_failure = FailureRecord.from_exception(
            component="controller", exception=fallback
        ).as_dict()
