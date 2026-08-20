import queue

import torch.multiprocessing as mp

from ..logger import get_logger
from ..schema import ConfigSchema, TrainingMode
from ..transport.base import ChannelFactory
from ..utils.artifacts import ArtifactPaths
from .centralized import CentralizedTrainer
from .client import Client, _client_worker
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
        self.dataset_manifests: dict[int, dict] = {}

    def setup(self) -> None:
        """
        Initializes multiprocessing manager, channels, and servers.
        """
        logger.info("=== TRAINING CONTROLLER SETUP ===")

        self._manager = self._mp_context.Manager()
        self._stop_event = self._manager.Event()
        self._dataset_report_queue = self._mp_context.Queue()

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
                    params, mp_context=self._mp_context
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

        except BaseException as exc:
            training_error = exc

            logger.error(
                "Exception while waiting for clients: %s", exc, exc_info=True
            )

            _cancel_training(self._stop_event, (ready_barrier, eval_barrier))
            _shutdown_processes(
                self._client_processes, self.PROCESS_SHUTDOWN_TIMEOUT
            )

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


def _raise_for_failed_processes(processes: list[mp.Process]) -> None:
    failures = [
        f"{process.name} (exitcode={process.exitcode})"
        for process in processes
        if process.exitcode not in (None, 0)
    ]

    if failures:
        message = "Client process failure: " + ", ".join(failures)
        logger.error(message)
        raise RuntimeError(message)


def _raise_for_failed_servers(servers: tuple[object, ...]) -> None:
    failures = [
        f"{type(server).__name__} (exitcode={server.exitcode})"
        for server in servers
        if server.exitcode not in (None, 0)
    ]

    if failures:
        message = "Server process failure: " + ", ".join(failures)
        logger.error(message)
        raise RuntimeError(message)


def _wait_for_training_processes(
    processes: list[mp.Process],
    servers: tuple[object, ...],
    poll_timeout: float,
) -> None:
    remaining = list(processes)

    while remaining:
        _raise_for_failed_servers(servers)

        for process in remaining[:]:
            process.join(timeout=poll_timeout)

            if not process.is_alive():
                remaining.remove(process)

        _raise_for_failed_processes(
            [process for process in processes if not process.is_alive()]
        )

    _raise_for_failed_servers(servers)


def _cancel_training(stop_event, barriers: tuple[object, ...]) -> None:
    stop_event.set()

    for barrier in barriers:
        try:
            barrier.abort()
        except Exception as exc:
            logger.debug("Could not abort training barrier: %s", exc)


def _shutdown_processes(
    processes: list[mp.Process], join_timeout: float
) -> None:
    for process in processes:
        process.join(timeout=join_timeout)

    alive_processes = [process for process in processes if process.is_alive()]

    for process in alive_processes:
        process.terminate()

    for process in alive_processes:
        process.join(timeout=join_timeout)

    killed_processes = []
    for process in alive_processes:
        if process.is_alive():
            process.kill()
            killed_processes.append(process)

    for process in killed_processes:
        process.join(timeout=join_timeout)

    unreaped = [
        process.name for process in killed_processes if process.is_alive()
    ]
    if unreaped:
        raise RuntimeError(
            "Processes remained alive after kill: " + ", ".join(unreaped)
        )

    def evaluate_all(self) -> None:
        """
        Runs evaluation on all clients sequentially and aggregates results.
        """

        results = []

        for client_cfg in self.client_cfgs:
            cid = client_cfg.client_id
            ch = self.channels[cid]

            client = Client(
                cfg=client_cfg,
                split_uplink_channel=ch[self.SPLIT_UPLINK],
                split_downlink_channel=ch[self.SPLIT_DOWNLINK],
                fed_uplink_channel=ch[self.FED_UPLINK],
                fed_downlink_channel=ch[self.FED_DOWNLINK],
            )

            try:
                metrics = client.evaluate()

                dataset_size = len(client.dataset.test_dataset)

                results.append((dataset_size, metrics))

                logger.info("Client '%s' eval: %s", cid, metrics)

            except Exception as e:
                logger.error(
                    "Client '%s' eval error: %s", cid, e, exc_info=True
                )

        if results:
            agg = FedServer.aggregate_metrics(results)
            logger.info("Aggregated eval metrics: %s", agg)
        else:
            logger.warning("No evaluation results collected")
