import torch.multiprocessing as mp

from .split_server import SplitServer
from .fed_server import FedServer
from .client import Client, _client_worker

from ..transport.base import ChannelFactory
from ..schema import ConfigSchema
from ..logger import get_logger

from typing import Dict, List, Optional

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

    def __init__(self, config: ConfigSchema):
        self.cfg = config
        self.client_cfgs = config.clients

        self.split_server: Optional[SplitServer] = None
        self.fed_server: Optional[FedServer] = None

        self.channels: Dict[str, Dict] = {}
        self._client_processes: List[mp.Process] = []
        self._manager: Optional[mp.managers.SyncManager] = None
        self._stop_events: Dict[str, mp.Event] = {}

    def setup(self) -> None:
        """
        Initializes multiprocessing manager, channels, and servers.
        """
        logger.info("=== TRAINING CONTROLLER SETUP ===")

        self._manager = mp.Manager()

        logger.info("Initialising channels...")
        self._init_channels()

        logger.info("Initialising servers...")
        self._init_servers()

        logger.info("=== SETUP COMPLETE ===")

    def teardown(self) -> None:
        """
        Gracefully shuts down multiprocessing manager.
        """
        if self._manager is not None:
            self._manager.shutdown()
            self._manager = None

    def _init_channels(self) -> None:
        """
        Creates per-client communication channels.
        """

        required = {
            self.SPLIT_UPLINK,
            self.SPLIT_DOWNLINK,
            self.FED_UPLINK,
            self.FED_DOWNLINK,
        }

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

                self.channels[cid][name] = ChannelFactory.create(params)

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
            stop_event=self._manager.Event(),
        )

        logger.info("SplitServer initialised")

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
            stop_event=self._manager.Event(),
        )

        logger.info("FedServer initialised")

    def start_training(self) -> None:
        """
        Starts distributed training across all clients and servers.
        """

        if self.split_server is None or self.fed_server is None:
            raise RuntimeError("Call setup() before start_training().")

        if self._manager is None:
            raise RuntimeError("Manager is not running — was setup() called?")

        logger.info("=== STARTING TRAINING ===")

        self.split_server.start()
        self.fed_server.start()

        num_clients = len(self.client_cfgs)

        # Barrier: ensures all clients are ready before training begins
        ready_barrier = self._manager.Barrier(num_clients)

        # Barrier: ensures all clients finish training before evaluation
        eval_barrier = self._manager.Barrier(num_clients)

        for client_cfg in self.client_cfgs:
            cid = client_cfg.client_id
            ch = self.channels[cid]

            stop_event = self._manager.Event()
            self._stop_events[cid] = stop_event

            p = mp.Process(
                target=_client_worker,
                args=(
                    client_cfg,
                    self.cfg.training,
                    ch[self.SPLIT_UPLINK],
                    ch[self.SPLIT_DOWNLINK],
                    ch[self.FED_UPLINK],
                    ch[self.FED_DOWNLINK],
                    stop_event,
                    ready_barrier,
                    eval_barrier,
                ),
                daemon=False,
                name=f"Client-{cid}",
            )

            p.start()
            self._client_processes.append(p)

            logger.info("Client '%s' process started (pid=%d)", cid, p.pid)

        training_error: Optional[BaseException] = None

        try:
            for p in self._client_processes:
                p.join()

                if p.exitcode != 0:
                    logger.error(
                        "Process %s exited with code %d", p.name, p.exitcode
                    )

        except BaseException as exc:
            training_error = exc

            logger.error(
                "Exception while waiting for clients: %s", exc, exc_info=True
            )

            for event in self._stop_events.values():
                event.set()

            for p in self._client_processes:
                p.join(timeout=10)

                if p.is_alive():
                    p.terminate()

        finally:
            logger.info("=== TRAINING COMPLETE — stopping servers ===")

            self.split_server.stop()
            self.fed_server.stop()

            self._client_processes.clear()

        if training_error is not None:
            raise training_error

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
