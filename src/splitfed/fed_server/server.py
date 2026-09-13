import queue
from pathlib import Path

import torch
import torch.multiprocessing as mp

from ...logger import logger
from ...schema import AggregationStrategy, FedServerConfig
from ...transport.base import Channel
from ...utils.persistence import deserialize_state_dict, save_checkpoint
from .aggregation import aggregate_metrics, aggregate_states
from .worker import _fed_server_worker


class FedServer:
    """
    Federated Averaging (FedAvg) server.

    This server runs in a separate process and performs:
        1. Collection of client model updates (state_dicts)
        2. Weighted aggregation using dataset sizes
        3. Broadcasting updated global model to all clients

    Communication model:
        - uplink:   client → server (client sends state_dict)
        - downlink: server → client (server sends aggregated model)
    """

    def __init__(
        self,
        config: FedServerConfig,
        client_channels: dict[str, dict[str, Channel]],
        num_clients: int,
        stop_event=None,
        mp_context=None,
        failure_queue=None,
        resource_metrics_queue=None,
        round_plan_queue=None,
    ):
        self.config = config
        self.client_channels = client_channels
        self.num_clients = num_clients

        self._mp_context = mp_context or mp.get_context("spawn")
        self._stop_event = (
            stop_event if stop_event is not None else self._mp_context.Event()
        )

        # Stores final aggregated model after shutdown
        self._result_queue: mp.Queue = self._mp_context.Queue(maxsize=1)
        self._process: mp.Process | None = None
        self._last_exitcode: int | None = None
        self._last_state_dict: dict | None = None
        self._failure_queue = failure_queue
        self._resource_metrics_queue = resource_metrics_queue
        self._round_plan_queue = round_plan_queue

    def start(self) -> None:
        """
        Starts the federated server worker process.
        """
        self._stop_event.clear()

        self._process = self._mp_context.Process(
            target=_fed_server_worker,
            args=(
                self.config,
                self.client_channels,
                self.num_clients,
                self._stop_event,
                self._result_queue,
                self._failure_queue,
                self._resource_metrics_queue,
                self._round_plan_queue,
            ),
            daemon=True,
            name="FedServer",
        )
        self._last_exitcode = None

        self._process.start()

        logger.info("FedServer process started (pid=%d)", self._process.pid)

    def stop(self) -> None:
        """
        Stops the federated server process safely.
        """
        self._stop_event.set()

        if self._process is not None:
            try:
                payload = self._result_queue.get(timeout=30)
                if payload is not None:
                    self._last_state_dict = deserialize_state_dict(payload)
            except queue.Empty:
                logger.warning("FedServer produced no final state_dict")
            self._process.join(timeout=30)

            if self._process.is_alive():
                logger.warning(
                    "FedServer worker did not exit within timeout — "
                    "terminating."
                )

                self._process.terminate()
                self._process.join(timeout=5)

                if self._process.is_alive():
                    self._process.kill()
                    self._process.join(timeout=5)
                    if self._process.is_alive():
                        raise RuntimeError(
                            "FedServer worker remained alive after kill"
                        )

            self._last_exitcode = self._process.exitcode

            self._process = None

        logger.info("FedServer process stopped")

    @property
    def exitcode(self) -> int | None:
        if self._process is not None:
            return self._process.exitcode
        return self._last_exitcode

    def get_state_dict(self) -> dict:
        """
        Returns final aggregated global model parameters.

        Must be called after stop().
        """
        try:
            if self._last_state_dict is not None:
                return self._last_state_dict
            payload = self._result_queue.get_nowait()
            if payload is None:
                raise RuntimeError("No federated aggregation completed.")
            return deserialize_state_dict(payload)
        except queue.Empty:
            raise RuntimeError(
                "No state_dict available. Ensure stop() was called and "
                "aggregation completed successfully."
            ) from None

    def save(self, path: str) -> None:
        """
        Saves final global model to disk.
        """
        save_path = Path(path) / "global_client_model.pt"
        state_dict = self.get_state_dict()

        save_checkpoint(
            save_path,
            mode=getattr(self, "training_mode", "splitfed"),
            model_state_dict=state_dict,
        )

        logger.info("FedServer global model saved to '%s'", save_path)

    @staticmethod
    def aggregate(
        client_params_list: list[dict],
        client_sizes: list[int],
        strategy: AggregationStrategy = AggregationStrategy.weighted_fedavg,
        device: torch.device | str = "cpu",
    ) -> dict:
        """
        Performs Federated Averaging (FedAvg).

        Args:
            client_params_list: list of model state_dicts
            client_sizes: number of samples per client

        Returns:
            Aggregated global state_dict
        """

        return aggregate_states(
            client_params_list,
            client_sizes,
            strategy,
            device,
        )

    @staticmethod
    def aggregate_metrics(
        eval_metrics: list[tuple[int, dict[str, float]]],
    ) -> dict[str, float]:
        """
        Computes weighted average of evaluation metrics across clients.
        """

        return aggregate_metrics(eval_metrics)
