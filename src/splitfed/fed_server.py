import copy
import queue
from pathlib import Path
import torch
import torch.multiprocessing as mp

from ..transport.base import Channel, Message
from ..utils.training import set_seed
from ..schema import FedServerConfig
from ..logger import logger

from typing import Dict, List, Optional, Tuple


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
        client_channels: Dict[str, Dict[str, Channel]],
        num_clients: int,
        stop_event=None,
    ):
        self.config = config
        self.client_channels = client_channels
        self.num_clients = num_clients

        self._stop_event = stop_event if stop_event is not None else mp.Event()

        # Stores final aggregated model after shutdown
        self._result_queue: mp.Queue = mp.Queue(maxsize=1)
        self._process: Optional[mp.Process] = None

    def start(self) -> None:
        """
        Starts the federated server worker process.
        """
        self._stop_event.clear()

        self._process = mp.Process(
            target=_fed_server_worker,
            args=(
                self.config,
                self.client_channels,
                self.num_clients,
                self._stop_event,
                self._result_queue,
            ),
            daemon=True,
            name="FedServer",
        )

        self._process.start()

        logger.info("FedServer process started (pid=%d)", self._process.pid)

    def stop(self) -> None:
        """
        Stops the federated server process safely.
        """
        self._stop_event.set()

        if self._process is not None:
            self._process.join(timeout=30)

            if self._process.is_alive():
                logger.warning(
                    "FedServer worker did not exit within timeout — terminating."
                )

                self._process.terminate()
                self._process.join(timeout=5)

                if self._process.is_alive():
                    self._process.kill()

            self._process = None

        logger.info("FedServer process stopped")

    def get_state_dict(self) -> dict:
        """
        Returns final aggregated global model parameters.

        Must be called after stop().
        """
        try:
            return self._result_queue.get_nowait()
        except queue.Empty:
            raise RuntimeError(
                "No state_dict available. Ensure stop() was called and "
                "aggregation completed successfully."
            )

    def save(self, path: str) -> None:
        """
        Saves final global model to disk.
        """
        save_path = Path(path) / "global_client_model.pt"
        state_dict = self.get_state_dict()

        torch.save(state_dict, save_path)

        logger.info("FedServer global model saved to '%s'", save_path)

    @staticmethod
    def aggregate(
        client_params_list: List[dict],
        client_sizes: List[int],
    ) -> dict:
        """
        Performs Federated Averaging (FedAvg).

        Args:
            client_params_list: list of model state_dicts
            client_sizes: number of samples per client

        Returns:
            Aggregated global state_dict
        """

        total_samples = sum(client_sizes)

        if total_samples == 0:
            raise ValueError(
                "FedAvg: total sample count is zero — cannot aggregate."
            )

        new_params = copy.deepcopy(client_params_list[0])

        for key in new_params.keys():
            new_params[key] = sum(
                client_params_list[i][key] * (client_sizes[i] / total_samples)
                for i in range(len(client_params_list))
            )

        return new_params

    @staticmethod
    def aggregate_metrics(
        eval_metrics: List[Tuple[int, Dict[str, float]]],
    ) -> Dict[str, float]:
        """
        Computes weighted average of evaluation metrics across clients.
        """

        total_num = sum(num for num, _ in eval_metrics)

        if total_num == 0:
            return {}

        all_keys = {k for _, m in eval_metrics for k in m}

        return {
            key: sum(m[key] * num for num, m in eval_metrics if key in m)
            / total_num
            for key in all_keys
        }


def _fed_server_worker(
    config: FedServerConfig,
    client_channels: Dict[str, Dict[str, Channel]],
    num_clients: int,
    stop_event,
    result_queue: mp.Queue,
) -> None:
    """
    Background worker implementing the federated aggregation loop.
    """

    set_seed(config.seed)  # Ensure deterministic behavior in server process,

    client_ids = list(client_channels.keys())

    logger.info("FedServer worker ready, serving clients: %s", client_ids)

    latest_params: Optional[dict] = None
    latest_round: int = 1

    updates: Dict[str, dict] = {}
    sizes: Dict[str, int] = {}

    try:
        while not stop_event.is_set():
            served_any = False

            for client_id in client_ids:
                if client_id in updates:
                    continue

                uplink: Channel = client_channels[client_id]["uplink"]
                msg = uplink.recv_nowait()

                if msg is None:
                    continue

                served_any = True

                if msg.type != "client_update":
                    logger.warning(
                        "Expected 'client_update' from %s, got '%s'",
                        client_id,
                        msg.type,
                    )
                    continue

                if (
                    not isinstance(msg.payload, dict)
                    or "state_dict" not in msg.payload
                    or "dataset_size" not in msg.payload
                ):
                    logger.warning(
                        "Malformed payload from client %s (round=%d)",
                        client_id,
                        msg.round,
                    )
                    continue

                updates[client_id] = msg.payload["state_dict"]
                sizes[client_id] = msg.payload["dataset_size"]
                latest_round = msg.round

                logger.info(
                    "Received update from client %s (round=%d, samples=%d)",
                    client_id,
                    msg.round,
                    sizes[client_id],
                )

            if len(updates) >= num_clients:
                params_list = [updates[cid] for cid in client_ids]
                sizes_list = [sizes[cid] for cid in client_ids]

                try:
                    latest_params = FedServer.aggregate(
                        params_list, sizes_list
                    )
                except Exception as exc:
                    logger.error(
                        "Aggregation failed (round=%d): %s",
                        latest_round,
                        exc,
                        exc_info=True,
                    )
                    updates.clear()
                    sizes.clear()
                    continue

                logger.info(
                    "Aggregated %d client updates (round=%d)",
                    len(updates),
                    latest_round,
                )

                for client_id in client_ids:
                    downlink: Channel = client_channels[client_id]["downlink"]

                    downlink.send(
                        Message(
                            type="global_update",
                            sender="fed_server",
                            round=latest_round,
                            step=1,
                            payload=latest_params,
                        )
                    )

                updates.clear()
                sizes.clear()

            if not served_any:
                stop_event.wait(timeout=0.001)

    finally:
        if latest_params is not None:
            try:
                result_queue.put_nowait(
                    {k: v.cpu() for k, v in latest_params.items()}
                )

                logger.info(
                    "FedServer worker exiting — final state_dict saved"
                )

            except queue.Full:
                logger.warning(
                    "Result queue full — final state_dict not stored"
                )

        else:
            logger.warning(
                "FedServer worker exiting — no aggregation completed"
            )
