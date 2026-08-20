import copy
import queue
import time
from pathlib import Path

import torch
import torch.multiprocessing as mp

from ..logger import logger
from ..schema import AggregationStrategy, FedServerConfig
from ..transport.base import Channel, Message
from ..transport.replay import ReplayGuard
from ..utils.checkpoint import save_checkpoint
from ..utils.state import deserialize_state_dict, serialize_state_dict
from ..utils.training import set_seed


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
    ) -> dict:
        """
        Performs Federated Averaging (FedAvg).

        Args:
            client_params_list: list of model state_dicts
            client_sizes: number of samples per client

        Returns:
            Aggregated global state_dict
        """

        if not client_params_list or len(client_params_list) != len(
            client_sizes
        ):
            raise ValueError(
                "FedAvg requires matching non-empty parameter and size lists"
            )
        if any(size <= 0 for size in client_sizes):
            raise ValueError("FedAvg client sample counts must be positive")

        if strategy is AggregationStrategy.fedavg:
            weights = [1.0 / len(client_params_list)] * len(client_params_list)
        elif strategy is AggregationStrategy.weighted_fedavg:
            total_samples = sum(client_sizes)
            weights = [size / total_samples for size in client_sizes]
        else:
            raise ValueError(f"Unsupported aggregation strategy: {strategy}")

        new_params = copy.deepcopy(client_params_list[0])
        largest_client = max(
            range(len(client_sizes)), key=client_sizes.__getitem__
        )

        for key in new_params.keys():
            value = client_params_list[0][key]

            if torch.is_tensor(value) and (
                torch.is_floating_point(value) or torch.is_complex(value)
            ):
                new_params[key] = sum(
                    client_params_list[i][key] * weights[i]
                    for i in range(len(client_params_list))
                )
            else:
                new_params[key] = client_params_list[largest_client][key]

        return new_params

    @staticmethod
    def aggregate_metrics(
        eval_metrics: list[tuple[int, dict[str, float]]],
    ) -> dict[str, float]:
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
    client_channels: dict[str, dict[str, Channel]],
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

    latest_params: dict | None = None
    latest_round: int = 1

    updates: dict[str, dict] = {}
    sizes: dict[str, int] = {}
    request_ids: dict[str, str] = {}
    active_round: int | None = None
    last_completed_round = 0
    expected_schema = None
    round_started_at: float | None = None
    replay_guard = ReplayGuard()

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
                replay_guard.accept(msg.request_id)

                if msg.round <= last_completed_round:
                    _validate_client_update(
                        msg,
                        expected_client_id=client_id,
                        expected_round=msg.round,
                        expected_schema=None,
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

                state_dict, dataset_size = _validate_client_update(
                    msg,
                    expected_client_id=client_id,
                    expected_round=active_round,
                    expected_schema=expected_schema,
                )
                if expected_schema is None:
                    expected_schema = _state_schema(state_dict)

                updates[client_id] = state_dict
                sizes[client_id] = dataset_size
                request_ids[client_id] = msg.request_id
                latest_round = active_round

                logger.info(
                    "Received update from client %s (round=%d, samples=%d)",
                    client_id,
                    msg.round,
                    sizes[client_id],
                )

            if active_round is not None:
                assert round_started_at is not None
                elapsed = time.monotonic() - round_started_at
                decision = _quorum_decision(
                    update_count=len(updates),
                    client_count=num_clients,
                    min_clients=config.min_clients,
                    elapsed=elapsed,
                    timeout=config.quorum_timeout_sec,
                )
            else:
                decision = "wait"

            if decision == "fail":
                raise RuntimeError(
                    f"Federated quorum timeout for round {active_round}: "
                    f"received {len(updates)}/{config.min_clients} required"
                )

            if decision == "aggregate":
                participant_ids = [cid for cid in client_ids if cid in updates]
                params_list = [updates[cid] for cid in participant_ids]
                sizes_list = [sizes[cid] for cid in participant_ids]

                try:
                    latest_params = FedServer.aggregate(
                        params_list, sizes_list, config.strategy
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
                    downlink: Channel = client_channels[client_id]["downlink"]

                    downlink.send(
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
                request_ids.clear()
                last_completed_round = active_round
                active_round = None
                expected_schema = None
                round_started_at = None

            if not served_any:
                stop_event.wait(timeout=0.001)

    finally:
        if latest_params is not None:
            try:
                state_dict = {k: v.cpu() for k, v in latest_params.items()}
                result_queue.put_nowait(serialize_state_dict(state_dict))

                logger.info(
                    "FedServer worker exiting — final state_dict saved"
                )

            except queue.Full:
                logger.warning(
                    "Result queue full — final state_dict not stored"
                )

        else:
            try:
                result_queue.put_nowait(None)
            except queue.Full:
                pass
            logger.warning(
                "FedServer worker exiting — no aggregation completed"
            )


def _state_schema(state_dict: dict) -> dict:
    return {
        key: (value.shape, value.dtype) for key, value in state_dict.items()
    }


def _quorum_decision(
    update_count: int,
    client_count: int,
    min_clients: int,
    elapsed: float,
    timeout: float,
) -> str:
    if update_count >= client_count:
        return "aggregate"
    if elapsed < timeout:
        return "wait"
    if update_count >= min_clients:
        return "aggregate"
    return "fail"


def _validate_client_update(
    message: Message,
    expected_client_id,
    expected_round: int | None,
    expected_schema: dict | None,
) -> tuple[dict, int]:
    message.validate_for_receive()
    if message.sender != expected_client_id:
        raise ValueError("Invalid client update sender")
    if message.type != "client_update":
        raise ValueError("Invalid client update type")
    if expected_round is not None and message.round != expected_round:
        raise ValueError("Invalid client update round")
    if message.round <= 0 or message.step != 1:
        raise ValueError("Invalid client update round or step")
    if not isinstance(message.payload, dict):
        raise ValueError("Invalid client update payload")

    state_dict = message.payload.get("state_dict")
    dataset_size = message.payload.get("dataset_size")
    if not isinstance(dataset_size, int) or isinstance(dataset_size, bool):
        raise ValueError("Invalid client update dataset_size")
    if dataset_size <= 0:
        raise ValueError("Invalid client update dataset_size")
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("Invalid client update state_dict")
    if not all(
        isinstance(value, torch.Tensor) for value in state_dict.values()
    ):
        raise ValueError("Invalid client update state tensor")

    if expected_schema is not None:
        if state_dict.keys() != expected_schema.keys():
            raise ValueError("Invalid client update state keys")
        for key, value in state_dict.items():
            shape, dtype = expected_schema[key]
            if value.shape != shape:
                raise ValueError(f"Invalid client update shape for {key}")
            if value.dtype != dtype:
                raise ValueError(f"Invalid client update dtype for {key}")

    return state_dict, dataset_size
