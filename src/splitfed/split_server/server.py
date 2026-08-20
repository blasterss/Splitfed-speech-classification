import queue
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

from ...logger import logger
from ...model.server_side_model import ServerSideModel
from ...schema import ServerModelScope, SplitServerConfig
from ...transport.base import Channel, Message
from ...transport.replay import ReplayGuard
from ...utils.checkpoint import save_checkpoint
from ...utils.failures import FailureRecord, publish_failure
from ...utils.process import ignore_parent_interrupts
from ...utils.state import deserialize_state_dict, serialize_state_dict
from ...utils.training import set_seed
from ...utils.training_stats import _RoundStats
from .operations import _handle_eval_single, _handle_train_batch
from .optimization import step_accumulated_gradients
from .personalized_worker import _split_server_worker_personalized
from .protocol import (
    _batch_is_ready,
    _evict_stale_batches,
    _store_pending_batch,
    _validate_message,
)

logger = logger.getChild("SplitServer")


class SplitServer:
    """
    Split-learning server.

    Runs in its own process. Receives activations from clients,
    completes the forward pass, computes loss / gradients, and
    sends gradients back (train) or logits back (eval).

    Channel layout (one pair per client):
        uplink   – server reads  (client writes activations)
        downlink – server writes (client reads  gradients / logits)

    After stop(), call get_state_dict() to retrieve the trained weights,
    or save() to write them directly to disk.
    """

    def __init__(
        self,
        config: SplitServerConfig,
        client_channels: dict[str, dict[str, Channel]],
        stop_event=None,
        mp_context=None,
        failure_queue=None,
    ):
        self.config = config
        self.client_channels = client_channels

        self._mp_context = mp_context or mp.get_context("spawn")
        self._stop_event = (
            stop_event if stop_event is not None else self._mp_context.Event()
        )
        self._result_queue: mp.Queue = self._mp_context.Queue(maxsize=1)
        self._process: mp.Process | None = None
        self._last_exitcode: int | None = None
        self._last_state_dict: dict | None = None
        self._failure_queue = failure_queue

    def start(self) -> None:
        """Spawn the server worker process."""
        self._stop_event.clear()
        worker = (
            _split_server_worker_personalized
            if self.config.model_scope is ServerModelScope.personalized
            else _split_server_worker_batch
        )
        self._process = self._mp_context.Process(
            target=worker,
            args=(
                self.config,
                self.client_channels,
                self._stop_event,
                self._result_queue,
                self._failure_queue,
            ),
            daemon=True,
            name="SplitServer",
        )
        self._last_exitcode = None
        self._process.start()
        logger.info("SplitServer process started (pid=%d)", self._process.pid)

    def stop(self) -> None:
        """Signal the worker to finish and wait for it to exit."""
        self._stop_event.set()
        if self._process is not None:
            try:
                payload = self._result_queue.get(timeout=30)
                self._last_state_dict = deserialize_state_dict(payload)
            except queue.Empty:
                logger.warning("SplitServer produced no final state_dict")
            self._process.join(timeout=30)
            if self._process.is_alive():
                logger.warning(
                    "SplitServer worker did not exit within 30 s — "
                    "terminating."
                )
                self._process.terminate()
                self._process.join(timeout=5)
                if self._process.is_alive():
                    self._process.kill()
                    self._process.join(timeout=5)
                    if self._process.is_alive():
                        raise RuntimeError(
                            "SplitServer worker remained alive after kill"
                        )

            self._last_exitcode = self._process.exitcode

            self._process = None
        logger.info("SplitServer process stopped")

    @property
    def exitcode(self) -> int | None:
        if self._process is not None:
            return self._process.exitcode
        return self._last_exitcode

    def get_state_dict(self) -> dict:
        """
        Return the trained server-side model weights.
        Must be called *after* stop().
        """
        try:
            if self._last_state_dict is not None:
                return self._last_state_dict
            return deserialize_state_dict(self._result_queue.get_nowait())
        except queue.Empty:
            raise RuntimeError(
                "No state_dict available. Either stop() has not been called "
                "yet "
                "or the worker exited abnormally."
            ) from None

    def save(self, path: str) -> None:
        """Retrieve the trained weights and save them to disk."""
        state_dict = self.get_state_dict()
        if self.config.model_scope is ServerModelScope.personalized:
            for client_id, client_state in state_dict.items():
                save_path = Path(path) / f"split_server_client_{client_id}.pt"
                save_checkpoint(
                    save_path,
                    mode=getattr(self, "training_mode", "split"),
                    server_model_scope="personalized",
                    client_id=client_id,
                    model_state_dict=client_state,
                )
                logger.info(
                    "Personalized SplitServer model saved to '%s'", save_path
                )
        else:
            save_path = Path(path) / "split_server.pt"
            save_checkpoint(
                save_path,
                mode=getattr(self, "training_mode", "splitfed"),
                server_model_scope="shared",
                model_state_dict=state_dict,
            )
            logger.info("SplitServer model saved to '%s'", save_path)


def _split_server_worker_batch(
    config: SplitServerConfig,
    client_channels: dict[str, dict[str, Channel]],
    stop_event,
    result_queue: mp.Queue,
    failure_queue=None,
) -> None:
    ignore_parent_interrupts()
    set_seed(config.seed)  # Ensure deterministic behavior in server process,

    device = torch.device(config.model.device)
    model = ServerSideModel(model_type="cnn_birnn").to(device)
    pos_weight = torch.tensor(config.model.pos_weight, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.model.learning_rate)

    client_ids = list(client_channels.keys())
    pending_batches: dict[tuple, dict[str, Message]] = defaultdict(dict)
    pending_timestamps: dict[tuple, float] = {}
    completed_rounds = defaultdict(set)

    logger.info("SplitServer worker ready, serving clients: %s", client_ids)

    current_round = 1
    stats = _RoundStats()
    replay_guard = ReplayGuard()
    current_client_id = None
    current_step = None

    all_eval_probs = []
    all_eval_labels = []
    accumulated_batches = 0
    optimizer.zero_grad()
    try:
        while not stop_event.is_set():
            served_any = False

            for client_id in client_ids:
                uplink: Channel = client_channels[client_id]["uplink"]

                msg = uplink.recv_nowait()
                if msg is None:
                    continue
                current_client_id = client_id
                current_round = msg.round
                current_step = msg.step

                if not _validate_message(msg, client_id):
                    raise ValueError(
                        f"Invalid split message from client {client_id}"
                    )

                replay_guard.accept(msg.request_id)

                served_any = True

                if msg.type == "eval_step":
                    _handle_eval_single(
                        msg,
                        client_id,
                        model,
                        device,
                        client_channels,
                        all_eval_probs,
                        all_eval_labels,
                    )
                elif msg.type == "round_end":
                    if client_id in completed_rounds[msg.round]:
                        raise ValueError(
                            f"Duplicate round_end from client {client_id} "
                            f"for round {msg.round}"
                        )
                    completed_rounds[msg.round].add(client_id)
                elif msg.type == "train_step":
                    if client_id in completed_rounds[msg.round]:
                        raise ValueError(
                            "Train step after round_end from client "
                            f"{client_id}"
                        )
                    key = (msg.round, msg.step)

                    if key not in pending_timestamps:
                        pending_timestamps[key] = time.monotonic()

                    _store_pending_batch(pending_batches, msg, client_id)

                    logger.info(
                        "Received '%s' from %s (round=%d step=%d) [%d/%d]",
                        msg.type,
                        client_id,
                        msg.round,
                        msg.step,
                        len(pending_batches[key]),
                        len(client_ids),
                    )

                else:
                    logger.warning(
                        "SplitServer: unknown msg type '%s' from %s - "
                        "discarding.",
                        msg.type,
                        client_id,
                    )

                ready_keys = [
                    key
                    for key, batch in pending_batches.items()
                    if _batch_is_ready(
                        batch,
                        set(client_ids),
                        completed_rounds[key[0]],
                    )
                ]
                for key in ready_keys:
                    batch_msgs = pending_batches.pop(key)
                    pending_timestamps.pop(key, None)

                    batch_round = next(iter(batch_msgs.values())).round
                    if batch_round != current_round:
                        stats.log_and_reset(current_round)
                        current_round = batch_round

                    batch_loss = _handle_train_batch(
                        batch_msgs,
                        model,
                        criterion,
                        device,
                        client_channels,
                    )
                    if batch_loss is not None:
                        stats.update(batch_loss)
                        accumulated_batches += 1
                        if (
                            accumulated_batches
                            == config.model.gradient_accumulation_steps
                        ):
                            step_accumulated_gradients(
                                model.parameters(),
                                optimizer,
                                accumulated_batches,
                            )
                            accumulated_batches = 0

                completed_round = msg.round
                round_is_complete = completed_rounds[completed_round] == set(
                    client_ids
                )
                round_has_pending = any(
                    key[0] == completed_round for key in pending_batches
                )
                if round_is_complete and not round_has_pending:
                    if accumulated_batches:
                        step_accumulated_gradients(
                            model.parameters(),
                            optimizer,
                            accumulated_batches,
                        )
                        accumulated_batches = 0
                    completed_rounds.pop(completed_round, None)

            _evict_stale_batches(
                pending_batches,
                pending_timestamps,
                client_channels,
                config.model.batch_timeout_sec,
            )

            if not served_any:
                stop_event.wait(timeout=0.001)

    except BaseException as exc:
        publish_failure(
            failure_queue,
            FailureRecord.from_exception(
                component="split_server",
                client_id=current_client_id,
                round=current_round,
                step=current_step,
                exception=exc,
            ),
        )
        stop_event.set()
        raise

    finally:
        stats.log_and_reset(current_round)
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        try:
            result_queue.put_nowait(serialize_state_dict(state_dict))
            logger.info(
                "SplitServer worker exiting — state_dict pushed to "
                "result queue"
            )
        except queue.Full:
            logger.warning(
                "SplitServer result queue was already full — state_dict "
                "NOT pushed."
            )
