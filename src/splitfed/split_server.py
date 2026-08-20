import queue
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from pathlib import Path

from ..utils.training_stats import _RoundStats

from ..model.server_side_model import ServerSideModel
from ..transport.base import Channel, Message
from ..utils.training import set_seed
from ..schema import SplitServerConfig
from ..logger import logger

from typing import Dict, Optional
from collections import defaultdict
import time

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
        client_channels: Dict[str, Dict[str, Channel]],
        stop_event=None,
    ):
        self.config = config
        self.client_channels = client_channels

        self._stop_event = stop_event if stop_event is not None else mp.Event()
        self._result_queue: mp.Queue = mp.Queue(maxsize=1)
        self._process: Optional[mp.Process] = None
        self._last_exitcode: int | None = None

    def start(self) -> None:
        """Spawn the server worker process."""
        self._stop_event.clear()
        self._process = mp.Process(
            target=_split_server_worker_batch,
            args=(
                self.config,
                self.client_channels,
                self._stop_event,
                self._result_queue,
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
            self._process.join(timeout=30)
            if self._process.is_alive():
                logger.warning(
                    "SplitServer worker did not exit within 30 s — terminating."
                )
                self._process.terminate()
                self._process.join(timeout=5)
                if self._process.is_alive():
                    self._process.kill()

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
            return self._result_queue.get_nowait()
        except queue.Empty:
            raise RuntimeError(
                "No state_dict available. Either stop() has not been called yet "
                "or the worker exited abnormally."
            )

    def save(self, path: str) -> None:
        """Retrieve the trained weights and save them to disk."""
        save_path = Path(path) / "split_server.pt"
        state_dict = self.get_state_dict()
        torch.save(state_dict, save_path)
        logger.info("SplitServer model saved to '%s'", save_path)


def _split_server_worker_batch(
    config: SplitServerConfig,
    client_channels: Dict[str, Dict[str, Channel]],
    stop_event,
    result_queue: mp.Queue,
) -> None:
    set_seed(config.seed)  # Ensure deterministic behavior in server process,

    device = torch.device(config.model.device)
    model = ServerSideModel(model_type="cnn_birnn").to(device)
    pos_weight = torch.tensor(config.model.pos_weight, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.model.learning_rate)

    client_ids = list(client_channels.keys())
    pending_batches: Dict[tuple, Dict[str, Message]] = defaultdict(dict)
    pending_timestamps: Dict[tuple, float] = {}
    completed_rounds = defaultdict(set)

    logger.info("SplitServer worker ready, serving clients: %s", client_ids)

    current_round = 1
    stats = _RoundStats()

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

                if not _validate_message(msg, client_id):
                    raise ValueError(
                        f"Invalid split message from client {client_id}"
                    )

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
                            f"Train step after round_end from client {client_id}"
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
                        "SplitServer: unknown msg type '%s' from %s - discarding.",
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
                            _step_accumulated_gradients(
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
                        _step_accumulated_gradients(
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

    finally:
        stats.log_and_reset(current_round)
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        try:
            result_queue.put_nowait(state_dict)
            logger.info(
                "SplitServer worker exiting — state_dict pushed to result queue"
            )
        except queue.Full:
            logger.warning(
                "SplitServer result queue was already full — state_dict NOT pushed."
            )


def _validate_message(msg: Message, client_id: str) -> bool:
    if msg.sender != client_id:
        logger.warning(
            "SplitServer: sender mismatch on client %s channel (got %s)",
            client_id,
            msg.sender,
        )
        return False

    if msg.type not in ("train_step", "eval_step", "round_end"):
        logger.warning(
            "SplitServer: unknown message type '%s' from client %s — discarding.",
            msg.type,
            client_id,
        )
        return False

    if msg.round <= 0 or msg.step <= 0:
        logger.warning(
            "SplitServer: invalid correlation from client %s (round=%d step=%d)",
            client_id,
            msg.round,
            msg.step,
        )
        return False

    if not isinstance(msg.payload, dict):
        logger.warning(
            "SplitServer: payload is not a dict (client %s, type %s) — discarding.",
            client_id,
            msg.type,
        )
        return False

    if msg.type == "round_end":
        return True

    if "activations" not in msg.payload:
        logger.warning(
            "SplitServer: missing 'activations' in payload (client %s) — discarding.",
            client_id,
        )
        return False

    if "labels" not in msg.payload:
        logger.warning(
            "SplitServer: missing 'labels' in payload (client %s) — discarding.",
            client_id,
        )
        return False

    act = msg.payload["activations"]
    if not isinstance(act, torch.Tensor) or act.dim() < 1:
        logger.warning(
            "SplitServer: 'activations' is not a valid tensor (client %s) — discarding.",
            client_id,
        )
        return False

    labels = msg.payload["labels"]
    if (
        not isinstance(labels, torch.Tensor)
        or labels.dim() < 1
        or labels.shape[0] != act.shape[0]
    ):
        logger.warning(
            "SplitServer: invalid labels for activation batch (client %s)",
            client_id,
        )
        return False

    return True


def _batch_is_ready(
    batch: Dict[str, Message],
    client_ids: set,
    completed_clients: set,
) -> bool:
    missing_clients = client_ids - set(batch)
    return missing_clients <= completed_clients


def _store_pending_batch(
    pending_batches: Dict[tuple, Dict[str, Message]],
    message: Message,
    client_id: str,
) -> None:
    key = (message.round, message.step)
    if client_id in pending_batches.setdefault(key, {}):
        raise ValueError(
            f"Duplicate split step from client {client_id} for key {key}"
        )
    pending_batches[key][client_id] = message


def _resolve_batch_type(
    batch_msgs: Dict[str, Message], key: tuple
) -> Optional[str]:
    types = {msg.type for msg in batch_msgs.values()}
    if len(types) == 1:
        return types.pop()
    logger.warning(
        "SplitServer: mixed message types %s for key %s — discarding batch.",
        types,
        key,
    )
    return None


def _evict_stale_batches(
    pending_batches: Dict[tuple, Dict[str, Message]],
    pending_timestamps: Dict[tuple, float],
    client_channels: Dict[str, Dict[str, Channel]],
    timeout_seconds: float,
) -> None:
    now = time.monotonic()
    stale_keys = [
        key
        for key, ts in pending_timestamps.items()
        if now - ts > timeout_seconds
    ]
    for key in stale_keys:
        batch = pending_batches.pop(key, {})
        pending_timestamps.pop(key, None)
        for client_id, message in batch.items():
            client_channels[client_id]["downlink"].send(
                Message(
                    type="error",
                    sender="split_server",
                    round=message.round,
                    step=message.step,
                    payload={"reason": "split_batch_timeout"},
                )
            )
        logger.warning(
            "SplitServer: evicting stale batch key=%s "
            "(received from %d client(s): %s, timeout=%gs).",
            key,
            len(batch),
            list(batch.keys()),
            timeout_seconds,
        )


def _handle_train_batch(
    batch_msgs: Dict[str, Message],
    model: ServerSideModel,
    criterion: nn.Module,
    device: torch.device,
    client_channels: Dict[str, Dict[str, Channel]],
    parallel: bool = True,
) -> None:
    """
    parallel=True  — single forward/backward over concatenated activations;
                     gradients are sliced back per client.
    parallel=False — sequential forward/backward per client;
                     gradients accumulate in model parameters.
    """
    model.train()

    n_clients = len(batch_msgs)

    if parallel:
        grads_per_client, batch_loss = _forward_parallel(
            batch_msgs,
            model,
            criterion,
            device,
        )
    else:
        grads_per_client, batch_loss = _forward_sequential(
            batch_msgs,
            model,
            criterion,
            device,
            n_clients,
        )

    if grads_per_client is None:
        return None

    for client_id, msg in batch_msgs.items():
        client_channels[client_id]["downlink"].send(
            Message(
                type="gradients",
                sender="split_server",
                round=msg.round,
                step=msg.step,
                payload={"gradients": grads_per_client[client_id]},
            )
        )

    return batch_loss


def _forward_parallel(
    batch_msgs: Dict[str, Message],
    model: ServerSideModel,
    criterion: nn.Module,
    device: torch.device,
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Concatenates activations from all clients → single forward → single backward.
    The gradient tensor is then split back according to each client's original size.
    """
    activations_list, labels_list, sizes, client_order = [], [], [], []

    ordered_clients = sorted(batch_msgs.keys())

    for client_id in ordered_clients:
        msg = batch_msgs[client_id]
        act = msg.payload["activations"].to(device)
        lbl = msg.payload["labels"].to(device).float()
        if lbl.dim() == 1:
            lbl = lbl.unsqueeze(1)
        sizes.append(act.shape[0])
        client_order.append(client_id)
        activations_list.append(act)
        labels_list.append(lbl)

    H = torch.cat(activations_list, dim=0).detach().requires_grad_(True)
    Y_true = torch.cat(labels_list, dim=0)
    outputs = model(H)

    if outputs.shape != Y_true.shape:
        logger.error(
            "SplitServer [parallel]: shape mismatch %s vs %s — aborting batch.",
            outputs.shape,
            Y_true.shape,
        )
        return None

    loss = criterion(outputs, Y_true)
    loss_value = loss.item()
    loss.backward()

    if H.grad is None:
        logger.error(
            "SplitServer [parallel]: H.grad is None — sending zero gradients."
        )
        grad_H = torch.zeros_like(H)
    else:
        grad_H = H.grad.detach()

    grad_splits = torch.split(grad_H, sizes, dim=0)

    return {
        cid: grad.cpu() for cid, grad in zip(client_order, grad_splits)
    }, loss_value


def _forward_sequential(
    batch_msgs: Dict[str, Message],
    model: ServerSideModel,
    criterion: nn.Module,
    device: torch.device,
    n_clients: int,
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Sequential forward/backward for each client individually.
    Gradients accumulate in model parameters — optimizer.step() is called
    once externally after this function returns.
    """
    grads: Dict[str, torch.Tensor] = {}
    total_loss = 0.0

    for client_id, msg in batch_msgs.items():
        act = msg.payload["activations"].to(device)
        lbl = msg.payload["labels"].to(device).float()
        if lbl.dim() == 1:
            lbl = lbl.unsqueeze(1)

        H = act.detach().requires_grad_(True)
        outputs = model(H)

        if outputs.shape != lbl.shape:
            logger.error(
                "SplitServer [sequential]: shape mismatch for client %s "
                "%s vs %s — aborting batch.",
                client_id,
                outputs.shape,
                lbl.shape,
            )
            model.zero_grad()
            return None

        # Divide by n_clients so each client contributes equally within this
        # server batch. Cross-batch averaging happens before optimizer.step().
        loss = criterion(outputs, lbl) / n_clients
        loss_value = loss.item() * n_clients
        total_loss += loss_value / n_clients  # average across clients
        loss.backward()

        if H.grad is None:
            logger.error(
                "SplitServer [sequential]: H.grad is None for client %s "
                "— sending zero gradients.",
                client_id,
            )
            grads[client_id] = torch.zeros_like(act)
        else:
            grads[client_id] = H.grad.detach().cpu()

    return grads, total_loss


def _step_accumulated_gradients(
    parameters,
    optimizer: optim.Optimizer,
    batch_count: int,
) -> None:
    if batch_count <= 0:
        raise ValueError("batch_count must be positive")

    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.div_(batch_count)

    optimizer.step()
    optimizer.zero_grad()


def _handle_eval_single(
    msg: Message,
    client_id: str,
    model: nn.Module,
    device: torch.device,
    client_channels: Dict[str, Dict[str, Channel]],
    all_eval_probs: list,
    all_eval_labels: list,
) -> None:
    model.eval()

    act = msg.payload["activations"].to(device)
    labels = msg.payload["labels"].to(device).float().reshape(-1)

    with torch.no_grad():
        logits = model(act).reshape(-1)
        probs = torch.sigmoid(logits)

    all_eval_probs.append(probs.cpu())
    all_eval_labels.append(labels.cpu())

    client_channels[client_id]["downlink"].send(
        Message(
            type="logits",
            sender="split_server",
            round=msg.round,
            step=msg.step,
            payload={"logits": logits.cpu()},
        )
    )
