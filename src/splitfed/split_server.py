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

# How long (seconds) a partial batch can linger before it is discarded.
_BATCH_TIMEOUT_SECONDS = 30.0

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
            self._process = None
        logger.info("SplitServer process stopped")

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

    logger.info("SplitServer worker ready, serving clients: %s", client_ids)

    current_round = 1
    stats = _RoundStats()

    all_eval_probs = []
    all_eval_labels = []
    try:
        while not stop_event.is_set():
            served_any = False

            for client_id in client_ids:
                uplink: Channel = client_channels[client_id]["uplink"]

                msg = uplink.recv_nowait()
                if msg is None:
                    continue

                if not _validate_message(msg, client_id):
                    continue

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
                elif msg.type == "train_step":
                    key = (msg.round, msg.step)

                    if key not in pending_timestamps:
                        pending_timestamps[key] = time.monotonic()

                    pending_batches[key][client_id] = msg

                    logger.info(
                        "Received '%s' from %s (round=%d step=%d) [%d/%d]",
                        msg.type,
                        client_id,
                        msg.round,
                        msg.step,
                        len(pending_batches[key]),
                        len(client_ids),
                    )

                    if len(pending_batches[key]) == len(client_ids):
                        batch_msgs = pending_batches.pop(key)
                        pending_timestamps.pop(key, None)

                        batch_round = next(iter(batch_msgs.values())).round
                        if batch_round != current_round:
                            stats.log_and_reset(current_round)
                            current_round = batch_round

                        batch_type = _resolve_batch_type(batch_msgs, key)
                        if batch_type is None:
                            continue

                        batch_loss = _handle_train_batch(
                            batch_msgs,
                            model,
                            optimizer,
                            criterion,
                            device,
                            client_channels,
                        )
                        if batch_loss is not None:
                            stats.update(batch_loss)
                else:
                    logger.warning(
                        "SplitServer: unknown msg type '%s' from %s - discarding.",
                        msg.type,
                        client_id,
                    )

            _evict_stale_batches(pending_batches, pending_timestamps)

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
    if msg.type not in ("train_step", "eval_step"):
        logger.warning(
            "SplitServer: unknown message type '%s' from client %s — discarding.",
            msg.type,
            client_id,
        )
        return False

    if not isinstance(msg.payload, dict):
        logger.warning(
            "SplitServer: payload is not a dict (client %s, type %s) — discarding.",
            client_id,
            msg.type,
        )
        return False

    if "activations" not in msg.payload:
        logger.warning(
            "SplitServer: missing 'activations' in payload (client %s) — discarding.",
            client_id,
        )
        return False

    if msg.type == "train_step" and "labels" not in msg.payload:
        logger.warning(
            "SplitServer: missing 'labels' in train payload (client %s) — discarding.",
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

    return True


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
) -> None:
    now = time.monotonic()
    stale_keys = [
        key
        for key, ts in pending_timestamps.items()
        if now - ts > _BATCH_TIMEOUT_SECONDS
    ]
    for key in stale_keys:
        batch = pending_batches.pop(key, {})
        pending_timestamps.pop(key, None)
        logger.warning(
            "SplitServer: evicting stale batch key=%s "
            "(received from %d client(s): %s, timeout=%gs).",
            key,
            len(batch),
            list(batch.keys()),
            _BATCH_TIMEOUT_SECONDS,
        )


def _handle_train_batch(
    batch_msgs: Dict[str, Message],
    model: ServerSideModel,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    client_channels: Dict[str, Dict[str, Channel]],
    accum_steps: int = 4,
    parallel: bool = True,
) -> None:
    """
    parallel=True  — single forward/backward over concatenated activations;
                     gradients are sliced back per client.
    parallel=False — sequential forward/backward per client;
                     gradients accumulate in model parameters,
                     with a single optimizer.step() at the end.
    """
    model.train()

    # Pull step from any message — all messages in the batch share the same step
    current_step = next(iter(batch_msgs.values())).step
    n_clients = len(batch_msgs)

    if parallel:
        grads_per_client, batch_loss = _forward_parallel(
            batch_msgs,
            model,
            criterion,
            device,
            n_clients,
            accum_steps,
        )
    else:
        grads_per_client, batch_loss = _forward_sequential(
            batch_msgs,
            model,
            criterion,
            device,
            n_clients,
            accum_steps,
        )

    if grads_per_client is None:
        optimizer.zero_grad()
        return None

    if current_step % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

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
    n_clients: int,
    accum_steps: int,
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

    loss = criterion(outputs, Y_true) / accum_steps
    loss_value = loss.item() * accum_steps
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
    accum_steps: int,
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

        # Divide by n_clients so each client contributes equally,
        # and by accum_steps to accumulate gradients across multiple batches.
        loss = criterion(outputs, lbl) / (n_clients * accum_steps)
        loss_value = loss.item() * n_clients * accum_steps
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
