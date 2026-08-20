import queue
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

from ..logger import logger
from ..model.server_side_model import ServerSideModel
from ..schema import ServerModelScope, SplitServerConfig
from ..transport.base import Channel, Message
from ..utils.checkpoint import save_checkpoint
from ..utils.state import deserialize_state_dict, serialize_state_dict
from ..utils.training import set_seed
from ..utils.training_stats import _RoundStats

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


def _build_personalized_models(
    client_ids: list,
    config: SplitServerConfig,
    device: torch.device,
) -> tuple[dict, dict]:
    """Create independently initialized model and optimizer ownership."""
    models = {}
    optimizers = {}
    for client_id in client_ids:
        set_seed(config.seed)
        model = ServerSideModel(model_type="cnn_birnn").to(device)
        models[client_id] = model
        optimizers[client_id] = optim.Adam(
            model.parameters(), lr=config.model.learning_rate
        )
        optimizers[client_id].zero_grad()
    return models, optimizers


def _split_server_worker_personalized(
    config: SplitServerConfig,
    client_channels: dict[str, dict[str, Channel]],
    stop_event,
    result_queue: mp.Queue,
) -> None:
    """Serve each client with an isolated model and optimizer."""
    set_seed(config.seed)
    device = torch.device(config.model.device)
    client_ids = list(client_channels)
    models, optimizers = _build_personalized_models(client_ids, config, device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(config.model.pos_weight, device=device)
    ).to(device)
    accumulated_batches = {client_id: 0 for client_id in client_ids}
    completed_steps = {client_id: set() for client_id in client_ids}
    stats = {client_id: _RoundStats() for client_id in client_ids}

    logger.info(
        "Personalized SplitServer worker ready, serving clients: %s",
        client_ids,
    )
    try:
        while not stop_event.is_set():
            served_any = False
            for client_id in client_ids:
                msg = client_channels[client_id]["uplink"].recv_nowait()
                if msg is None:
                    continue
                if not _validate_message(msg, client_id):
                    raise ValueError(
                        f"Invalid split message from client {client_id}"
                    )
                served_any = True
                correlation = (msg.type, msg.round, msg.step)
                if correlation in completed_steps[client_id]:
                    raise ValueError(
                        f"Duplicate split message from client {client_id} "
                        f"for key {correlation}"
                    )
                completed_steps[client_id].add(correlation)

                if msg.type == "train_step":
                    loss = _handle_train_batch(
                        {client_id: msg},
                        models[client_id],
                        criterion,
                        device,
                        client_channels,
                    )
                    if loss is not None:
                        stats[client_id].update(loss)
                        accumulated_batches[client_id] += 1
                    if (
                        accumulated_batches[client_id]
                        == config.model.gradient_accumulation_steps
                    ):
                        _step_accumulated_gradients(
                            models[client_id].parameters(),
                            optimizers[client_id],
                            accumulated_batches[client_id],
                        )
                        accumulated_batches[client_id] = 0
                elif msg.type == "eval_step":
                    _handle_eval_single(
                        msg,
                        client_id,
                        models[client_id],
                        device,
                        client_channels,
                        [],
                        [],
                    )
                elif msg.type == "round_end":
                    if accumulated_batches[client_id]:
                        _step_accumulated_gradients(
                            models[client_id].parameters(),
                            optimizers[client_id],
                            accumulated_batches[client_id],
                        )
                        accumulated_batches[client_id] = 0
                    logger.info(
                        "Personalized SplitServer metrics for client %s",
                        client_id,
                    )
                    stats[client_id].log_and_reset(msg.round)

            if not served_any:
                stop_event.wait(timeout=0.001)
    finally:
        for client_id in client_ids:
            if accumulated_batches[client_id]:
                _step_accumulated_gradients(
                    models[client_id].parameters(),
                    optimizers[client_id],
                    accumulated_batches[client_id],
                )
        state = {
            client_id: {
                key: value.cpu()
                for key, value in models[client_id].state_dict().items()
            }
            for client_id in client_ids
        }
        try:
            result_queue.put_nowait(serialize_state_dict(state))
        except queue.Full:
            logger.warning(
                "Personalized SplitServer result queue was already full"
            )


def _split_server_worker_batch(
    config: SplitServerConfig,
    client_channels: dict[str, dict[str, Channel]],
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
    pending_batches: dict[tuple, dict[str, Message]] = defaultdict(dict)
    pending_timestamps: dict[tuple, float] = {}
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
            "SplitServer: unknown message type '%s' from client %s — "
            "discarding.",
            msg.type,
            client_id,
        )
        return False

    if msg.round <= 0 or msg.step <= 0:
        logger.warning(
            "SplitServer: invalid correlation from client %s "
            "(round=%d step=%d)",
            client_id,
            msg.round,
            msg.step,
        )
        return False

    if not isinstance(msg.payload, dict):
        logger.warning(
            "SplitServer: payload is not a dict (client %s, type %s) — "
            "discarding.",
            client_id,
            msg.type,
        )
        return False

    if msg.type == "round_end":
        return True

    if "activations" not in msg.payload:
        logger.warning(
            "SplitServer: missing 'activations' in payload (client %s) — "
            "discarding.",
            client_id,
        )
        return False

    if "labels" not in msg.payload:
        logger.warning(
            "SplitServer: missing 'labels' in payload (client %s) — "
            "discarding.",
            client_id,
        )
        return False

    act = msg.payload["activations"]
    if not isinstance(act, torch.Tensor) or act.dim() < 1:
        logger.warning(
            "SplitServer: 'activations' is not a valid tensor "
            "(client %s) — discarding.",
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
    batch: dict[str, Message],
    client_ids: set,
    completed_clients: set,
) -> bool:
    missing_clients = client_ids - set(batch)
    return missing_clients <= completed_clients


def _store_pending_batch(
    pending_batches: dict[tuple, dict[str, Message]],
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
    batch_msgs: dict[str, Message], key: tuple
) -> str | None:
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
    pending_batches: dict[tuple, dict[str, Message]],
    pending_timestamps: dict[tuple, float],
    client_channels: dict[str, dict[str, Channel]],
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
    batch_msgs: dict[str, Message],
    model: ServerSideModel,
    criterion: nn.Module,
    device: torch.device,
    client_channels: dict[str, dict[str, Channel]],
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
    batch_msgs: dict[str, Message],
    model: ServerSideModel,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, torch.Tensor] | None:
    """
    Concatenates activations for one forward/backward pass. The gradient tensor
    is split back according to each client's original batch size.
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
            "SplitServer [parallel]: shape mismatch %s vs %s — "
            "aborting batch.",
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
        cid: grad.cpu()
        for cid, grad in zip(client_order, grad_splits, strict=True)
    }, loss_value


def _forward_sequential(
    batch_msgs: dict[str, Message],
    model: ServerSideModel,
    criterion: nn.Module,
    device: torch.device,
    n_clients: int,
) -> dict[str, torch.Tensor] | None:
    """
    Sequential forward/backward for each client individually.
    Gradients accumulate in model parameters — optimizer.step() is called
    once externally after this function returns.
    """
    grads: dict[str, torch.Tensor] = {}
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
    client_channels: dict[str, dict[str, Channel]],
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
