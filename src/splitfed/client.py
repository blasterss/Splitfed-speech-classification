from pathlib import Path

import torch
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from ..dataset.dataset import ConflictEmotionalDataset
from ..logger import logger
from ..model.client_side_model import ClientSideModel
from ..model.speech_model import SpeechRecognitionModel
from ..schema import ClientConfig, TrainingConfig, TrainingMode
from ..transport.base import Channel, Message
from ..utils.training import set_seed

logger = logger.getChild("Client")


class Client:
    """
    Client for Split Learning / Federated Learning.

    Responsible for:
        - Local data loading
        - Client-side forward pass
        - Sending activations to server
        - Receiving gradients from server
        - Optional federated weight aggregation
        - Evaluation
    """

    def __init__(
        self,
        cfg: ClientConfig,
        split_uplink_channel: Channel,
        split_downlink_channel: Channel,
        fed_uplink_channel: Channel,
        fed_downlink_channel: Channel,
        mode: TrainingMode = TrainingMode.splitfed,
        metrics_path: str | Path | None = None,
    ):
        self.cfg = cfg
        self.client_id = cfg.client_id
        self.device = torch.device(cfg.runtime.device)
        self.mode = mode
        self.metrics_path = (
            Path(metrics_path) if metrics_path is not None else None
        )

        # ==== DATASET ====
        self.dataset = ConflictEmotionalDataset(cfg.dataset)

        _pin = self.device.type == "cuda"

        g = torch.Generator()
        g.manual_seed(self.cfg.runtime.seed)

        self.train_loader = DataLoader(
            self.dataset.train_dataset,
            batch_size=cfg.runtime.batch_size,
            shuffle=True,
            pin_memory=_pin,
            generator=g,
        )

        self.test_loader = DataLoader(
            self.dataset.test_dataset,
            batch_size=cfg.runtime.batch_size,
            shuffle=False,
            pin_memory=_pin,
            generator=g,
        )

        if len(self.dataset.test_dataset) == 0:
            logger.warning(
                "Client %s: test dataset is empty — evaluate() will "
                "return zero metrics.",
                cfg.client_id,
            )

        # ==== CLIENT MODEL ====
        input_channels = self.dataset.train_dataset.data.shape[1]
        if mode is TrainingMode.federated:
            self.model = SpeechRecognitionModel(
                input_channels=input_channels,
                server_side_model_type="cnn_birnn",
                noise_std=cfg.noise.std if cfg.noise else 0.0,
                noise_type=cfg.noise.type if cfg.noise else None,
            ).to(self.device)
            self.criterion = torch.nn.BCEWithLogitsLoss().to(self.device)
        else:
            self.model = ClientSideModel(
                input_channels=input_channels,
                noise=cfg.noise is not None,
                noise_std=cfg.noise.std if cfg.noise else 0.0,
                noise_type=cfg.noise.type if cfg.noise else "gauss",
            ).to(self.device)

        self.optimizer = self._build_optimizer()

        # ==== COMMUNICATION CHANNELS ====
        self.to_server = split_uplink_channel
        self.from_server = split_downlink_channel
        self.agg_to_server = fed_uplink_channel
        self.agg_from_server = fed_downlink_channel

    def _build_optimizer(self) -> optim.Optimizer:
        """Build optimizer for client-side model."""
        if self.cfg.model.optimizer.lower() == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.cfg.model.learning_rate,
            )
        raise ValueError(f"Unsupported optimizer: {self.cfg.model.optimizer}")

    def train_one_round(self, round: int) -> None:
        """
        Executes one local training round.

        Workflow:
            1. Forward pass on client model
            2. Send activations to server
            3. Receive gradients from server
            4. Backpropagate locally
        """

        self.model.train()

        if self.mode is TrainingMode.federated:
            self._train_federated_round()
            return

        last_step = 0
        for step, (x, y) in enumerate(self.train_loader, start=1):
            last_step = step
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            activations = self.model(x)

            msg = Message(
                type="train_step",
                sender=self.client_id,
                round=round,
                step=step,
                payload={
                    "activations": activations.detach().cpu(),
                    "labels": y.cpu(),
                },
            )

            logger.info(
                "Client %s → split_server: train_step r=%d s=%d",
                self.client_id,
                round,
                step,
            )

            self.to_server.send(msg)

            response = self.from_server.recv()

            grad = _extract_payload(
                response,
                "gradients",
                "gradients",
                self.client_id,
                round,
                step,
                msg.request_id,
            )

            if grad is None:
                raise RuntimeError(
                    f"Client {self.client_id}: invalid split gradient "
                    f"response for round={round} step={step}"
                )

            activations.backward(grad.to(self.device))
            self.optimizer.step()
            self.optimizer.zero_grad()

            if step >= self.cfg.runtime.local_steps:
                break

        self.to_server.send(
            Message(
                type="round_end",
                sender=self.client_id,
                round=round,
                step=last_step + 1,
                payload={},
            )
        )

    def _train_federated_round(self) -> None:
        for step, (x, y) in enumerate(self.train_loader, start=1):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True).float().reshape(-1, 1)
            logits = self.model(x)
            if logits.shape != y.shape:
                raise RuntimeError(
                    f"Federated client {self.client_id}: logits/labels shape "
                    f"mismatch {logits.shape} != {y.shape}"
                )
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            if step >= self.cfg.runtime.local_steps:
                break

    def federative_aggregate(self, round: int) -> None:
        """Exchange local parameters for validated global weights."""

        msg = Message(
            type="client_update",
            sender=self.client_id,
            round=round,
            step=1,
            payload={
                "state_dict": self.model.state_dict(),
                "dataset_size": len(self.dataset.train_dataset),
            },
        )

        logger.info(
            "Client %s → fed_server: client_update r=%d", self.client_id, round
        )

        self.agg_to_server.send(msg)

        response = self.agg_from_server.recv()
        state_dict = _validate_global_update(
            response,
            self.model.state_dict(),
            round,
            msg.request_id,
        )
        self.model.load_state_dict(state_dict)

        logger.info(
            "Client %s ← fed_server: global model received r=%d",
            self.client_id,
            round,
        )

    @torch.no_grad()
    def evaluate(self, round: int = 0) -> dict[str, float | int]:
        """
        Evaluates model on local test set.

        Returns:
            accuracy, f1, precision, recall
        """

        if len(self.dataset.test_dataset) == 0:
            logger.warning(
                "Client %s: test dataset is empty — returning zero metrics.",
                self.client_id,
            )
            return _empty_metrics()

        self.model.eval()

        correct = 0
        total = 0

        all_probs = []
        all_preds = []
        all_labels = []

        for step, (x, y) in enumerate(self.test_loader, start=1):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            if self.mode is TrainingMode.federated:
                logits = self.model(x).reshape(-1)
            else:
                activations = self.model(x)

                # Send activations for server-side inference
                request = Message(
                    type="eval_step",
                    sender=self.client_id,
                    round=round,
                    step=step,
                    payload={
                        "activations": activations.cpu(),
                        "labels": y.cpu(),
                    },
                )
                self.to_server.send(request)
                response = self.from_server.recv()

                logits = _extract_payload(
                    response,
                    "logits",
                    "logits",
                    self.client_id,
                    round,
                    step,
                    request.request_id,
                )

                if logits is None:
                    raise RuntimeError(
                        f"Client {self.client_id}: invalid split evaluation "
                        f"response for round={round} step={step}"
                    )

                logits = logits.to(self.device).reshape(-1)

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()

            y = y.reshape(-1)

            correct += (preds == y).sum().item()
            total += y.size(0)

            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

        if total == 0:
            logger.warning("Client %s: no samples evaluated.", self.client_id)
            return _empty_metrics()

        all_preds_np = torch.cat(all_preds).numpy()
        all_labels_np = torch.cat(all_labels).numpy()
        all_probs_np = torch.cat(all_probs).numpy()

        import pandas as pd

        df = pd.DataFrame(
            {
                "probs": all_probs_np,
                "preds": all_preds_np,
                "labels": all_labels_np,
            }
        )

        metrics_path = getattr(self, "metrics_path", None)
        if metrics_path is not None:
            metrics_path.mkdir(parents=True, exist_ok=True)
            df.to_csv(
                metrics_path
                / f"Client{self.client_id}_round_{round}_eval.csv",
                index=False,
            )

        f1 = f1_score(
            all_labels_np, all_preds_np, average="binary", zero_division=0
        )
        precision = precision_score(
            all_labels_np,
            all_preds_np,
            average="binary",
            zero_division=0,
        )
        recall = recall_score(
            all_labels_np, all_preds_np, average="binary", zero_division=0
        )

        return {
            "accuracy": correct / total,
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "num_samples": total,
            "num_positive_labels": int(all_labels_np.sum()),
            "num_positive_predictions": int(all_preds_np.sum()),
        }


def _empty_metrics() -> dict[str, float | int]:
    return {
        "accuracy": 0.0,
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "num_samples": 0,
        "num_positive_labels": 0,
        "num_positive_predictions": 0,
    }


def _extract_payload(
    response: Message | None,
    key: str,
    expected_type: str,
    client_id: str,
    round: int,
    step: int,
    expected_request_id: str,
) -> object | None:
    """
    Safely extracts a tensor from server response payload.
    """

    if response is None:
        logger.error(
            "Client %s: received None response from server (r=%d s=%d).",
            client_id,
            round,
            step,
        )
        return None

    try:
        response.validate_for_receive()
    except (ValueError, TimeoutError) as exc:
        logger.error(
            "Client %s: invalid response deadline: %s", client_id, exc
        )
        return None

    if (
        response.sender != "split_server"
        or response.type != expected_type
        or response.round != round
        or response.step != step
        or response.request_id != expected_request_id
    ):
        logger.error(
            "Client %s: uncorrelated server response "
            "(expected type=%s r=%d s=%d, got sender=%s type=%s r=%d s=%d).",
            client_id,
            expected_type,
            round,
            step,
            response.sender,
            response.type,
            response.round,
            response.step,
        )
        return None

    if not isinstance(getattr(response, "payload", None), dict):
        logger.error(
            "Client %s: invalid response payload type (r=%d s=%d, type=%s).",
            client_id,
            round,
            step,
            response.type,
        )
        return None

    if key not in response.payload:
        logger.error(
            "Client %s: missing key '%s' in server response (r=%d s=%d).",
            client_id,
            key,
            round,
            step,
        )
        return None

    tensor = response.payload[key]

    if not isinstance(tensor, torch.Tensor):
        logger.error(
            "Client %s: payload['%s'] is not a Tensor (r=%d s=%d, got %s).",
            client_id,
            key,
            round,
            step,
            type(tensor).__name__,
        )
        return None

    return tensor


def _validate_global_update(
    response: Message,
    expected_state: dict,
    round: int,
    expected_request_id: str,
) -> dict:
    response.validate_for_receive()
    if response.sender != "fed_server":
        raise ValueError("Invalid global update sender")
    if response.type != "global_update":
        raise ValueError("Invalid global update type")
    if response.round != round:
        raise ValueError("Invalid global update round")
    if response.step != 1:
        raise ValueError("Invalid global update step")
    if response.request_id != expected_request_id:
        raise ValueError("Invalid global update request_id")
    if not isinstance(response.payload, dict):
        raise ValueError("Invalid global update payload")
    if response.payload.keys() != expected_state.keys():
        raise ValueError("Invalid global update state keys")

    for key, expected in expected_state.items():
        value = response.payload[key]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Invalid global update tensor for {key}")
        if value.shape != expected.shape:
            raise ValueError(f"Invalid global update shape for {key}")
        if value.dtype != expected.dtype:
            raise ValueError(f"Invalid global update dtype for {key}")

    return response.payload


def _client_worker(
    cfg: ClientConfig,
    training_cfg: TrainingConfig,
    split_uplink: Channel,
    split_downlink: Channel,
    fed_uplink: Channel,
    fed_downlink: Channel,
    stop_event,
    ready_barrier,
    eval_barrier,
    metrics_path=None,
) -> None:
    """
    Persistent client process.

    Synchronization design:

    1. ready_barrier:
        Ensures all clients finish dataset loading before training starts.

    2. eval_barrier:
        Ensures all clients finish training before evaluation begins,
        preventing server deadlocks due to missing eval streams.
    """

    try:
        set_seed(cfg.runtime.seed + int(cfg.client_id))

        client = Client(
            cfg=cfg,
            split_uplink_channel=split_uplink,
            split_downlink_channel=split_downlink,
            fed_uplink_channel=fed_uplink,
            fed_downlink_channel=fed_downlink,
            mode=training_cfg.mode,
            metrics_path=metrics_path,
        )

        logger.info(
            "Client %s: dataset loaded — waiting at ready_barrier.",
            cfg.client_id,
        )
        ready_barrier.wait(timeout=training_cfg.barrier_timeout_sec)

        logger.info(
            "Client %s: ready_barrier passed — starting training.",
            cfg.client_id,
        )

        last_round = 0
        for round_idx in range(1, training_cfg.num_rounds + 1):
            if stop_event.is_set():
                return

            last_round = round_idx
            client.train_one_round(round_idx)

            if (
                training_cfg.mode
                in (TrainingMode.federated, TrainingMode.splitfed)
                and round_idx % training_cfg.fed_every == 0
            ):
                client.federative_aggregate(round_idx)
            if _should_evaluate(
                round_idx,
                training_cfg.num_rounds,
                training_cfg.eval_every,
            ):
                _evaluate_at_barrier(
                    client,
                    round_idx,
                    eval_barrier,
                    training_cfg.barrier_timeout_sec,
                )

        if last_round == 0:
            _evaluate_at_barrier(
                client,
                0,
                eval_barrier,
                training_cfg.barrier_timeout_sec,
            )

    except BaseException as exc:
        stop_event.set()
        _abort_barriers((ready_barrier, eval_barrier))
        logger.error(
            "Client %s crashed: %s", cfg.client_id, exc, exc_info=True
        )
        raise


def _should_evaluate(round_idx: int, num_rounds: int, eval_every: int) -> bool:
    return round_idx % eval_every == 0 or round_idx == num_rounds


def _evaluate_at_barrier(
    client: Client,
    round_idx: int,
    eval_barrier,
    timeout: float,
) -> None:
    logger.info(
        "Client %s: waiting to evaluate round %d", client.client_id, round_idx
    )
    eval_barrier.wait(timeout=timeout)
    metrics = client.evaluate(round=round_idx)
    logger.info(
        "Client %s round %d metrics: %s",
        client.client_id,
        round_idx,
        metrics,
    )
    eval_barrier.wait(timeout=timeout)


def _abort_barriers(barriers: tuple[object, ...]) -> None:
    for barrier in barriers:
        try:
            barrier.abort()
        except Exception as exc:
            logger.debug("Client could not abort lifecycle barrier: %s", exc)
