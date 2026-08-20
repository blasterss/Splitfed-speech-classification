import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

from typing import Dict, Optional

from ..dataset.dataset import ConflictEmotionalDataset
from ..model.client_side_model import ClientSideModel
from ..transport.base import Channel, Message
from ..schema import ClientConfig, TrainingConfig
from ..utils.training import set_seed

from ..logger import logger

from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import WeightedRandomSampler

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
    ):
        self.cfg = cfg
        self.client_id = cfg.client_id
        self.device = torch.device(cfg.runtime.device)

        # ==== DATASET ====
        self.dataset = ConflictEmotionalDataset(cfg.dataset)

        _pin = self.device.type == "cuda"

        sample_weights = self.dataset.get_sample_weights()

        # NOTE: sampler is defined but currently not used in DataLoader
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

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
                "Client %s: test dataset is empty — evaluate() will return zero metrics.",
                cfg.client_id,
            )

        # ==== CLIENT MODEL ====
        self.model = ClientSideModel(
            input_channels=self.dataset.train_dataset.data.shape[1],
            noise_std=cfg.noise.std if cfg.noise else 0.0,
            noise_type=cfg.noise.type if cfg.noise else None,
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
            )

            if grad is None:
                logger.error(
                    "Client %s: skipping backward pass r=%d s=%d — invalid server response.",
                    self.client_id,
                    round,
                    step,
                )
                continue

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

    def federative_aggregate(self, round: int) -> None:
        """
        Sends local model parameters to federated server and receives updated global weights.
        """

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
            response, self.model.state_dict(), round
        )
        self.model.load_state_dict(state_dict)

        logger.info(
            "Client %s ← fed_server: global model received r=%d",
            self.client_id,
            round,
        )

    @torch.no_grad()
    def evaluate(self, round: int = 0) -> Dict[str, float]:
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
            return {"accuracy": 0.0, "f1": 0.0}

        self.model.eval()

        correct = 0
        total = 0

        all_probs = []
        all_preds = []
        all_labels = []

        for step, (x, y) in enumerate(self.test_loader, start=1):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            activations = self.model(x)

            # Send activations for server-side inference
            self.to_server.send(
                Message(
                    type="eval_step",
                    sender=self.client_id,
                    round=round,
                    step=step,
                    payload={
                        "activations": activations.cpu(),
                        "labels": y.cpu(),
                    },
                )
            )

            response = self.from_server.recv()

            logits = _extract_payload(
                response,
                "logits",
                "logits",
                self.client_id,
                round,
                step,
            )

            if logits is None:
                logger.error(
                    "Client %s: skipping eval step r=%d s=%d — invalid server response.",
                    self.client_id,
                    round,
                    step,
                )
                continue

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
            return {"accuracy": 0.0, "f1": 0.0}

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

        df.to_csv(
            f"experiments/results/Client{self.client_id}_eval.csv", index=False
        )

        f1 = f1_score(all_labels_np, all_preds_np, average="binary")
        precision = precision_score(
            all_labels_np, all_preds_np, average="binary"
        )
        recall = recall_score(all_labels_np, all_preds_np, average="binary")

        return {
            "accuracy": correct / total,
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
        }


def _extract_payload(
    response: Optional[Message],
    key: str,
    expected_type: str,
    client_id: str,
    round: int,
    step: int,
) -> Optional[object]:
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

    if (
        response.sender != "split_server"
        or response.type != expected_type
        or response.round != round
        or response.step != step
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
) -> dict:
    if response.sender != "fed_server":
        raise ValueError("Invalid global update sender")
    if response.type != "global_update":
        raise ValueError("Invalid global update type")
    if response.round != round:
        raise ValueError("Invalid global update round")
    if response.step != 1:
        raise ValueError("Invalid global update step")
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

            if round_idx % training_cfg.fed_every == 0:
                client.federative_aggregate(round_idx)

        logger.info(
            "Client %s: training finished — waiting at eval_barrier.",
            cfg.client_id,
        )
        eval_barrier.wait(timeout=training_cfg.barrier_timeout_sec)

        logger.info(
            "Client %s: eval_barrier passed — starting evaluation.",
            cfg.client_id,
        )
        metrics = client.evaluate(round=last_round)
        logger.info("Client %s final metrics: %s", client.client_id, metrics)

    except BaseException as exc:
        stop_event.set()
        _abort_barriers((ready_barrier, eval_barrier))
        logger.error(
            "Client %s crashed: %s", cfg.client_id, exc, exc_info=True
        )
        raise


def _abort_barriers(barriers: tuple[object, ...]) -> None:
    for barrier in barriers:
        try:
            barrier.abort()
        except Exception as exc:
            logger.debug("Client could not abort lifecycle barrier: %s", exc)
