from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from ...dataset.dataset import ConflictEmotionalDataset, build_dataset_manifest
from ...logger import logger
from ...model.client_side_model import ClientSideModel
from ...model.speech_model import SpeechRecognitionModel
from ...schema import (
    ClientConfig,
    ServerModelScope,
    TrainingMode,
    WorkloadPolicy,
)
from ...transport.base import Channel, Message
from .evaluation import evaluate_client
from .protocol import (
    _extract_payload,
    _validate_global_update,
    _validate_round_ack,
)

logger = logger.getChild("Client")


def _cpu_state_dict_snapshot(
    model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


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
        split_server_scope: ServerModelScope | None = None,
    ):
        self.cfg = cfg
        self.client_id = cfg.client_id
        self.device = torch.device(cfg.runtime.device)
        self.mode = mode
        self.split_server_scope = split_server_scope
        self.metrics_path = (
            Path(metrics_path) if metrics_path is not None else None
        )

        # ==== DATASET ====
        self.dataset = ConflictEmotionalDataset(cfg.dataset)
        self.dataset_manifest = build_dataset_manifest(
            cfg.dataset, self.dataset
        )

        _pin = self.device.type == "cuda"

        g = torch.Generator()
        g.manual_seed(self.cfg.runtime.seed)

        self.train_loader = DataLoader(
            self.dataset.train_dataset,
            batch_size=cfg.runtime.batch_size,
            shuffle=True,
            pin_memory=_pin,
            generator=g,
            drop_last=cfg.runtime.drop_last,
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
        for step, (x, y) in enumerate(self._round_batches(), start=1):
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

            if self._round_is_complete(step):
                break

        round_end = Message(
            type="round_end",
            sender=self.client_id,
            round=round,
            step=last_step + 1,
            payload={"dataset_size": len(self.dataset.train_dataset)},
        )
        self.to_server.send(round_end)
        if (
            self.mode is TrainingMode.splitfed
            and self.split_server_scope is ServerModelScope.personalized
        ):
            _validate_round_ack(
                self.from_server.recv(),
                client_id=self.client_id,
                round_idx=round,
                step=round_end.step,
                request_id=round_end.request_id,
            )

    def _train_federated_round(self) -> None:
        for step, (x, y) in enumerate(self._round_batches(), start=1):
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
            if self._round_is_complete(step):
                break

    def _round_batches(self):
        """Yield the batches selected by the configured workload policy."""
        policy = getattr(
            self.cfg.runtime,
            "workload_policy",
            WorkloadPolicy.max_steps_v1,
        )
        if policy is not WorkloadPolicy.fixed_steps_v1:
            yield from self.train_loader
            return

        if len(self.train_loader) == 0:
            raise RuntimeError(
                f"Client {self.client_id}: fixed_steps_v1 requires a "
                "non-empty train loader"
            )

        loader_iterator = iter(self.train_loader)
        for _ in range(self.cfg.runtime.local_steps):
            try:
                yield next(loader_iterator)
            except StopIteration:
                loader_iterator = iter(self.train_loader)
                yield next(loader_iterator)

    def _round_is_complete(self, step: int) -> bool:
        policy = getattr(
            self.cfg.runtime,
            "workload_policy",
            WorkloadPolicy.max_steps_v1,
        )
        return (
            policy
            in (
                WorkloadPolicy.max_steps_v1,
                WorkloadPolicy.fixed_steps_v1,
            )
            and step >= self.cfg.runtime.local_steps
        )

    def federative_aggregate(self, round: int) -> None:
        """Exchange local parameters for validated global weights."""

        msg = Message(
            type="client_update",
            sender=self.client_id,
            round=round,
            step=1,
            payload={
                "state_dict": _cpu_state_dict_snapshot(self.model),
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

    def evaluate(self, round: int = 0) -> dict[str, float | int]:
        return evaluate_client(self, round)
