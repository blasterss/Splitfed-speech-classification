import queue
from pathlib import Path

import torch
import torch.multiprocessing as mp
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from ..dataset.dataset import ConflictEmotionalDataset, build_dataset_manifest
from ..logger import logger
from ..model.speech_model import SpeechRecognitionModel
from ..schema import ConfigSchema
from ..utils.checkpoint import save_checkpoint
from ..utils.state import deserialize_state_dict, serialize_state_dict
from ..utils.training import set_seed


class CentralizedTrainer:
    """Own the single complete model used by the centralized baseline."""

    def __init__(
        self,
        config: ConfigSchema,
        stop_event=None,
        mp_context=None,
        dataset_report_queue=None,
    ):
        self.config = config
        self._mp_context = mp_context or mp.get_context("spawn")
        self._stop_event = (
            stop_event if stop_event is not None else self._mp_context.Event()
        )
        self._result_queue: mp.Queue = self._mp_context.Queue(maxsize=1)
        self._process: mp.Process | None = None
        self._last_exitcode: int | None = None
        self._last_state_dict: dict | None = None
        self._dataset_report_queue = dataset_report_queue

    def start(self) -> None:
        self._stop_event.clear()
        self._process = self._mp_context.Process(
            target=_centralized_training_worker,
            args=(
                self.config,
                self._stop_event,
                self._result_queue,
                None,
                self._dataset_report_queue,
            ),
            daemon=False,
            name="CentralizedTrainer",
        )
        self._last_exitcode = None
        self._process.start()
        logger.info(
            "Centralized trainer process started (pid=%d)", self._process.pid
        )

    def stop(self) -> None:
        if self._process is not None:
            self._stop_event.set()
            self._process.join(timeout=30)
            if self._process.is_alive():
                self._stop_event.set()
                self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=5)
                if self._process.is_alive():
                    raise RuntimeError(
                        "Centralized trainer remained alive after kill"
                    )
            self._last_exitcode = self._process.exitcode
            self._process = None
        if self._last_state_dict is None and self._last_exitcode == 0:
            try:
                payload = self._result_queue.get(timeout=5)
                self._last_state_dict = deserialize_state_dict(payload)
            except queue.Empty:
                pass

    def wait(self, poll_timeout: float = 0.5) -> None:
        """Drain the result before join so large states cannot block exit."""
        if self._process is None:
            raise RuntimeError("Centralized trainer has not been started")
        while self._last_state_dict is None:
            try:
                payload = self._result_queue.get(timeout=poll_timeout)
                self._last_state_dict = deserialize_state_dict(payload)
            except queue.Empty:
                if not self._process.is_alive():
                    self._process.join(timeout=1)
                    raise RuntimeError(
                        "Centralized trainer failed before producing state "
                        f"(exitcode={self._process.exitcode})"
                    ) from None
        self._process.join(timeout=30)
        if self._process.is_alive():
            raise RuntimeError(
                "Centralized trainer did not exit after publishing state"
            )
        if self._process.exitcode != 0:
            raise RuntimeError(
                "Centralized trainer process failure: "
                f"exitcode={self._process.exitcode}"
            )

    @property
    def exitcode(self) -> int | None:
        if self._process is not None:
            return self._process.exitcode
        return self._last_exitcode

    def get_state_dict(self) -> dict:
        if self._last_state_dict is not None:
            return self._last_state_dict
        try:
            return deserialize_state_dict(self._result_queue.get_nowait())
        except queue.Empty:
            raise RuntimeError(
                "No centralized state_dict is available after training"
            ) from None

    def save(self, path: str | Path) -> None:
        save_path = Path(path) / "centralized_model.pt"
        save_checkpoint(
            save_path,
            mode="centralized",
            model_state_dict=self.get_state_dict(),
        )
        logger.info("Centralized model saved to '%s'", save_path)


def _centralized_training_worker(
    config: ConfigSchema,
    stop_event,
    result_queue: mp.Queue,
    datasets: (
        list[tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]] | None
    ) = None,
    dataset_report_queue=None,
) -> None:
    """Train and evaluate one complete model over the combined dataset view."""
    set_seed(config.training.seed)
    if datasets is None:
        loaded = [
            ConflictEmotionalDataset(client.dataset)
            for client in config.clients
        ]
        if dataset_report_queue is not None:
            for client_config, dataset in zip(
                config.clients, loaded, strict=True
            ):
                dataset_report_queue.put(
                    {
                        "client_id": client_config.client_id,
                        **build_dataset_manifest(
                            client_config.dataset, dataset
                        ),
                    },
                    timeout=5,
                )
        datasets = [
            (dataset.train_dataset, dataset.test_dataset) for dataset in loaded
        ]

    train_parts = [train for train, _ in datasets]
    test_parts = [test for _, test in datasets]
    input_channels = _validate_centralized_shapes(train_parts + test_parts)
    owner = config.clients[0]
    device = torch.device(owner.runtime.device)
    model = SpeechRecognitionModel(
        input_channels=input_channels,
        server_side_model_type="cnn_birnn",
        noise_std=owner.noise.std if owner.noise else 0.0,
        noise_type=owner.noise.type if owner.noise else None,
    ).to(device)
    if owner.model.optimizer.lower() != "adam":
        raise ValueError(
            f"Unsupported centralized optimizer: {owner.model.optimizer}"
        )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=owner.model.learning_rate
    )
    criterion = nn.BCEWithLogitsLoss().to(device)
    generator = torch.Generator().manual_seed(config.training.seed)
    train_loader = DataLoader(
        ConcatDataset(train_parts),
        batch_size=owner.runtime.batch_size,
        shuffle=True,
        generator=generator,
    )
    test_loader = DataLoader(
        ConcatDataset(test_parts), batch_size=owner.runtime.batch_size
    )

    for round_idx in range(1, config.training.num_rounds + 1):
        model.train()
        losses = []
        for features, labels in train_loader:
            if stop_event.is_set():
                break
            features = features.to(device)
            labels = labels.to(device).float().reshape(-1, 1)
            optimizer.zero_grad()
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        if stop_event.is_set():
            break
        logger.info(
            "Centralized round %d loss=%.6f batches=%d",
            round_idx,
            sum(losses) / len(losses) if losses else 0.0,
            len(losses),
        )
        if round_idx % config.training.eval_every == 0 or round_idx == (
            config.training.num_rounds
        ):
            accuracy, sample_count = _evaluate_centralized(
                model, test_loader, device
            )
            logger.info(
                "Centralized round %d evaluation accuracy=%.6f samples=%d",
                round_idx,
                accuracy,
                sample_count,
            )
    state = {key: value.cpu() for key, value in model.state_dict().items()}
    result_queue.put(serialize_state_dict(state), timeout=5)


def _evaluate_centralized(model, test_loader, device) -> tuple[float, int]:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in test_loader:
            logits = model(features.to(device)).reshape(-1)
            predictions = (torch.sigmoid(logits) > 0.5).long().cpu()
            labels = labels.long().reshape(-1)
            correct += (predictions == labels).sum().item()
            total += labels.numel()
    return (correct / total if total else 0.0), total


def _validate_centralized_shapes(datasets: list) -> int:
    shapes = {
        tuple(dataset[0][0].shape) for dataset in datasets if len(dataset) > 0
    }
    if not shapes:
        raise ValueError("Centralized mode requires at least one sample")
    if len(shapes) != 1:
        raise ValueError(
            f"Centralized datasets require identical feature shapes: {shapes}"
        )
    return next(iter(shapes))[0]
