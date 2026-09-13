"""Centralized baseline child-process training loop."""

import torch
import torch.multiprocessing as mp
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from ...dataset.dataset import ConflictEmotionalDataset, build_dataset_manifest
from ...logger import logger
from ...model.speech_model import SpeechRecognitionModel
from ...schema import ConfigSchema
from ...utils.persistence import serialize_state_dict
from ...utils.runtime import ignore_parent_interrupts
from ...utils.runtime.resource_metrics import (
    ResourceTracker,
    publish_resource_metric,
)
from ...utils.training import build_optimizer, set_seed
from .data import _pad_feature_batch, _validate_centralized_shapes


def _centralized_training_worker(
    config: ConfigSchema,
    stop_event,
    result_queue: mp.Queue,
    datasets: (
        list[tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]] | None
    ) = None,
    dataset_report_queue=None,
    resource_metrics_queue=None,
) -> None:
    """Train and evaluate one complete model over the combined dataset view."""
    ignore_parent_interrupts()
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
    optimizer = build_optimizer(model.parameters(), owner.model)
    criterion = nn.BCEWithLogitsLoss().to(device)
    generator = torch.Generator().manual_seed(config.training.seed)
    train_loader = DataLoader(
        ConcatDataset(train_parts),
        batch_size=owner.runtime.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=_pad_feature_batch,
    )
    test_loader = DataLoader(
        ConcatDataset(test_parts),
        batch_size=owner.runtime.batch_size,
        collate_fn=_pad_feature_batch,
    )

    for round_idx in range(1, config.training.num_rounds + 1):
        tracker = ResourceTracker("centralized", device)
        model.train()
        losses = []
        sample_count = 0
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
            sample_count += labels.numel()
        if stop_event.is_set():
            break
        logger.info(
            "Centralized round %d loss=%.6f batches=%d",
            round_idx,
            sum(losses) / len(losses) if losses else 0.0,
            len(losses),
        )
        publish_resource_metric(
            resource_metrics_queue,
            tracker.snapshot(
                round_idx=round_idx,
                phase="train",
                samples=sample_count,
                batches=len(losses),
            ),
        )
        if round_idx % config.training.eval_every == 0 or round_idx == (
            config.training.num_rounds
        ):
            tracker.reset()
            accuracy, sample_count = _evaluate_centralized(
                model, test_loader, device
            )
            publish_resource_metric(
                resource_metrics_queue,
                tracker.snapshot(
                    round_idx=round_idx,
                    phase="evaluation",
                    samples=sample_count,
                    batches=len(test_loader),
                ),
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
