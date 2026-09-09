"""Cross-corpus evaluation for validated complete-model checkpoints."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..dataset.dataset import ConflictEmotionalDataset, build_dataset_manifest
from ..model.speech_model import SpeechRecognitionModel
from ..schema import ConfigSchema
from ..splitfed.centralized.data import _pad_feature_batch
from ..utils.config import save_yaml
from ..utils.persistence import load_checkpoint
from .metrics import (
    BINARY_METRICS_SCHEMA_VERSION,
    evaluate_binary_model,
)

CHECKPOINT_EVALUATION_SCHEMA_VERSION = 1
CORPUS_AGGREGATE_METRICS = (
    "accuracy",
    "anger_f1",
    "macro_f1",
    "precision",
    "recall",
    "uar",
    "pr_auc",
)


def evaluate_model_on_corpora(
    model: nn.Module,
    datasets: dict[str, ConflictEmotionalDataset],
    *,
    batch_size: int,
    device: torch.device,
    dataset_configs: dict | None = None,
) -> dict:
    """Evaluate one complete model on each corpus test actor view.

    Each test view retains the normalization statistics computed from its own
    training actors, matching the normalization used by joint training.
    """
    rows = []
    manifests = {}
    model = model.to(device)
    for corpus, dataset in datasets.items():
        if dataset_configs is not None:
            manifests[corpus] = build_dataset_manifest(
                dataset_configs[corpus], dataset
            )
        loader = DataLoader(
            dataset.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=_pad_feature_batch,
        )
        rows.append(
            {
                "eval_corpus": corpus,
                **evaluate_binary_model(model, loader, device),
            }
        )

    summary = {
        "schema_version": CHECKPOINT_EVALUATION_SCHEMA_VERSION,
        "metrics_schema_version": BINARY_METRICS_SCHEMA_VERSION,
        "rows": rows,
        "dataset_manifests": manifests,
        "macro": {
            metric: _mean(rows, metric) for metric in CORPUS_AGGREGATE_METRICS
        },
        "worst_corpus": {
            metric: _minimum(rows, metric)
            for metric in CORPUS_AGGREGATE_METRICS
        },
    }
    return summary


def evaluate_full_model_checkpoint(
    config: ConfigSchema,
    checkpoint_path: str | Path,
    *,
    expected_mode: str,
    artifact_root: str | Path | None = None,
    datasets: dict[str, ConflictEmotionalDataset] | None = None,
    model_factory=None,
) -> dict:
    """Load, validate, and evaluate a complete-model checkpoint."""
    if not config.clients:
        raise ValueError("Checkpoint evaluation requires at least one client")
    loaded = datasets or {
        client.dataset.name.value: ConflictEmotionalDataset(client.dataset)
        for client in config.clients
    }
    configured_names = [client.dataset.name.value for client in config.clients]
    if set(loaded) != set(configured_names):
        raise ValueError("datasets must match configured corpus names")
    input_channels = int(
        loaded[configured_names[0]].train_dataset.data.shape[1]
    )
    factory = model_factory or _default_model_factory
    model = factory(input_channels, config.clients[0]).to("cpu")
    state = load_checkpoint(
        checkpoint_path,
        expected_mode=expected_mode,
        expected_state_dict=model.state_dict(),
    )
    model.load_state_dict(state)
    summary = evaluate_model_on_corpora(
        model,
        loaded,
        batch_size=config.clients[0].runtime.batch_size,
        device=torch.device(config.clients[0].runtime.device),
        dataset_configs={
            client.dataset.name.value: client.dataset
            for client in config.clients
        },
    )
    summary.update(
        {
            "method": expected_mode,
            "checkpoint": str(Path(checkpoint_path).absolute()),
            "seed": config.training.seed,
            "normalization_policy": "corpus_train_stats_v1",
        }
    )
    if artifact_root is not None:
        destination = Path(artifact_root)
        destination.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summary["rows"]).to_csv(
            destination / "cross_corpus_metrics.csv", index=False
        )
        save_yaml(destination / "cross_corpus_summary.yaml", summary)
    return summary


def _default_model_factory(input_channels: int, client_config) -> nn.Module:
    return SpeechRecognitionModel(
        input_channels=input_channels,
        server_side_model_type="cnn_birnn",
        noise_std=(client_config.noise.std if client_config.noise else 0.0),
        noise_type=(client_config.noise.type if client_config.noise else None),
    )


def _numeric_values(rows: list[dict], metric: str) -> list[float]:
    if not rows or any(row.get(metric) is None for row in rows):
        return []
    return [float(row[metric]) for row in rows]


def _mean(rows: list[dict], metric: str) -> float | None:
    values = _numeric_values(rows, metric)
    return sum(values) / len(values) if values else None


def _minimum(rows: list[dict], metric: str) -> float | None:
    values = _numeric_values(rows, metric)
    return min(values) if values else None
