"""Local-only training with a train-corpus by evaluation-corpus matrix."""

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader

from ..dataset.dataset import (
    ConflictEmotionalDataset,
    EmotionalDataset,
    build_dataset_manifest,
)
from ..model.speech_model import SpeechRecognitionModel
from ..schema import ConfigSchema, TrainingMode
from ..utils.config import read_yaml, save_yaml
from ..utils.persistence import save_checkpoint
from ..utils.training import set_seed

LOCAL_CROSS_CORPUS_SCHEMA_VERSION = 1
NORMALIZATION_POLICY = "train_corpus_stats_v1"
ModelFactory = Callable[[int, object], nn.Module]


def run_local_cross_corpus(
    config: ConfigSchema,
    *,
    artifact_root: str | Path | None = None,
    rounds: int | None = None,
    datasets: dict[str, ConflictEmotionalDataset] | None = None,
    model_factory: ModelFactory | None = None,
) -> dict:
    """Train one isolated model per corpus and evaluate every corpus pair."""
    _validate_local_config(config)
    training_rounds = config.training.num_rounds if rounds is None else rounds
    if training_rounds <= 0:
        raise ValueError("rounds must be positive")

    corpus_configs = {
        client.dataset.name.value: client for client in config.clients
    }
    loaded = datasets or {
        corpus: ConflictEmotionalDataset(client.dataset)
        for corpus, client in corpus_configs.items()
    }
    if set(loaded) != set(corpus_configs):
        raise ValueError(
            "datasets must contain exactly the configured corpus names"
        )
    manifests = {
        corpus: build_dataset_manifest(
            corpus_configs[corpus].dataset,
            loaded[corpus],
        )
        for corpus in corpus_configs
    }

    destination = _resolve_artifact_root(config, artifact_root)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = destination / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    checkpoints = {}

    for train_corpus, client_config in corpus_configs.items():
        set_seed(config.training.seed)
        source = loaded[train_corpus]
        input_channels = int(source.train_dataset.data.shape[1])
        factory = model_factory or _default_model_factory
        model = factory(input_channels, client_config).to(
            client_config.runtime.device
        )
        _train_local_model(
            model,
            source.train_dataset,
            client_config,
            seed=config.training.seed,
            rounds=training_rounds,
        )

        checkpoint_path = checkpoint_dir / f"{_artifact_key(train_corpus)}.pt"
        save_checkpoint(
            checkpoint_path,
            mode="local",
            model_state_dict={
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
            client_id=train_corpus,
        )
        checkpoints[train_corpus] = str(checkpoint_path)

        for eval_corpus in corpus_configs:
            evaluation_dataset = _cross_corpus_evaluation_dataset(
                source,
                loaded[eval_corpus],
            )
            loader = DataLoader(
                evaluation_dataset,
                batch_size=client_config.runtime.batch_size,
                shuffle=False,
            )
            rows.append(
                {
                    "train_corpus": train_corpus,
                    "eval_corpus": eval_corpus,
                    **_evaluate_binary_model(
                        model,
                        loader,
                        torch.device(client_config.runtime.device),
                    ),
                }
            )

    metric_names = [
        key for key in rows[0] if key not in {"train_corpus", "eval_corpus"}
    ]
    summary = {
        "schema_version": LOCAL_CROSS_CORPUS_SCHEMA_VERSION,
        "method": "local",
        "seed": config.training.seed,
        "rounds": training_rounds,
        "normalization_policy": NORMALIZATION_POLICY,
        "corpora": list(corpus_configs),
        "dataset_manifests": manifests,
        "checkpoints": checkpoints,
        "rows": rows,
        "matrices": {
            metric: _metric_matrix(rows, metric) for metric in metric_names
        },
    }
    pd.DataFrame(rows).to_csv(
        destination / "local_cross_corpus.csv", index=False
    )
    save_yaml(
        destination / "local_cross_corpus_summary.yaml",
        summary,
        verbose=False,
    )
    return summary


def _validate_local_config(config: ConfigSchema) -> None:
    if config.training.mode is not TrainingMode.centralized:
        raise ValueError(
            "Local cross-corpus training requires training.mode=centralized "
            "to guarantee a channel-free complete-model topology"
        )
    corpus_names = [client.dataset.name.value for client in config.clients]
    if len(corpus_names) < 2:
        raise ValueError("Local cross-corpus training requires two corpora")
    if len(corpus_names) != len(set(corpus_names)):
        raise ValueError("Each configured client must own a unique corpus")


def _default_model_factory(input_channels: int, client_config) -> nn.Module:
    return SpeechRecognitionModel(
        input_channels=input_channels,
        server_side_model_type="cnn_birnn",
        noise_std=(client_config.noise.std if client_config.noise else 0.0),
        noise_type=(client_config.noise.type if client_config.noise else None),
    )


def _train_local_model(
    model: nn.Module,
    dataset,
    client_config,
    *,
    seed: int,
    rounds: int,
) -> None:
    if client_config.model.optimizer.lower() != "adam":
        raise ValueError(
            f"Unsupported local optimizer: {client_config.model.optimizer}"
        )
    device = torch.device(client_config.runtime.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=client_config.model.learning_rate
    )
    criterion = nn.BCEWithLogitsLoss().to(device)
    loader = DataLoader(
        dataset,
        batch_size=client_config.runtime.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    for _ in range(rounds):
        model.train()
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device).float().reshape(-1, 1)
            optimizer.zero_grad()
            logits = model(features)
            if logits.shape != labels.shape:
                raise RuntimeError(
                    "Local model logits/labels shape mismatch: "
                    f"{logits.shape} != {labels.shape}"
                )
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()


def _cross_corpus_evaluation_dataset(
    source: ConflictEmotionalDataset,
    target: ConflictEmotionalDataset,
) -> EmotionalDataset:
    """Apply source-train normalization to the target test actor view."""
    mean = source.train_dataset.mean
    std = source.train_dataset.std
    if mean is None or std is None:
        raise ValueError("Training corpus is missing normalization statistics")
    target_test = target.test_dataset
    valid_frames = target_test.valid_frames
    return EmotionalDataset(
        target_test.data.detach().cpu().numpy(),
        target_test.labels.detach().cpu().numpy(),
        mean.detach().cpu().numpy(),
        std.detach().cpu().numpy(),
        (
            None
            if valid_frames is None
            else valid_frames.detach().cpu().numpy()
        ),
    )


@torch.no_grad()
def _evaluate_binary_model(model, loader, device: torch.device) -> dict:
    model.eval()
    probabilities = []
    labels = []
    for features, target in loader:
        logits = model(features.to(device)).reshape(-1)
        probabilities.append(torch.sigmoid(logits).cpu())
        labels.append(target.long().reshape(-1).cpu())
    if not labels:
        return _empty_binary_metrics()

    y_true = torch.cat(labels).numpy()
    y_score = torch.cat(probabilities).numpy()
    y_pred = (y_score > 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    has_both_classes = len(np.unique(y_true)) == 2
    return {
        "accuracy": float((y_pred == y_true).mean()),
        "anger_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        ),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "uar": float(
            recall_score(
                y_true,
                y_pred,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        ),
        "pr_auc": (
            float(average_precision_score(y_true, y_score))
            if has_both_classes
            else None
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "num_samples": int(len(y_true)),
        "num_positive_labels": int(y_true.sum()),
        "num_positive_predictions": int(y_pred.sum()),
    }


def _empty_binary_metrics() -> dict:
    return {
        "accuracy": 0.0,
        "anger_f1": 0.0,
        "macro_f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "uar": 0.0,
        "pr_auc": None,
        "tn": 0,
        "fp": 0,
        "fn": 0,
        "tp": 0,
        "num_samples": 0,
        "num_positive_labels": 0,
        "num_positive_predictions": 0,
    }


def _metric_matrix(rows: list[dict], metric: str) -> dict:
    return {
        row["train_corpus"]: {
            candidate["eval_corpus"]: candidate[metric]
            for candidate in rows
            if candidate["train_corpus"] == row["train_corpus"]
        }
        for row in rows
    }


def _resolve_artifact_root(
    config: ConfigSchema, artifact_root: str | Path | None
) -> Path:
    root = artifact_root or config.models_save_path
    if root is None:
        raise ValueError(
            "artifact_root or config.models_save_path is required"
        )
    return Path(root) / config.experiment.name / "local_cross_corpus"


def _artifact_key(corpus: str) -> str:
    normalized = (
        character.lower() if character.isalnum() else "_"
        for character in corpus
    )
    return "".join(normalized).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--config-file", required=True)
    parser.add_argument("--artifact-root")
    parser.add_argument("--rounds", type=int)
    args = parser.parse_args()
    config = ConfigSchema(**read_yaml(args.config_file))
    run_local_cross_corpus(
        config,
        artifact_root=args.artifact_root,
        rounds=args.rounds,
    )


if __name__ == "__main__":
    main()
