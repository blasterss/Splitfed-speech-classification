"""Local-only training with a train-corpus by evaluation-corpus matrix."""

import argparse
import multiprocessing as mp
import queue
import time
import traceback
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..application.artifacts import _save_resource_metrics
from ..dataset.dataset import (
    ConflictEmotionalDataset,
    EmotionalDataset,
    build_dataset_manifest,
)
from ..model.speech_model import SpeechRecognitionModel
from ..schema import ConfigSchema, TrainingMode
from ..utils.config import read_yaml, save_yaml
from ..utils.persistence import load_checkpoint, save_checkpoint
from ..utils.runtime.resource_metrics import (
    ResourceTracker,
    owned_resource_bytes,
)
from ..utils.training import build_optimizer, set_seed
from .metrics import evaluate_binary_model

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
    destination = _resolve_artifact_root(config, artifact_root)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = destination / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    resource_metrics = []
    checkpoints = {}

    if datasets is None and model_factory is None:
        manifests, rows, resource_metrics, checkpoints = (
            _run_isolated_local_models(
                config, corpus_configs, destination, training_rounds
            )
        )
    else:
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
        for train_corpus, client_config in corpus_configs.items():
            model, training_metrics, checkpoint_path = _train_participant(
                config,
                client_config,
                loaded[train_corpus],
                destination,
                training_rounds,
                train_corpus,
                model_factory=model_factory,
            )
            resource_metrics.extend(training_metrics)
            checkpoints[train_corpus] = str(checkpoint_path)
            evaluation_rows, evaluation_metrics = _evaluate_local_model(
                model,
                loaded[train_corpus],
                loaded,
                client_config,
                training_rounds,
                train_corpus,
            )
            rows.extend(evaluation_rows)
            resource_metrics.extend(evaluation_metrics)

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
    _save_resource_metrics(resource_metrics, destination)
    return summary


def _train_participant(
    config,
    client_config,
    source,
    destination,
    training_rounds,
    train_corpus,
    *,
    model_factory=None,
):
    set_seed(config.training.seed)
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
        resource_metrics=(resource_metrics := []),
        client_id=train_corpus,
    )

    checkpoint_path = (
        destination / "checkpoints" / f"{_artifact_key(train_corpus)}.pt"
    )
    save_checkpoint(
        checkpoint_path,
        mode="local",
        model_state_dict={
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        client_id=train_corpus,
    )
    return model, resource_metrics, checkpoint_path


def _evaluate_local_model(
    model,
    source,
    targets,
    client_config,
    training_rounds,
    train_corpus,
):
    rows = []
    resource_metrics = []
    for eval_corpus, target in targets.items():
        tracker = ResourceTracker(
            "local_evaluation",
            client_config.runtime.device,
            client_id=train_corpus,
        )
        evaluation_dataset = _cross_corpus_evaluation_dataset(source, target)
        loader = DataLoader(
            evaluation_dataset,
            batch_size=client_config.runtime.batch_size,
            shuffle=False,
        )
        evaluation_metrics = evaluate_binary_model(
            model,
            loader,
            torch.device(client_config.runtime.device),
        )
        resource_metrics.append(
            tracker.snapshot(
                round_idx=training_rounds,
                phase=f"evaluation:{eval_corpus}",
                samples=len(evaluation_dataset),
                batches=len(loader),
            )
        )
        rows.append(
            {
                "train_corpus": train_corpus,
                "eval_corpus": eval_corpus,
                **evaluation_metrics,
            }
        )

    return rows, resource_metrics


def _local_training_worker(
    config, train_corpus, destination, rounds, report_queue
):
    try:
        client_config = next(
            client
            for client in config.clients
            if client.dataset.name.value == train_corpus
        )
        source = ConflictEmotionalDataset(client_config.dataset)
        _, metrics, checkpoint_path = _train_participant(
            config,
            client_config,
            source,
            destination,
            rounds,
            train_corpus,
        )
        report_queue.put(
            {
                "ok": True,
                "metrics": metrics,
                "checkpoint": str(checkpoint_path),
                "manifest": build_dataset_manifest(
                    client_config.dataset, source
                ),
            }
        )
    except BaseException as exc:
        report_queue.put(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        raise


def _local_evaluation_worker(
    config, train_corpus, checkpoint_path, rounds, report_queue
):
    try:
        corpus_configs = {
            client.dataset.name.value: client for client in config.clients
        }
        client_config = corpus_configs[train_corpus]
        loaded = {
            corpus: ConflictEmotionalDataset(client.dataset)
            for corpus, client in corpus_configs.items()
        }
        source = loaded[train_corpus]
        model = _default_model_factory(
            int(source.train_dataset.data.shape[1]), client_config
        ).to(client_config.runtime.device)
        state = load_checkpoint(
            checkpoint_path,
            expected_mode="local",
            expected_state_dict=model.state_dict(),
            expected_client_id=train_corpus,
        )
        model.load_state_dict(state)
        rows, metrics = _evaluate_local_model(
            model,
            source,
            loaded,
            client_config,
            rounds,
            train_corpus,
        )
        report_queue.put({"ok": True, "rows": rows, "metrics": metrics})
    except BaseException as exc:
        report_queue.put(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        raise


def _run_spawn_worker(context, target, args, *, name, timeout):
    report_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=target,
        args=(*args, report_queue),
        name=name,
    )
    process.start()
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            process.terminate()
            process.join(timeout=5)
            raise TimeoutError(f"{name} exceeded the configured timeout")
        try:
            report = report_queue.get(timeout=min(remaining, 0.1))
            break
        except queue.Empty:
            if not process.is_alive():
                process.join(timeout=1)
                try:
                    report = report_queue.get(timeout=1)
                    break
                except queue.Empty as exc:
                    raise RuntimeError(
                        f"{name} exited with code {process.exitcode} "
                        "without publishing a report"
                    ) from exc
    process.join(timeout=5)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        raise RuntimeError(f"{name} did not exit after publishing its report")
    report_queue.close()
    report_queue.join_thread()
    if process.exitcode != 0 or not report.get("ok"):
        raise RuntimeError(
            f"{name} failed: {report.get('error', 'unknown error')}\n"
            f"{report.get('traceback', '')}"
        )
    return report


def _run_isolated_local_models(
    config, corpus_configs, destination, training_rounds
):
    context = mp.get_context("spawn")
    manifests = {}
    rows = []
    resource_metrics = []
    checkpoints = {}
    timeout = config.training.barrier_timeout_sec
    for train_corpus in corpus_configs:
        training = _run_spawn_worker(
            context,
            _local_training_worker,
            (config, train_corpus, destination, training_rounds),
            name=f"LocalTrain-{train_corpus}",
            timeout=timeout,
        )
        manifests[train_corpus] = training["manifest"]
        checkpoints[train_corpus] = training["checkpoint"]
        resource_metrics.extend(training["metrics"])
        evaluation = _run_spawn_worker(
            context,
            _local_evaluation_worker,
            (
                config,
                train_corpus,
                training["checkpoint"],
                training_rounds,
            ),
            name=f"LocalEval-{train_corpus}",
            timeout=timeout,
        )
        rows.extend(evaluation["rows"])
        resource_metrics.extend(evaluation["metrics"])
    return manifests, rows, resource_metrics, checkpoints


def _validate_local_config(config: ConfigSchema) -> None:
    if config.training.mode is not TrainingMode.local:
        raise ValueError(
            "Local cross-corpus training requires training.mode=local"
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
    resource_metrics: list[dict] | None = None,
    client_id: str | int | None = None,
) -> None:
    device = torch.device(client_config.runtime.device)
    optimizer = build_optimizer(model.parameters(), client_config.model)
    criterion = nn.BCEWithLogitsLoss().to(device)
    loader = DataLoader(
        dataset,
        batch_size=client_config.runtime.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    for round_idx in range(1, rounds + 1):
        tracker = ResourceTracker("local", device, client_id=client_id)
        model.train()
        samples = 0
        batches = 0
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
            samples += labels.numel()
            batches += 1
        if resource_metrics is not None:
            metric = tracker.snapshot(
                round_idx=round_idx,
                phase="train",
                samples=samples,
                batches=batches,
            )
            metric.update(owned_resource_bytes(model, optimizer, dataset))
            resource_metrics.append(metric)


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
