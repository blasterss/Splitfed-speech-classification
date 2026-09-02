from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.dataset.dataset import EmotionalDataset
from src.experiments.local_cross_corpus import (
    NORMALIZATION_POLICY,
    _cross_corpus_evaluation_dataset,
    run_local_cross_corpus,
)
from src.experiments.metrics import evaluate_binary_model
from src.schema import ConfigSchema


def _config(tmp_path):
    clients = []
    for client_id, (name, directory) in enumerate(
        [
            ("CREMA-D", "crema"),
            ("RAVDESS", "ravdess"),
            ("SAVEE", "savee"),
        ]
    ):
        root = tmp_path / directory
        root.mkdir()
        clients.append(
            {
                "client_id": client_id,
                "dataset": {
                    "name": name,
                    "root": str(root),
                    "feature_names": ["mfcc"],
                    "test_size": 0.5,
                },
                "model": {"optimizer": "adam", "lr": 0.01},
                "runtime": {
                    "local_steps": 1,
                    "batch_size": 2,
                    "seed": 7,
                    "device": "cpu",
                },
            }
        )
    return ConfigSchema(
        data_path=str(tmp_path),
        models_save_path=str(tmp_path / "artifacts"),
        experiment={
            "name": "local-matrix",
            "transport": "queue",
            "seed": 7,
        },
        training={
            "mode": "centralized",
            "num_rounds": 1,
            "seed": 7,
            "eval_every": 1,
            "fed_every": 1,
        },
        clients=clients,
        channels={},
    )


def _corpus(offset: float):
    train = EmotionalDataset(
        np.asarray(
            [
                [[offset, offset + 1]],
                [[offset + 2, offset + 3]],
            ],
            dtype=np.float32,
        ),
        np.asarray([0.0, 1.0]),
        mean=np.asarray([offset + 1.5], dtype=np.float32),
        std=np.asarray([1.0], dtype=np.float32),
        valid_frames=np.asarray([2, 2]),
    )
    test = EmotionalDataset(
        np.asarray(
            [
                [[offset + 1, offset + 1]],
                [[offset + 3, offset + 3]],
            ],
            dtype=np.float32,
        ),
        np.asarray([0.0, 1.0]),
        mean=np.asarray([offset + 100], dtype=np.float32),
        std=np.asarray([9.0], dtype=np.float32),
        valid_frames=np.asarray([2, 2]),
    )
    return SimpleNamespace(
        train_dataset=train,
        test_dataset=test,
        extraction_report={
            "discovered": 4,
            "loaded": 4,
            "failed": 0,
            "failure_reasons": {},
        },
        coverage={
            "train": {"samples": 2},
            "test": {"samples": 2},
        },
        train_actor_ids=("train-actor",),
        test_actor_ids=("test-actor",),
    )


class TinyBinaryModel(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output = nn.Linear(input_channels, 1)

    def forward(self, features):
        return self.output(self.pool(features).squeeze(-1))


def test_cross_corpus_view_uses_source_train_normalization():
    source = _corpus(10.0)
    target = _corpus(20.0)

    evaluation = _cross_corpus_evaluation_dataset(source, target)
    features, _ = evaluation[0]

    expected = target.test_dataset.data[0] - source.train_dataset.mean[:, None]
    expected /= source.train_dataset.std[:, None]
    torch.testing.assert_close(features, expected)
    torch.testing.assert_close(evaluation.mean, source.train_dataset.mean)


def test_binary_metrics_marks_one_class_pr_auc_unavailable():
    model = TinyBinaryModel(1)
    model.output.weight.data.zero_()
    model.output.bias.data.fill_(-1.0)
    dataset = EmotionalDataset(
        np.ones((2, 1, 2), dtype=np.float32),
        np.zeros(2, dtype=np.float32),
    )

    metrics = evaluate_binary_model(
        model,
        DataLoader(dataset, batch_size=2),
        torch.device("cpu"),
    )

    assert metrics["pr_auc"] is None
    assert metrics["tn"] == 2
    assert metrics["num_positive_labels"] == 0


def test_local_training_writes_complete_three_by_three_matrix(tmp_path):
    config = _config(tmp_path)
    datasets = {
        "CREMA-D": _corpus(0.0),
        "RAVDESS": _corpus(10.0),
        "SAVEE": _corpus(20.0),
    }

    summary = run_local_cross_corpus(
        config,
        datasets=datasets,
        model_factory=lambda input_channels, client: TinyBinaryModel(
            input_channels
        ),
    )

    assert summary["normalization_policy"] == NORMALIZATION_POLICY
    assert set(summary["dataset_manifests"]) == set(datasets)
    assert len(summary["rows"]) == 9
    assert set(summary["matrices"]["anger_f1"]) == set(datasets)
    assert all(
        set(row) == set(datasets)
        for row in summary["matrices"]["anger_f1"].values()
    )
    artifact_root = (
        tmp_path / "artifacts" / "local-matrix" / "local_cross_corpus"
    )
    assert (artifact_root / "local_cross_corpus.csv").is_file()
    assert (artifact_root / "local_cross_corpus_summary.yaml").is_file()
    assert len(list((artifact_root / "checkpoints").glob("*.pt"))) == 3
