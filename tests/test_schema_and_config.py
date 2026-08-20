from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.schema import (
    ClientModelConfig,
    DatasetConfig,
    DatasetType,
    ExperimentConfig,
    FedServerConfig,
    NoiseConfig,
    SplitServerModelConfig,
    TrainingConfig,
    _validate_device_available,
)


def test_experiment_config_rejects_unknown_profile():
    with pytest.raises(ValidationError, match="profile"):
        ExperimentConfig(
            name="invalid-profile",
            profile="missing",
            transport="queue",
            seed=42,
        )


@pytest.mark.parametrize("dataset_type", list(DatasetType))
def test_dataset_config_resolves_existing_root(tmp_path, dataset_type):
    config = DatasetConfig(name=dataset_type, root=str(tmp_path))

    assert config.root == str(tmp_path.resolve())


def test_dataset_config_rejects_missing_root(tmp_path):
    missing_root = tmp_path / "missing"

    with pytest.raises(ValidationError, match="Path does not exist"):
        DatasetConfig(name=DatasetType.savee, root=str(missing_root))


def test_example_config_uses_portable_dataset_paths():
    config_path = Path("configs/config.example.yaml")
    config = yaml.safe_load(config_path.read_text())

    assert config["models_save_path"] == "checkpoints"
    assert "model_save_path" not in config
    assert config["data_path"] == "../datasets"
    assert [client["dataset"]["root"] for client in config["clients"]] == [
        "../datasets/CREMA-D/raw",
        "../datasets/RAVDESS/raw",
        "../datasets/SAVEE/raw",
    ]


def test_training_config_requires_positive_barrier_timeout():
    with pytest.raises(ValidationError, match="barrier_timeout_sec"):
        TrainingConfig(
            num_rounds=1,
            seed=42,
            eval_every=1,
            fed_every=1,
            barrier_timeout_sec=0,
        )


def test_split_server_requires_positive_gradient_accumulation_steps():
    with pytest.raises(ValidationError, match="gradient_accumulation_steps"):
        SplitServerModelConfig(
            pos_weight=1,
            optimizer="adam",
            lr=0.001,
            device="cpu",
            gradient_accumulation_steps=0,
        )


def test_split_server_requires_positive_batch_timeout():
    with pytest.raises(ValidationError, match="batch_timeout_sec"):
        SplitServerModelConfig(
            pos_weight=1,
            optimizer="adam",
            lr=0.001,
            device="cpu",
            batch_timeout_sec=0,
        )


def test_fed_server_requires_positive_quorum_timeout():
    with pytest.raises(ValidationError, match="quorum_timeout_sec"):
        FedServerConfig(
            strategy="fedavg",
            seed=42,
            device="cpu",
            aggregation_freq=1,
            min_clients=1,
            quorum_timeout_sec=0,
            federated_uplink_channel="federated_uplink",
            federated_downlink_channel="federated_downlink",
        )


def test_device_validation_rejects_unknown_device_name():
    with pytest.raises(ValueError, match="must be"):
        _validate_device_available("gpu", "client.device")


def test_device_validation_rejects_cuda_when_unavailable(monkeypatch):
    monkeypatch.setattr("src.schema.torch.cuda.is_available", lambda: False)

    with pytest.raises(ValueError, match="unavailable"):
        _validate_device_available("cuda", "client.device")


def test_device_validation_rejects_missing_cuda_index(monkeypatch):
    monkeypatch.setattr("src.schema.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("src.schema.torch.cuda.device_count", lambda: 1)

    with pytest.raises(ValueError, match="only 1"):
        _validate_device_available("cuda:1", "client.device")


def test_schema_rejects_unsupported_optimizer():
    with pytest.raises(ValidationError, match="optimizer"):
        ClientModelConfig(lr=0.001, optimizer="sgd")


def test_schema_rejects_unsupported_noise_type():
    with pytest.raises(ValidationError, match="type"):
        NoiseConfig(type="uniform", std=0.1)
