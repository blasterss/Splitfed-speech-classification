from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.schema import (
    ClientModelConfig,
    ClientRuntimeConfig,
    ConfigSchema,
    DatasetConfig,
    DatasetType,
    ExperimentConfig,
    FedServerConfig,
    NoiseConfig,
    QueueChannelConfig,
    SplitServerConfig,
    SplitServerModelConfig,
    TrainingConfig,
    WorkloadPolicy,
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


@pytest.mark.parametrize("name", ["", ".", "..", "nested/run", "nested\\run"])
def test_experiment_name_must_be_safe_artifact_component(name):
    with pytest.raises(ValidationError, match="name"):
        ExperimentConfig(name=name, transport="queue", seed=42)


@pytest.mark.parametrize("dataset_type", list(DatasetType))
def test_dataset_config_resolves_existing_root(tmp_path, dataset_type):
    config = DatasetConfig(name=dataset_type, root=str(tmp_path))

    assert config.root == str(tmp_path.resolve())


def test_dataset_config_rejects_missing_root(tmp_path):
    missing_root = tmp_path / "missing"

    with pytest.raises(ValidationError, match="Path does not exist"):
        DatasetConfig(name=DatasetType.savee, root=str(missing_root))


def test_dataset_config_rejects_empty_feature_list(tmp_path):
    with pytest.raises(ValidationError, match="feature_names"):
        DatasetConfig(
            name=DatasetType.savee,
            root=str(tmp_path),
            feature_names=[],
        )


def test_dataset_config_rejects_null_feature_list(tmp_path):
    with pytest.raises(ValidationError, match="feature_names"):
        DatasetConfig(
            name=DatasetType.savee,
            root=str(tmp_path),
            feature_names=None,
        )


def test_dataset_config_rejects_duplicate_features(tmp_path):
    with pytest.raises(ValidationError, match="must not contain duplicates"):
        DatasetConfig(
            name=DatasetType.savee,
            root=str(tmp_path),
            feature_names=["mfcc", "mfcc"],
        )


def test_example_config_uses_portable_dataset_paths():
    config_path = Path("configs/config.example.yaml")
    config = yaml.safe_load(config_path.read_text())

    assert config["models_save_path"] == "artifacts"
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


def test_client_runtime_rejects_unknown_workload_policy():
    with pytest.raises(ValidationError, match="workload_policy"):
        ClientRuntimeConfig(
            local_steps=1,
            workload_policy="until_tired",
            batch_size=1,
            seed=42,
        )


def test_client_runtime_accepts_fixed_steps_workload_policy():
    runtime = ClientRuntimeConfig(
        local_steps=5,
        workload_policy="fixed_steps_v1",
        batch_size=1,
        seed=42,
    )

    assert runtime.workload_policy is WorkloadPolicy.fixed_steps_v1


def test_queue_channel_requires_positive_timeout():
    with pytest.raises(ValidationError, match="timeout"):
        QueueChannelConfig(
            transport="queue",
            name="invalid-timeout",
            timeout=0,
        )


def test_split_server_requires_one_gradient_accumulation_step():
    with pytest.raises(ValidationError, match="gradient_accumulation_steps"):
        SplitServerModelConfig(
            pos_weight=1,
            optimizer="adam",
            lr=0.001,
            device="cpu",
            gradient_accumulation_steps=2,
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


def test_sequential_strategy_requires_shared_server_model():
    with pytest.raises(ValidationError, match="applies only to shared"):
        SplitServerConfig(
            model={"lr": 0.001, "device": "cpu"},
            model_scope="personalized",
            training_strategy="sequential_v1",
            seed=42,
            split_uplink_channel="split_uplink",
            split_downlink_channel="split_downlink",
        )


def _mergesfl_config():
    raw = yaml.safe_load(Path("configs/config.example.yaml").read_text())
    raw["split_server"]["training_strategy"] = "mergesfl_v1"
    for client in raw["clients"]:
        client["runtime"]["workload_policy"] = "fixed_steps_v1"
        client["runtime"]["drop_last"] = True
    return raw


def test_mergesfl_accepts_fixed_equal_steps_and_unequal_batch_sizes():
    raw = _mergesfl_config()
    raw["clients"][0]["runtime"]["batch_size"] = 4
    raw["clients"][1]["runtime"]["batch_size"] = 8
    raw["clients"][2]["runtime"]["batch_size"] = 16

    config = ConfigSchema(**raw)

    assert config.split_server.training_strategy.value == "mergesfl_v1"


def test_mergesfl_requires_fixed_steps_policy():
    raw = _mergesfl_config()
    raw["clients"][1]["runtime"]["workload_policy"] = "max_steps_v1"

    with pytest.raises(ValidationError, match="requires fixed_steps_v1"):
        ConfigSchema(**raw)


def test_mergesfl_requires_equal_local_steps():
    raw = _mergesfl_config()
    raw["clients"][1]["runtime"]["local_steps"] += 1

    with pytest.raises(ValidationError, match="requires equal local_steps"):
        ConfigSchema(**raw)


def _mergesfl_algorithm1_config():
    raw = yaml.safe_load(Path("configs/config.example.yaml").read_text())
    raw["split_server"]["training_strategy"] = "mergesfl_algorithm1_v1"
    raw["fed_server"]["strategy"] = "mergesfl_batch_weighted_v1"
    raw["load_controller"] = {
        "max_batch_size": 16,
        "local_steps": 42,
        "ingress_budget_bytes": 4096,
        "feature_bytes_per_sample": 128,
        "min_clients": 2,
        "max_clients": 3,
        "initial_worker_states": {
            client["client_id"]: {
                "compute_seconds_per_sample": 0.01,
                "transfer_seconds_per_sample": 0.001,
            }
            for client in raw["clients"]
        },
    }
    return raw


def test_algorithm1_requires_consistent_control_plane():
    config = ConfigSchema(**_mergesfl_algorithm1_config())

    assert config.load_controller.name == "mergesfl_algorithm1_v1"


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda raw: raw["fed_server"].update(strategy="fedavg"),
            "batch-weighted",
        ),
        (
            lambda raw: raw["load_controller"][
                "initial_worker_states"
            ].pop(2),
            "initial_worker_states",
        ),
    ],
)
def test_algorithm1_rejects_inconsistent_control_plane(mutation, match):
    raw = _mergesfl_algorithm1_config()
    mutation(raw)

    with pytest.raises(ValidationError, match=match):
        ConfigSchema(**raw)


def test_mergesfl_requires_dropping_incomplete_batches():
    raw = _mergesfl_config()
    raw["clients"][0]["runtime"]["drop_last"] = False

    with pytest.raises(ValidationError, match="requires drop_last=true"):
        ConfigSchema(**raw)


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
        ClientModelConfig(lr=0.001, optimizer="rmsprop")


def test_schema_accepts_sgd_parameters():
    config = ClientModelConfig(
        lr=0.1,
        optimizer="sgd",
        momentum=0.9,
        nesterov=True,
        weight_decay=0.0005,
    )

    assert config.optimizer.value == "sgd"
    assert config.momentum == 0.9


def test_schema_rejects_nesterov_without_sgd_momentum():
    with pytest.raises(ValidationError, match="nesterov requires SGD"):
        ClientModelConfig(lr=0.1, optimizer="sgd", nesterov=True)


def test_schema_rejects_unsupported_noise_type():
    with pytest.raises(ValidationError, match="type"):
        NoiseConfig(type="uniform", std=0.1)


def _federated_cross_eval_config(tmp_path):
    return {
        "data_path": str(tmp_path),
        "models_save_path": str(tmp_path / "artifacts"),
        "experiment": {
            "name": "cross-eval",
            "transport": "queue",
            "seed": 42,
            "cross_corpus_evaluation": True,
        },
        "training": {
            "mode": "federated",
            "num_rounds": 2,
            "seed": 42,
            "eval_every": 2,
            "fed_every": 1,
        },
        "clients": [
            {
                "client_id": 0,
                "dataset": {"name": "SAVEE", "root": str(tmp_path)},
                "model": {"lr": 0.001},
                "runtime": {
                    "local_steps": 1,
                    "batch_size": 1,
                    "seed": 42,
                },
            }
        ],
        "fed_server": {
            "strategy": "weighted_fedavg",
            "seed": 42,
            "device": "cpu",
            "aggregation_freq": 1,
            "min_clients": 1,
            "quorum_timeout_sec": 1,
            "federated_uplink_channel": "federated_uplink",
            "federated_downlink_channel": "federated_downlink",
        },
        "channels": {
            "federated_uplink": {
                "transport": "queue",
                "name": "federated_uplink",
                "timeout": 1,
            },
            "federated_downlink": {
                "transport": "queue",
                "name": "federated_downlink",
                "timeout": 1,
            },
        },
    }


@pytest.mark.parametrize(
    "training_override",
    [
        {"aggregate_final": False},
        {"num_rounds": 3, "fed_every": 2},
    ],
)
def test_federated_cross_eval_requires_final_global_model(
    tmp_path, training_override
):
    raw = _federated_cross_eval_config(tmp_path)
    raw["training"].update(training_override)
    raw["fed_server"]["aggregation_freq"] = raw["training"]["fed_every"]

    with pytest.raises(ValidationError, match="final-round aggregation"):
        ConfigSchema(**raw)
