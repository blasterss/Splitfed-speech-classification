from src.main import _save_resolved_config, _save_run_metadata
from src.schema import ConfigSchema
from src.utils.common import read_yaml


def test_resolved_config_round_trips_mode_seed_and_topology(tmp_path):
    config = ConfigSchema(
        data_path=str(tmp_path),
        models_save_path=str(tmp_path / "checkpoints"),
        experiment={"name": "artifact", "transport": "queue", "seed": 17},
        training={
            "mode": "centralized",
            "num_rounds": 1,
            "seed": 19,
            "eval_every": 1,
            "fed_every": 1,
        },
        clients=[
            {
                "client_id": 0,
                "dataset": {"name": "SAVEE", "root": str(tmp_path)},
                "model": {"lr": 0.001, "optimizer": "adam"},
                "runtime": {
                    "local_steps": 1,
                    "batch_size": 1,
                    "seed": 23,
                    "device": "cpu",
                },
            }
        ],
        split_server=None,
        fed_server=None,
        channels={},
    )
    artifact_path = tmp_path / "run"
    artifact_path.mkdir()

    _save_resolved_config(config, artifact_path)

    raw = read_yaml(artifact_path / "resolved_config.yaml")
    restored = ConfigSchema(**raw)
    assert restored.training.mode.value == "centralized"
    assert restored.experiment.seed == 17
    assert restored.training.seed == 19
    assert restored.clients[0].runtime.seed == 23
    assert restored.channels == {}


def test_run_metadata_records_environment_and_seed_tree(tmp_path):
    config = ConfigSchema(
        data_path=str(tmp_path),
        models_save_path=str(tmp_path / "checkpoints"),
        experiment={"name": "metadata", "transport": "queue", "seed": 11},
        training={
            "mode": "centralized",
            "num_rounds": 1,
            "seed": 13,
            "eval_every": 1,
            "fed_every": 1,
        },
        clients=[
            {
                "client_id": 7,
                "dataset": {
                    "name": "SAVEE",
                    "root": str(tmp_path),
                    "split_seed": 17,
                },
                "model": {"lr": 0.001, "optimizer": "adam"},
                "runtime": {
                    "local_steps": 1,
                    "batch_size": 1,
                    "seed": 19,
                    "device": "cpu",
                },
            }
        ],
        split_server=None,
        fed_server=None,
        channels={},
    )
    artifact_path = tmp_path / "run"
    artifact_path.mkdir()

    _save_run_metadata(
        config,
        artifact_path,
        cli_overrides=["training.num_rounds=1"],
    )

    metadata = read_yaml(artifact_path / "run_metadata.yaml")
    assert metadata["seed_tree"] == {
        "experiment": 11,
        "training": 13,
        "clients": {"7": {"runtime": 19, "dataset_split": 17}},
        "split_server": None,
        "fed_server": None,
    }
    assert metadata["environment"]["python"]
    assert metadata["environment"]["pytorch"]
    assert isinstance(metadata["environment"]["cuda_available"], bool)
    assert metadata["configuration"]["cli_overrides"] == [
        "training.num_rounds=1"
    ]
    assert metadata["created_at_utc"].endswith("+00:00")
