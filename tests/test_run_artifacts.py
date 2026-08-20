from src.main import (
    _config_sha256,
    _file_sha256,
    _save_first_failure,
    _save_resolved_config,
    _save_run_metadata,
)
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
        experiment={
            "name": "metadata",
            "profile": "smoke",
            "transport": "queue",
            "seed": 11,
        },
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
        resolved_config_sha256=_config_sha256(config),
        configuration_provenance={
            "profile": {
                "name": "smoke",
                "version": "1",
                "source": "yaml",
            },
            "cli_overrides": ["training.num_rounds=1"],
            "overrides": [
                {
                    "source": "cli",
                    "path": "training.num_rounds",
                    "expression": "training.num_rounds=1",
                }
            ],
        },
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
    assert metadata["configuration"] == {
        "profile": {"name": "smoke", "version": "1", "source": "yaml"},
        "cli_overrides": ["training.num_rounds=1"],
        "overrides": [
            {
                "source": "cli",
                "path": "training.num_rounds",
                "expression": "training.num_rounds=1",
            }
        ],
        "resolved_config_sha256": _config_sha256(config),
    }
    assert "git_revision" in metadata["environment"]
    assert isinstance(metadata["environment"]["git_dirty"], bool)
    assert "dependency_lock_sha256" in metadata["environment"]
    assert metadata["runtime"] == {
        "backend": "local_multiprocessing",
        "start_method": "spawn",
        "container_image_digest": None,
    }
    assert metadata["policies"] == {
        "transport": {"name": "queue", "version": "1"},
        "aggregation": None,
        "scheduler": None,
    }
    assert metadata["protocols"] == {
        "message": {"name": "secureasr.queue", "version": 3}
    }
    assert metadata["created_at_utc"].endswith("+00:00")


def test_config_hash_is_canonical_and_sensitive_to_values(tmp_path):
    first = {"training": {"num_rounds": 1, "seed": 42}}
    reordered = {"training": {"seed": 42, "num_rounds": 1}}
    changed = {"training": {"seed": 42, "num_rounds": 2}}

    assert _config_sha256(first) == _config_sha256(reordered)
    assert _config_sha256(first) != _config_sha256(changed)

    lock_file = tmp_path / "uv.lock"
    lock_file.write_bytes(b"locked\n")
    assert _file_sha256(lock_file) == (
        "3a52732e0c98263090a2cd2509e7d2244d7194bd65f78b29e6ef6448e8143666"
    )


def test_first_failure_artifact_is_versioned_and_structured(tmp_path):
    failure = {
        "schema_version": 1,
        "timestamp_utc": "2026-08-21T00:00:00+00:00",
        "component": "client",
        "client_id": 2,
        "round": 3,
        "step": None,
        "exception_type": "RuntimeError",
        "message": "failed",
        "traceback": "RuntimeError: failed\n",
    }

    _save_first_failure(failure, tmp_path)

    assert read_yaml(tmp_path / "first_failure.yaml") == failure
