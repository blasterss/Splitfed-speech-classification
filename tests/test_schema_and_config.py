from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.schema import DatasetConfig, DatasetType


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
