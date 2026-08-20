import numpy as np
import pytest

from src.dataset.dataset import ConflictEmotionalDataset
from src.dataset.processors.base_processor import BaseDatasetLoader
from src.dataset.processors.crema_d_processor import CremaDLoader
from src.dataset.processors.factory import DatasetLoaderFactory
from src.dataset.processors.ravdess_processor import RavdessLoader
from src.dataset.processors.savee_processor import SaveeLoader
from src.schema import DatasetConfig, DatasetType


@pytest.mark.parametrize(
    ("dataset_type", "filename", "label", "actor_id", "sex"),
    [
        (
            DatasetType.crema_d,
            "1001_IEO_ANG_HI.wav",
            1,
            "1001",
            "M",
        ),
        (
            DatasetType.crema_d,
            "1002_IEO_NEU_XX.wav",
            0,
            "1002",
            "F",
        ),
        (
            DatasetType.ravdess,
            "03-01-05-01-02-01-12.wav",
            1,
            "12",
            "F",
        ),
        (
            DatasetType.ravdess,
            "03-01-04-01-01-01-13.wav",
            0,
            "13",
            "M",
        ),
        (DatasetType.savee, "DC_a01.wav", 1, "DC", "M"),
        (DatasetType.savee, "KL_n10.wav", 0, "KL", "M"),
    ],
)
def test_dataset_filename_parsers(
    tmp_path, dataset_type, filename, label, actor_id, sex
):
    config = DatasetConfig(name=dataset_type, root=str(tmp_path))
    loader = DatasetLoaderFactory.create(config)

    assert loader.parse_label(filename) == label
    assert loader.parse_actor_id(filename) == actor_id
    assert loader.parse_sex(filename) == sex


@pytest.mark.parametrize(
    ("dataset_type", "loader_type"),
    [
        (DatasetType.crema_d, CremaDLoader),
        (DatasetType.ravdess, RavdessLoader),
        (DatasetType.savee, SaveeLoader),
    ],
)
def test_factory_registers_all_supported_datasets(
    tmp_path, dataset_type, loader_type
):
    config = DatasetConfig(name=dataset_type, root=str(tmp_path))

    assert isinstance(DatasetLoaderFactory.create(config), loader_type)


class _ActorDatasetLoader:
    def load(self):
        data = [np.full((3, 8), value) for value in range(4)]
        metadata = [
            {"label": value % 2, "actor_id": f"actor-{value}"}
            for value in range(4)
        ]
        return data, metadata


def test_actor_split_uses_configured_seed_and_remains_disjoint(
    tmp_path, monkeypatch
):
    observed = {}

    def split(actors, *, test_size, random_state):
        observed.update(test_size=test_size, random_state=random_state)
        return actors[:2], actors[2:]

    monkeypatch.setattr(
        "src.dataset.dataset.DatasetLoaderFactory.create",
        lambda config: _ActorDatasetLoader(),
    )
    monkeypatch.setattr("src.dataset.dataset.train_test_split", split)
    config = DatasetConfig(
        name=DatasetType.savee,
        root=str(tmp_path),
        test_size=0.5,
        split_seed=123,
    )

    dataset = ConflictEmotionalDataset(config)

    assert observed == {"test_size": 0.5, "random_state": 123}
    assert set(dataset.train_actor_ids).isdisjoint(dataset.test_actor_ids)
    assert set(dataset.train_actor_ids) | set(dataset.test_actor_ids) == {
        "actor-0",
        "actor-1",
        "actor-2",
        "actor-3",
    }
    assert dataset.coverage == {
        "train": {"samples": 2, "actors": 2, "class_0": 1, "class_1": 1},
        "test": {"samples": 2, "actors": 2, "class_0": 1, "class_1": 1},
    }


def test_training_split_rejects_missing_binary_class(tmp_path, monkeypatch):
    class OneClassLoader:
        def load(self):
            return (
                [np.zeros((3, 8)) for _ in range(4)],
                [
                    {"label": 0, "actor_id": f"actor-{index}"}
                    for index in range(4)
                ],
            )

    monkeypatch.setattr(
        "src.dataset.dataset.DatasetLoaderFactory.create",
        lambda config: OneClassLoader(),
    )

    with pytest.raises(ValueError, match="both binary classes"):
        ConflictEmotionalDataset(
            DatasetConfig(
                name=DatasetType.savee,
                root=str(tmp_path),
                test_size=0.5,
            )
        )


class _ReportingLoader(BaseDatasetLoader):
    def parse_label(self, filename):
        return 1 if "good" in filename else 0

    def parse_actor_id(self, filename):
        return filename.split("_")[0]

    def parse_sex(self, filename):
        return "M"


def test_loader_accounts_for_extraction_failure_reasons(tmp_path, monkeypatch):
    (tmp_path / "actor-0_bad.wav").touch()
    (tmp_path / "actor-1_good.wav").touch()

    def load_audio(path, **kwargs):
        if "bad" in path.name:
            raise ValueError("corrupt audio")
        return np.ones(16), 16000

    monkeypatch.setattr(
        "src.dataset.processors.base_processor.FeatureUtils.load_audio",
        load_audio,
    )
    monkeypatch.setattr(
        "src.dataset.processors.base_processor.FeatureExtraction.get_all_features",
        lambda *args, **kwargs: np.ones((3, 4)),
    )
    loader = _ReportingLoader(
        DatasetConfig(name=DatasetType.savee, root=str(tmp_path))
    )

    data, metadata = loader.load()

    assert len(data) == len(metadata) == 1
    assert loader.last_load_report == {
        "discovered": 2,
        "loaded": 1,
        "failed": 1,
        "failure_reasons": {"ValueError": 1},
    }
