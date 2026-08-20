import pytest

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
