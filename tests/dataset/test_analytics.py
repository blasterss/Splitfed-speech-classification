import numpy as np
import pytest

from src.dataset.analytics import DataConcatenator
from src.schema import DatasetConfig, DatasetType


def _concatenator(tmp_path, feature_names):
    config = DatasetConfig(
        name=DatasetType.savee,
        root=str(tmp_path),
        feature_names=feature_names,
    )
    return DataConcatenator([config])


def test_singleton_feature_channel_uses_stable_unindexed_keys(tmp_path):
    result = _concatenator(tmp_path, ["rms"]).aggregate_features(
        [[np.asarray([[1.0, 3.0]])]]
    )

    assert result == [{"rms_mean": 2.0, "rms_std": 1.0}]


def test_multiband_feature_channel_uses_enum_values_in_keys(tmp_path):
    result = _concatenator(tmp_path, ["mfcc"]).aggregate_features(
        [[np.asarray([[1.0, 3.0], [2.0, 4.0]])]]
    )

    assert result == [
        {
            "mfcc_0_mean": 2.0,
            "mfcc_0_std": 1.0,
            "mfcc_1_mean": 3.0,
            "mfcc_1_std": 1.0,
        }
    ]


def test_combined_datasets_use_their_own_feature_names(tmp_path, monkeypatch):
    configs = [
        DatasetConfig(
            name=DatasetType.savee,
            root=str(tmp_path),
            feature_names=["rms"],
        ),
        DatasetConfig(
            name=DatasetType.savee,
            root=str(tmp_path),
            feature_names=["zcr"],
        ),
    ]

    class Loader:
        def load(self, *, feature_mode):
            assert feature_mode == "multi_channel"
            return [[np.asarray([[1.0, 3.0]])]], [{}]

    monkeypatch.setattr(
        "src.dataset.analytics.aggregation.DatasetLoaderFactory.create",
        lambda config: Loader(),
    )

    features, _ = DataConcatenator(configs).get_agg_data()

    assert features == [
        {"rms_mean": 2.0, "rms_std": 1.0},
        {"zcr_mean": 2.0, "zcr_std": 1.0},
    ]


def test_concatenator_requires_at_least_one_config():
    with pytest.raises(ValueError, match="at least one config"):
        DataConcatenator([])


def test_dataframe_rejects_mismatched_feature_and_metadata_rows(tmp_path):
    concatenator = _concatenator(tmp_path, ["rms"])

    with pytest.raises(ValueError, match="equal row counts"):
        concatenator.to_dataframe([{"rms_mean": 1.0}], [{}, {}])
