import numpy as np

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
