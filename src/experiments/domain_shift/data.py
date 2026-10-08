"""E0 feature layout, masked summaries and analysis config loading."""

import numpy as np

from ...schema import DatasetConfig, FeatureType

FEATURE_WIDTHS = {
    FeatureType.mfcc: 13,
    FeatureType.rms: 1,
    FeatureType.zcr: 1,
}


def _masked_record_means(
    data: np.ndarray, valid_frames: np.ndarray
) -> np.ndarray:
    if data.ndim != 3 or len(data) != len(valid_frames):
        raise ValueError("Expected aligned [N, F, T] data and valid frames")
    if np.any(valid_frames <= 0) or np.any(valid_frames > data.shape[-1]):
        raise ValueError("valid_frames must fit the temporal dimension")
    mask = np.arange(data.shape[-1])[None, :] < valid_frames[:, None]
    return (data * mask[:, None, :]).sum(axis=2) / valid_frames[:, None]


def _feature_channel_names(
    feature_types: list[FeatureType], feature_count: int
) -> list[str]:
    names = []
    for feature_type in feature_types:
        if feature_type not in FEATURE_WIDTHS:
            raise ValueError("E0 supports only MFCC, RMS and ZCR features")
        width = FEATURE_WIDTHS[feature_type]
        if width == 1:
            names.append(feature_type.value)
        else:
            names.extend(
                f"{feature_type.value}_{index}" for index in range(width)
            )
    if len(names) != feature_count:
        raise ValueError(
            "Configured feature layout does not match extracted channel count"
        )
    return names


def _validate_matrix(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not values.shape[1]:
        raise ValueError(f"{name} must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values")
    return values


def _dataset_configs_from_mapping(raw_config: dict) -> list[DatasetConfig]:
    if "datasets" in raw_config:
        items = raw_config["datasets"]
    elif "clients" in raw_config:
        items = [client["dataset"] for client in raw_config["clients"]]
    else:
        raise ValueError("Config must contain datasets or clients[*].dataset")
    return [DatasetConfig(**item) for item in items]
