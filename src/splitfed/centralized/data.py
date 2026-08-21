"""Dataset-shape validation and collation for centralized training."""

import torch
import torch.nn.functional as functional
from torch.utils.data import default_collate


def _validate_centralized_shapes(datasets: list) -> int:
    samples = [dataset[0][0] for dataset in datasets if len(dataset) > 0]
    if not samples:
        raise ValueError("Centralized mode requires at least one sample")
    shapes = {tuple(sample.shape) for sample in samples}
    if any(sample.ndim != 2 for sample in samples):
        raise ValueError(
            "Centralized datasets require [features, time] samples: "
            f"{shapes}"
        )
    feature_counts = {sample.shape[0] for sample in samples}
    if len(feature_counts) != 1:
        raise ValueError(
            "Centralized datasets require identical feature counts: "
            f"{shapes}"
        )
    return next(iter(feature_counts))


def _pad_feature_batch(batch):
    """Right-pad variable temporal lengths within one centralized batch."""
    features, labels = zip(*batch, strict=True)
    max_time = max(feature.shape[-1] for feature in features)
    padded = [
        functional.pad(feature, (0, max_time - feature.shape[-1]))
        for feature in features
    ]
    return torch.stack(padded), default_collate(labels)
