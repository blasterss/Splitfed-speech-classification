from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from ...logger import logger
from ...schema import DatasetConfig
from ..audio import load_audio
from ..feature_extraction import FeatureExtraction


class BaseDatasetLoader(ABC):
    """
    Base class for loading and preparing an audio dataset.
    """

    SAMPLING_RATE = 16000

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.root = Path(config.root)

    @abstractmethod
    def parse_label(self, filename: str) -> int:
        """
        Returns the label for a specific file.
        """
        ...

    @abstractmethod
    def parse_actor_id(self, filename: str) -> str:
        """
        Returns the actor ID for a specific file.
        """
        ...

    @abstractmethod
    def parse_sex(self, filename: str) -> str:
        """
        Returns the actor's sex for a specific file.
        """
        ...

    def load(
        self, feature_mode: str = "stacked"
    ) -> tuple[list[np.ndarray], list[dict[str, Any]]]:

        if feature_mode not in {"stacked", "multi_channel"}:
            raise ValueError(f"Unsupported feature mode: {feature_mode}")

        logger.info(f"Loading data from {self.root}...")

        data, metadata = [], []

        files = sorted(self.root.rglob("*.wav"))

        if self.config.reduced:
            files = files[: self.config.reduced_size or 1000]

        iterator = tqdm(files, desc=f"Loading {self.config.name}", unit="file")

        failed_files = []
        failure_reasons = Counter()

        for file_path in iterator:
            try:
                label = self.parse_label(file_path.name)
                actor_id = self.parse_actor_id(file_path.name)
                sex = self.parse_sex(file_path.name)

                y, sr = load_audio(
                    file_path,
                    target_sample_rate=self.config.target_sample_rate,
                )

                features = FeatureExtraction.get_all_features(
                    y,
                    sr,
                    feature_mode=feature_mode,
                    feature_names=self.config.feature_names,
                )

                valid_frames = _feature_frame_count(features)
                sample_metadata = {
                    "label": label,
                    "actor_id": actor_id,
                    "sex": sex,
                    "dataset": str(self.config.name),
                    "valid_frames": valid_frames,
                }
                data.append(features)
                metadata.append(sample_metadata)

            except Exception as e:
                logger.warning(f"Error: {file_path.name} - {e}")
                failed_files.append(str(file_path))
                failure_reasons[type(e).__name__] += 1

        if failed_files:
            logger.warning(
                "Failed to load %d files: %s",
                len(failed_files),
                dict(sorted(failure_reasons.items())),
            )

        if feature_mode == "stacked":
            padded_data = self._pad_stacked(data)
        else:
            padded_data = data

        logger.info(f"Loading completed. Total files: {len(padded_data)}")
        self.last_load_report = {
            "discovered": len(files),
            "loaded": len(padded_data),
            "failed": len(failed_files),
            "failure_reasons": dict(sorted(failure_reasons.items())),
        }

        if padded_data:
            first = padded_data[0]

            if isinstance(first, np.ndarray):
                logger.info(f"Feature shape (stacked): {first.shape}")

            elif isinstance(first, list):
                shapes = [f.shape for f in first]

                logger.info("Multi-channel features:")

                for name, shape in zip(
                    self.config.feature_names, shapes, strict=True
                ):
                    logger.info(f"  {name}: {shape}")

            else:
                logger.warning(f"Unknown data format: {type(first)}")

        return padded_data, metadata

    @staticmethod
    def _pad_stacked(features: list[np.ndarray]) -> list[np.ndarray]:
        """
        Padding for stacked format [F, T]
        """
        if not features:
            return []

        max_time = max(feat.shape[1] for feat in features)

        padded = []

        for feat in features:
            curr_time = feat.shape[1]

            if curr_time < max_time:
                pad_width = ((0, 0), (0, max_time - curr_time))

                feat = np.pad(
                    feat, pad_width, mode="constant", constant_values=0
                )

            padded.append(feat)

        return padded

    @staticmethod
    def _pad_multichannel(features: list[np.ndarray]) -> list[np.ndarray]:
        """
        Padding for multi-channel format [C, H, T]
        """
        if not features:
            return []

        max_time = max(feat.shape[-1] for feat in features)

        padded = []

        for feat in features:
            curr_time = feat.shape[-1]

            if curr_time < max_time:
                pad_width = [(0, 0)] * (feat.ndim - 1) + [
                    (0, max_time - curr_time)
                ]

                feat = np.pad(
                    feat, pad_width, mode="constant", constant_values=0
                )

            padded.append(feat)

        return padded


def _feature_frame_count(features: np.ndarray | list[np.ndarray]) -> int:
    """Validate one extracted sample and return its shared time length."""
    if isinstance(features, np.ndarray):
        if features.ndim < 1 or features.shape[-1] <= 0:
            raise ValueError("Extracted features have no time frames")
        return int(features.shape[-1])

    if not isinstance(features, list) or not features:
        raise TypeError("Extracted features must be an array or channel list")
    if any(
        not isinstance(channel, np.ndarray) or channel.ndim < 1
        for channel in features
    ):
        raise TypeError("Every feature channel must be a non-scalar array")
    frame_counts = {int(channel.shape[-1]) for channel in features}
    if len(frame_counts) != 1:
        raise ValueError(
            "Multi-channel features have inconsistent time frames"
        )
    frame_count = next(iter(frame_counts))
    if frame_count <= 0:
        raise ValueError("Extracted features have no time frames")
    return frame_count
