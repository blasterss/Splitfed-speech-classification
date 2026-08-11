from abc import ABC, abstractmethod
from pathlib import Path

from typing import Tuple, List, Dict, Any
import numpy as np

from tqdm import tqdm

from ..feature_extraction import FeatureExtraction
from ..feature_utils import FeatureUtils

from ...schema import DatasetConfig
from ...logger import logger


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
    ) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:

        logger.info(f"Loading data from {self.root}...")

        data, metadata = [], []

        files = sorted(self.root.rglob("*.wav"))

        if self.config.reduced:
            files = files[: self.config.reduced_size or 1000]

        iterator = tqdm(files, desc=f"Loading {self.config.name}", unit="file")

        failed_files = []

        for file_path in iterator:
            try:
                label = self.parse_label(file_path.name)
                actor_id = self.parse_actor_id(file_path.name)
                sex = self.parse_sex(file_path.name)

                y, sr = FeatureUtils.load_audio(
                    file_path,
                    sample_rate=self.SAMPLING_RATE,
                    target_sample_rate=self.config.target_sample_rate,
                )

                features = FeatureExtraction.get_all_features(
                    y,
                    sr,
                    feature_mode=feature_mode,
                    feature_names=self.config.feature_names,
                )

                data.append(features)

                metadata.append(
                    {
                        "label": label,
                        "actor_id": actor_id,
                        "sex": sex,
                        "dataset": str(self.config.name),
                    }
                )

            except Exception as e:
                logger.warning(f"Error: {file_path.name} - {e}")
                failed_files.append(str(file_path))

        if failed_files:
            logger.warning(f"Failed to load {len(failed_files)} files")

        if feature_mode == "stacked":
            padded_data = self._pad_stacked(data)
        else:
            padded_data = data

        logger.info(f"Loading completed. Total files: {len(padded_data)}")

        if padded_data:
            first = padded_data[0]

            if isinstance(first, np.ndarray):
                logger.info(f"Feature shape (stacked): {first.shape}")

            elif isinstance(first, list):
                shapes = [f.shape for f in first]

                logger.info("Multi-channel features:")

                for name, shape in zip(self.config.feature_names, shapes):
                    logger.info(f"  {name}: {shape}")

            else:
                logger.warning(f"Unknown data format: {type(first)}")

        return padded_data, metadata

    @staticmethod
    def _pad_stacked(features: List[np.ndarray]) -> List[np.ndarray]:
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
    def _pad_multichannel(features: List[np.ndarray]) -> List[np.ndarray]:
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
