import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from ..dataset.processors.factory import DatasetLoaderFactory
from ..logger import logger
from ..schema import DatasetConfig


class EmotionalDataset(Dataset):
    """
    PyTorch dataset for emotional speech classification.
    Supports both stacked and multi-channel feature formats.
    """

    def __init__(
        self, data: np.ndarray, labels: np.ndarray, mean=None, std=None
    ):
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).float()

        # --- normalization parameters ---
        self.mean = None if mean is None else torch.from_numpy(mean).float()

        self.std = None if std is None else torch.from_numpy(std).float()

        # Prevent division by zero
        if self.std is not None:
            self.std = torch.clamp(self.std, min=1e-8)

        # Detect feature format
        if self.data.dim() == 3:
            self.format = "stacked"

        elif self.data.dim() == 4:
            self.format = "multi_channel"

        else:
            raise ValueError(f"Unsupported data dimension: {self.data.dim()}")

        logger.debug(
            f"Dataset format: {self.format}, data shape: {self.data.shape}"
        )

    def __len__(self):
        """
        Returns dataset size.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns one sample and its label.
        """

        x = self.data[idx]
        y = self.labels[idx]

        # Apply normalization
        if self.mean is not None and self.std is not None:
            x = (x - self.mean[:, None]) / self.std[:, None]

        return x, y


class ConflictEmotionalDataset:
    """
    Wrapper class for loading, splitting,
    normalizing, and preparing emotional datasets.
    """

    def __init__(self, config: DatasetConfig):

        # Create dataset loader
        loader = DatasetLoaderFactory.create(config)

        # Load raw features and metadata
        data, metadata = loader.load()

        if not data or not metadata or len(data) != len(metadata):
            raise ValueError(
                "Dataset extraction must produce matching non-empty data and "
                "metadata"
            )

        data = np.stack(data)

        labels = np.asarray([item["label"] for item in metadata])

        unsupported_labels = set(np.unique(labels)) - {0, 1}
        if unsupported_labels:
            raise ValueError(
                f"Dataset contains unsupported binary labels: "
                f"{sorted(unsupported_labels)}"
            )

        actor_ids = np.asarray([item["actor_id"] for item in metadata])

        # Split dataset by actor IDs
        unique_actors = np.unique(actor_ids)
        if len(unique_actors) < 2:
            raise ValueError(
                "Actor-disjoint split requires at least two unique actors"
            )

        train_actors, test_actors = train_test_split(
            unique_actors,
            test_size=config.test_size,
            random_state=config.split_seed,
        )
        self.train_actor_ids = tuple(train_actors.tolist())
        self.test_actor_ids = tuple(test_actors.tolist())

        train_mask = np.isin(actor_ids, train_actors)
        test_mask = np.isin(actor_ids, test_actors)

        train_data = data[train_mask]
        test_data = data[test_mask]
        train_labels = labels[train_mask]
        test_labels = labels[test_mask]
        if set(np.unique(train_labels)) != {0, 1}:
            raise ValueError(
                "Training split must contain both binary classes 0 and 1"
            )
        self.coverage = {
            "train": _coverage(train_labels, train_actors),
            "test": _coverage(test_labels, test_actors),
        }

        # --------- NORMALIZATION (TRAIN ONLY) ---------

        # Compute normalization statistics
        # over frequency and time dimensions
        mean = train_data.mean(axis=(0, 2))
        std = train_data.std(axis=(0, 2)) + 1e-8

        # Create PyTorch datasets
        self.train_dataset = EmotionalDataset(
            train_data, train_labels, mean, std
        )

        self.test_dataset = EmotionalDataset(test_data, test_labels, mean, std)

        logger.info("Dataset coverage: %s", self.coverage)

    def get_sample_weights(self) -> np.ndarray:
        """
        Computes sample weights for handling
        class imbalance in the training dataset.
        """

        # Count samples for each class
        label_counts = np.bincount(self.train_dataset.labels.long().numpy())

        total_samples = len(self.train_dataset)

        # Inverse-frequency class weights
        class_weights = total_samples / (len(label_counts) * label_counts)

        # Assign weight to each sample
        sample_weights = class_weights[
            self.train_dataset.labels.long().numpy()
        ]

        return sample_weights


def _coverage(labels: np.ndarray, actors: np.ndarray) -> dict:
    return {
        "samples": int(len(labels)),
        "actors": int(len(actors)),
        "class_0": int((labels == 0).sum()),
        "class_1": int((labels == 1).sum()),
    }
