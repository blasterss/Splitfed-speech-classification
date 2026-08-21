from ...logger import get_logger
from ...schema import DatasetConfig, DatasetType
from .base_processor import BaseDatasetLoader
from .crema_d_processor import CremaDLoader
from .ravdess_processor import RavdessLoader
from .savee_processor import SaveeLoader

logger = get_logger(__name__)


class DatasetLoaderFactory:
    """
    Factory class for dataset loaders.
    """

    _REGISTRY = {
        DatasetType.crema_d: CremaDLoader,
        DatasetType.ravdess: RavdessLoader,
        DatasetType.savee: SaveeLoader,
    }

    @classmethod
    def create(cls, config: DatasetConfig) -> BaseDatasetLoader:
        """
        Creates and returns a dataset loader instance
        based on the dataset configuration.
        """

        logger.info(f"Loading dataset {config.name}")

        loader_cls = cls._REGISTRY.get(config.name)

        if loader_cls is None:
            logger.error(f"Unsupported dataset: {config.name}")
            raise ValueError(f"Unsupported dataset: {config.name}")

        return loader_cls(config)
