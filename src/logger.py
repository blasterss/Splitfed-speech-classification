"""
Logging configuration.

This module sets up logging based on the logger.yaml configuration file.
"""

import logging
import logging.config
from importlib.resources import files
from pathlib import Path
from typing import Optional

import yaml


def setup_logging(
    config_path: Optional[Path] = None,
    default_level: int = logging.INFO,
) -> None:
    """
    Setup logging configuration.

    Args:
        config_path (Optional[Path]):
            Path to logger.yaml config file.
        default_level (int):
            Default logging level if config file not found.
    """
    if config_path is None:
        config_path = Path("configs/logger.yaml")

    try:
        if hasattr(config_path, "read_text") and not isinstance(
            config_path, Path
        ):
            # importlib.resources path object
            config_text = config_path.read_text()
            config = yaml.safe_load(config_text)
            logging.config.dictConfig(config)
        elif config_path.exists():
            # Regular Path object
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            logging.config.dictConfig(config)
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")
    except Exception as e:
        logging.basicConfig(level=default_level)
        logging.getLogger(__name__).warning(
            f"Failed to load logging config from {config_path}: {e}. "
            "Using basic configuration."
        )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.

    Args:
        name (str):
            Logger name, typically __name__ of the calling module.

    Returns:
        logging.Logger:
            Logger instance.
    """
    return logging.getLogger(name)


setup_logging()

logger = get_logger("splitfed_asr")
