import yaml

from ..logger import logger

from pathlib import Path
from typing import Any, Dict


def read_yaml(path, verbose: bool = False) -> Dict[Any, Any]:
    """
    Reads a yaml file, and returns a dict.

    Args:
        path:
            Path to the yaml file (Path, str, or importlib.resources path).
        verbose (bool):
            Whether to do any info logs.

    Returns:
        Dict[Any, Any]:
            The yaml content as a dict.

    Raises:
        ValueError:
            If the file is not a YAML file.
        FileNotFoundError:
            If the file is not found.
        yaml.YAMLError:
            If there is an error parsing the yaml file.
    """

    if isinstance(path, str):
        path = Path(path)

    path_str = str(path)
    if not any(path_str.endswith(ext) for ext in [".yaml", ".yml"]):
        msg = f"The file {path} is not a YAML file"
        logger.error(msg)
        raise ValueError(msg)

    try:
        if hasattr(path, "read_text") and not isinstance(path, Path):
            # importlib.resources path object
            content_text = path.read_text()
            content = yaml.safe_load(content_text)
        else:
            # Regular Path object
            with open(path, "r") as file:
                content = yaml.safe_load(file)

        if verbose:
            logger.info(f"YAML file {path} has been loaded")
        return content
    except FileNotFoundError as e:
        msg = f"File {path} not found"
        logger.error(f"{msg}: {e}")
        raise FileNotFoundError(msg) from e
    except yaml.YAMLError as e:
        msg = f"Error parsing YAML file {path}"
        logger.error(f"{msg}: {e}")
        raise yaml.YAMLError(msg) from e
    except Exception as e:
        msg = f"An unexpected error occurred while reading YAML file {path}"
        logger.error(f"{msg}: {e}")
        raise Exception(msg) from e


def save_yaml(path: Path, data: Dict, verbose: bool = True) -> None:
    """
    Writes a dictionary to a YAML file.

    Args:
        path (Path):
            Path to the YAML file where the data will be written.
        data (Dict):
            The dictionary content to write to the YAML file.
        verbose (bool, optional):
            Whether to log informational messages. Defaults to True.

    Raises:
        ValueError:
            If the file is not a YAML file.
        IOError:
            If there is an error writing to the file.
        yaml.YAMLError:
            If there is an error serializing the dictionary to YAML.
        Exception:
            If an unexpected error occurs.
    """
    if path.suffix not in [".yaml", ".yml"]:
        msg = f"The file {path} is not a YAML file."
        logger.error(msg)
        raise ValueError(msg)

    try:
        with open(path, "w") as file:
            yaml.safe_dump(
                data, file, sort_keys=False, default_flow_style=False
            )
        if verbose:
            logger.info(f"YAML file {path} has been written successfully.")
    except IOError as e:
        msg = f"Error writing to file {path}."
        logger.error(f"{msg}: {e}")
        raise IOError(msg) from e
    except yaml.YAMLError as e:
        msg = f"Error serializing data to YAML file {path}."
        logger.error(f"{msg}: {e}")
        raise yaml.YAMLError(msg) from e
    except Exception as e:
        msg = f"An unexpected error occurred while writing YAML file {path}."
        logger.error(f"{msg}: {e}")
        raise Exception(msg) from e
