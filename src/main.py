import argparse
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.multiprocessing as mp

from .logger import logger
from .schema import ConfigSchema
from .splitfed.controller import TrainingController
from .utils.common import read_yaml, save_yaml
from .utils.training import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--config-file", type=str, required=True)
    args = parser.parse_args()

    logger.info("=== READING CONFIG ===")
    raw_config = read_yaml(args.config_file, verbose=1)

    logger.info("=== VALIDATING CONFIG ===")
    config = ConfigSchema(**raw_config)

    set_seed(config.experiment.seed)

    logger.info("=== SETTING UP CONTROLLER ===")
    controller = TrainingController(config=config)
    controller.setup()

    try:
        controller.start_training()
    except KeyboardInterrupt:
        logger.info("Interrupted by user — shutting down")
    except Exception:
        logger.exception("Training failed with unhandled exception")
        raise
    finally:
        # Ensure server processes are always cleaned up
        if controller.split_server is not None:
            controller.split_server.stop()
        if controller.fed_server is not None:
            controller.fed_server.stop()

        if config.models_save_path:
            model_path = Path(config.models_save_path)
            model_path.mkdir(parents=True, exist_ok=True)

            experiment_name = config.experiment.name
            model_path = model_path / experiment_name
            model_path.mkdir(parents=True, exist_ok=True)
            _save_resolved_config(config, model_path)
            _save_run_metadata(config, model_path)
            if controller.split_server is not None:
                try:
                    controller.split_server.save(model_path)
                except RuntimeError as e:
                    logger.warning(
                        "Could not save split-server weights: %s", e
                    )

            if controller.fed_server is not None:
                try:
                    controller.fed_server.save(model_path)
                except RuntimeError as e:
                    logger.warning("Could not save fed-server weights: %s", e)

            if controller.centralized_trainer is not None:
                try:
                    controller.centralized_trainer.save(model_path)
                except RuntimeError as e:
                    logger.warning("Could not save centralized weights: %s", e)


def _save_resolved_config(config: ConfigSchema, artifact_path: Path) -> None:
    """Persist the validated, alias-preserving run configuration."""
    save_yaml(
        artifact_path / "resolved_config.yaml",
        config.model_dump(mode="json", by_alias=True),
    )


def _save_run_metadata(config: ConfigSchema, artifact_path: Path) -> None:
    """Persist environment provenance and the configured seed tree."""
    seed_tree = {
        "experiment": config.experiment.seed,
        "training": config.training.seed,
        "clients": {
            str(client.client_id): {
                "runtime": client.runtime.seed,
                "dataset_split": client.dataset.split_seed,
            }
            for client in config.clients
        },
        "split_server": (
            config.split_server.seed if config.split_server else None
        ),
        "fed_server": config.fed_server.seed if config.fed_server else None,
    }
    save_yaml(
        artifact_path / "run_metadata.yaml",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "environment": {
                "python": sys.version.split()[0],
                "pytorch": str(torch.__version__),
                "platform": platform.platform(),
                "cuda_available": torch.cuda.is_available(),
                "cuda_runtime": (
                    str(torch.version.cuda) if torch.version.cuda else None
                ),
            },
            "seed_tree": seed_tree,
        },
    )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
