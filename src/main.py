import argparse
from pathlib import Path

import torch.multiprocessing as mp

from .logger import logger
from .schema import ConfigSchema
from .splitfed.controller import TrainingController
from .utils.common import read_yaml
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


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
