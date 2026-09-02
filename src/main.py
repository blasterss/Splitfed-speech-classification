"""Command-line entry point for SecureASR training."""

import argparse

import torch.multiprocessing as mp

from .application.configuration import resolve_raw_config
from .application.lifecycle import execute_training
from .config_profiles import PROFILE_REGISTRY
from .logger import logger
from .schema import ConfigSchema
from .splitfed.controller import TrainingController
from .utils.config import read_yaml
from .utils.training import set_seed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--config-file", type=str, required=True)
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_REGISTRY),
        help="Apply a versioned built-in experiment profile before YAML.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help=(
            "Override an existing config value using a dotted path; VALUE is "
            "parsed as YAML. Repeat for multiple overrides."
        ),
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    logger.info("=== READING CONFIG ===")
    raw_config, config_provenance = resolve_raw_config(
        read_yaml(args.config_file, verbose=1),
        profile_name=args.profile,
        overrides=args.overrides,
    )

    logger.info("=== VALIDATING CONFIG ===")
    config = ConfigSchema(**raw_config)
    set_seed(config.experiment.seed)

    logger.info("=== SETTING UP CONTROLLER ===")
    execute_training(
        TrainingController(config=config),
        config,
        configuration_provenance=config_provenance,
    )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
