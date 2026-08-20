import argparse
import copy
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.multiprocessing as mp
import yaml

from .logger import logger
from .schema import ConfigSchema
from .splitfed.controller import TrainingController
from .utils.common import read_yaml, save_yaml
from .utils.training import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--config-file", type=str, required=True)
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
    args = parser.parse_args()

    logger.info("=== READING CONFIG ===")
    raw_config = apply_cli_overrides(
        read_yaml(args.config_file, verbose=1), args.overrides
    )

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
            _save_run_metadata(
                config, model_path, cli_overrides=args.overrides
            )
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


def apply_cli_overrides(raw_config: dict, overrides: list[str]) -> dict:
    """Apply typed, existing-path-only CLI overrides to a raw config."""
    resolved = copy.deepcopy(raw_config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(
                f"Invalid override {override!r}; expected PATH=VALUE"
            )
        raw_path, raw_value = override.split("=", 1)
        parts = raw_path.split(".")
        if not raw_path or any(not part for part in parts):
            raise ValueError(f"Invalid override path {raw_path!r}")

        target = resolved
        for part in parts[:-1]:
            target = _descend_override_path(target, part, raw_path)

        leaf = parts[-1]
        value = yaml.safe_load(raw_value)
        if isinstance(target, dict):
            if leaf not in target:
                raise ValueError(f"Unknown override path {raw_path!r}")
            target[leaf] = value
        elif isinstance(target, list):
            index = _override_list_index(leaf, raw_path)
            if index >= len(target):
                raise ValueError(f"Unknown override path {raw_path!r}")
            target[index] = value
        else:
            raise ValueError(f"Unknown override path {raw_path!r}")
    return resolved


def _descend_override_path(target, part: str, raw_path: str):
    if isinstance(target, dict):
        if part not in target:
            raise ValueError(f"Unknown override path {raw_path!r}")
        return target[part]
    if isinstance(target, list):
        index = _override_list_index(part, raw_path)
        if index >= len(target):
            raise ValueError(f"Unknown override path {raw_path!r}")
        return target[index]
    raise ValueError(f"Unknown override path {raw_path!r}")


def _override_list_index(part: str, raw_path: str) -> int:
    try:
        index = int(part)
    except ValueError as exc:
        raise ValueError(
            f"Override path {raw_path!r} requires a numeric list index"
        ) from exc
    if index < 0:
        raise ValueError(f"Unknown override path {raw_path!r}")
    return index


def _save_run_metadata(
    config: ConfigSchema,
    artifact_path: Path,
    *,
    cli_overrides: list[str] | None = None,
) -> None:
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
            "configuration": {
                "cli_overrides": cli_overrides or [],
            },
        },
    )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
