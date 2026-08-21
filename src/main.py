import argparse
import copy
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.multiprocessing as mp
import yaml

from .config_profiles import PROFILE_REGISTRY
from .logger import logger
from .schema import ConfigSchema
from .splitfed.controller import TrainingController
from .transport.message import MESSAGE_PROTOCOL, MESSAGE_PROTOCOL_VERSION
from .utils.common import read_yaml, save_yaml
from .utils.persistence import ArtifactPaths
from .utils.training import set_seed


def main():
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
    args = parser.parse_args()

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
    controller = TrainingController(config=config)
    _execute_training(
        controller,
        config,
        configuration_provenance=config_provenance,
    )


def _execute_training(
    controller: TrainingController,
    config: ConfigSchema,
    *,
    configuration_provenance: dict | None = None,
) -> None:
    """Run one controller lifecycle and always release manager resources."""
    setup_succeeded = False

    try:
        controller.setup()
        setup_succeeded = True
        controller.start_training()
    except KeyboardInterrupt:
        logger.info("Interrupted by user — shutting down")
    except Exception:
        logger.exception("Training failed with unhandled exception")
        raise
    finally:
        try:
            if setup_succeeded:
                _finalize_run(
                    controller,
                    config,
                    configuration_provenance=configuration_provenance,
                )
        finally:
            controller.teardown()


def _finalize_run(
    controller: TrainingController,
    config: ConfigSchema,
    *,
    configuration_provenance: dict | None,
) -> None:
    """Stop worker roles and persist available run artifacts."""
    stop_errors = []
    for server in (controller.split_server, controller.fed_server):
        if server is None:
            continue
        try:
            server.stop()
        except BaseException as exc:
            stop_errors.append(exc)
            logger.error(
                "Could not stop %s", type(server).__name__, exc_info=True
            )

    if stop_errors:
        raise stop_errors[0]

    if not config.models_save_path:
        return

    artifact_paths = ArtifactPaths.from_root(
        config.models_save_path, config.experiment.name
    )
    artifact_paths.mkdir()
    _save_resolved_config(config, artifact_paths.metadata)
    _save_run_metadata(
        config,
        artifact_paths.metadata,
        resolved_config_sha256=_config_sha256(config),
        configuration_provenance=configuration_provenance,
    )
    _save_dataset_manifest(
        artifact_paths.metadata,
        expected_client_ids=[client.client_id for client in config.clients],
        client_manifests=controller.dataset_manifests,
    )
    _save_first_failure(
        controller.first_failure,
        artifact_paths.diagnostics,
    )
    if controller.split_server is not None:
        try:
            controller.split_server.save(artifact_paths.checkpoints)
        except RuntimeError as exc:
            logger.warning("Could not save split-server weights: %s", exc)

    if controller.fed_server is not None:
        try:
            controller.fed_server.save(artifact_paths.checkpoints)
        except RuntimeError as exc:
            logger.warning("Could not save fed-server weights: %s", exc)

    if controller.centralized_trainer is not None:
        try:
            controller.centralized_trainer.save(artifact_paths.checkpoints)
        except RuntimeError as exc:
            logger.warning("Could not save centralized weights: %s", exc)


def _save_first_failure(failure: dict | None, diagnostics_path: Path) -> None:
    """Persist the first bounded structured failure, when one exists."""
    if failure is None:
        return
    save_yaml(diagnostics_path / "first_failure.yaml", failure)


def _save_resolved_config(config: ConfigSchema, artifact_path: Path) -> None:
    """Persist the validated, alias-preserving run configuration."""
    save_yaml(
        artifact_path / "resolved_config.yaml",
        config.model_dump(mode="json", by_alias=True),
    )


def _save_dataset_manifest(
    artifact_path: Path,
    *,
    expected_client_ids: list[int],
    client_manifests: dict[int, dict],
) -> None:
    """Persist bounded per-client dataset summaries and missing reports."""
    missing = sorted(set(expected_client_ids) - set(client_manifests))
    save_yaml(
        artifact_path / "dataset_manifest.yaml",
        {
            "schema_version": 1,
            "complete": not missing,
            "missing_client_ids": missing,
            "clients": [
                client_manifests[client_id]
                for client_id in sorted(client_manifests)
            ],
        },
    )


def _config_sha256(config: ConfigSchema | dict) -> str:
    """Return a stable hash of a validated config or raw mapping."""
    if isinstance(config, ConfigSchema):
        value = config.model_dump(mode="json", by_alias=True)
    else:
        value = config
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _file_sha256(path: Path) -> str | None:
    """Hash an optional file without making artifact writes depend on it."""
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str | None:
    """Return the current Git revision when running inside a checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _git_dirty() -> bool | None:
    """Return checkout dirty state, or null outside an available checkout."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return bool(result.stdout.strip())


def _implemented_policies(config: ConfigSchema) -> dict:
    aggregation = None
    if config.fed_server is not None:
        aggregation = {
            "name": config.fed_server.strategy.value,
            "version": "1",
        }
    return {
        "transport": {
            "name": config.experiment.transport.value,
            "version": "1",
        },
        "aggregation": aggregation,
        "scheduler": None,
    }


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


def resolve_raw_config(
    raw_config: dict,
    *,
    profile_name: str | None,
    overrides: list[str],
) -> tuple[dict, dict]:
    """Resolve profile, YAML and CLI layers in increasing precedence."""
    yaml_profile = raw_config.get("experiment", {}).get("profile")
    selected_profile = profile_name or yaml_profile
    profile_provenance = None
    if selected_profile is None:
        merged = copy.deepcopy(raw_config)
    else:
        try:
            profile = PROFILE_REGISTRY[selected_profile]
        except KeyError as exc:
            raise ValueError(
                f"Unknown experiment profile {selected_profile!r}"
            ) from exc
        merged = _deep_merge(profile.config_copy(), raw_config)
        merged.setdefault("experiment", {})["profile"] = selected_profile
        profile_provenance = {
            "name": selected_profile,
            "version": profile.version,
            "source": "cli" if profile_name is not None else "yaml",
        }

    resolved = apply_cli_overrides(merged, overrides)
    return resolved, {
        "profile": profile_provenance,
        "cli_overrides": list(overrides),
        "overrides": [
            {
                "source": "cli",
                "path": override.split("=", 1)[0],
                "expression": override,
            }
            for override in overrides
        ],
    }


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge mappings while replacing non-mapping values."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


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
    configuration_provenance: dict | None = None,
    resolved_config_sha256: str | None = None,
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
    configuration = copy.deepcopy(configuration_provenance) or {
        "profile": None,
        "cli_overrides": cli_overrides or [],
        "overrides": [
            {
                "source": "cli",
                "path": override.split("=", 1)[0],
                "expression": override,
            }
            for override in (cli_overrides or [])
        ],
    }
    configuration["resolved_config_sha256"] = (
        resolved_config_sha256 or _config_sha256(config)
    )
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
                "git_revision": _git_revision(),
                "git_dirty": _git_dirty(),
                "dependency_lock_sha256": _file_sha256(Path("uv.lock")),
            },
            "runtime": {
                "backend": "local_multiprocessing",
                "start_method": "spawn",
                "container_image_digest": None,
            },
            "seed_tree": seed_tree,
            "configuration": configuration,
            "policies": _implemented_policies(config),
            "protocols": {
                "message": {
                    "name": MESSAGE_PROTOCOL,
                    "version": MESSAGE_PROTOCOL_VERSION,
                }
            },
        },
    )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
