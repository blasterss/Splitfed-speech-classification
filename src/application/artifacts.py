"""Run artifact persistence and reproducibility metadata."""

import copy
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from ..logger import logger
from ..schema import ConfigSchema
from ..transport.message import MESSAGE_PROTOCOL, MESSAGE_PROTOCOL_VERSION
from ..utils.config import save_yaml
from ..utils.persistence import ArtifactPaths


def finalize_run_artifacts(
    controller,
    config: ConfigSchema,
    *,
    configuration_provenance: dict | None,
) -> None:
    """Stop worker roles and persist every available run artifact."""
    _stop_servers(controller)
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
    _save_first_failure(controller.first_failure, artifact_paths.diagnostics)
    _save_models(controller, artifact_paths.checkpoints)


def _stop_servers(controller) -> None:
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


def _save_models(controller, checkpoint_path: Path) -> None:
    owners = (
        (controller.split_server, "split-server"),
        (controller.fed_server, "fed-server"),
        (controller.centralized_trainer, "centralized"),
    )
    for owner, name in owners:
        if owner is None:
            continue
        try:
            owner.save(checkpoint_path)
        except RuntimeError as exc:
            logger.warning("Could not save %s weights: %s", name, exc)


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
    return _run_git("rev-parse", "HEAD") or None


def _git_dirty() -> bool | None:
    """Return checkout dirty state, or null outside an available checkout."""
    result = _run_git("status", "--porcelain")
    return None if result is None else bool(result)


def _run_git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


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
