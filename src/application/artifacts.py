"""Run artifact persistence and reproducibility metadata."""

import copy
import csv
import hashlib
import json
import platform
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from ..experiments.checkpoint_evaluation import CORPUS_AGGREGATE_METRICS
from ..experiments.metrics import (
    BINARY_METRICS_SCHEMA_VERSION,
    evaluate_binary_predictions,
)
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
    _save_resource_metrics(
        getattr(controller, "resource_metrics", []), artifact_paths.metrics
    )
    _save_splitfed_quality(
        artifact_paths.metrics,
        config,
        created_after=getattr(controller, "run_started_at", None),
    )
    _save_models(controller, artifact_paths.checkpoints)


def _save_resource_metrics(metrics: list[dict], metrics_path: Path) -> None:
    """Persist process/round measurements and a compact run summary."""
    if not metrics:
        return
    ordered = sorted(
        metrics,
        key=lambda item: (
            str(item.get("role")),
            str(item.get("client_id")),
            item.get("round") or 0,
            str(item.get("phase")),
        ),
    )
    fieldnames = list(dict.fromkeys(key for row in ordered for key in row))
    with (metrics_path / "resource_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ordered)

    _save_resource_averages(ordered, metrics_path)

    numeric_maxima = {}
    for field in (
        "peak_rss_bytes",
        "peak_cuda_allocated_bytes",
        "peak_cuda_reserved_bytes",
    ):
        values = [row[field] for row in ordered if row.get(field) is not None]
        numeric_maxima[f"max_{field}"] = max(values) if values else None
    save_yaml(
        metrics_path / "resource_summary.yaml",
        {
            "schema_version": 1,
            "measurement_scope": (
                "per-process peaks; values must not be summed as concurrent "
                "host usage"
            ),
            "event_count": len(ordered),
            "experiment_wall_time_seconds": next(
                (
                    row["wall_time_seconds"]
                    for row in ordered
                    if row["role"] == "controller"
                    and row["phase"] == "experiment"
                ),
                None,
            ),
            "summed_process_interval_seconds": sum(
                row["wall_time_seconds"] for row in ordered
            ),
            "total_cpu_user_seconds": sum(
                row["cpu_user_seconds"] for row in ordered
            ),
            "total_cpu_system_seconds": sum(
                row["cpu_system_seconds"] for row in ordered
            ),
            "total_transport_bytes": sum(
                row.get("bytes_sent", 0) for row in ordered
            ),
            "total_transport_messages": sum(
                row.get("messages_sent", 0) for row in ordered
            ),
            **numeric_maxima,
            "roles": sorted({row["role"] for row in ordered}),
        },
    )


def _save_resource_averages(metrics: list[dict], metrics_path: Path) -> None:
    """Persist average client/role resource intervals without merging peaks."""
    groups = defaultdict(list)
    for row in metrics:
        group = (
            str(row.get("role")),
            row.get("client_id"),
            str(row.get("phase")),
        )
        groups[group].append(row)

    rows = []
    for (role, client_id, phase), group in sorted(groups.items()):
        row = {
            "role": role,
            "client_id": client_id,
            "phase": phase,
            "event_count": len(group),
        }
        for field in (
            "wall_time_seconds",
            "cpu_user_seconds",
            "cpu_system_seconds",
            "samples",
            "batches",
            "samples_per_second",
            "batches_per_second",
        ):
            values = [
                item[field] for item in group if item.get(field) is not None
            ]
            row[f"avg_{field}"] = sum(values) / len(values) if values else None
        for field in (
            "peak_rss_bytes",
            "peak_cuda_allocated_bytes",
            "peak_cuda_reserved_bytes",
        ):
            values = [
                item[field] for item in group if item.get(field) is not None
            ]
            row[f"max_{field}"] = max(values) if values else None
        rows.append(row)

    fieldnames = list(rows[0]) if rows else []
    with (metrics_path / "resource_by_client.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _save_splitfed_quality(
    metrics_path: Path,
    config: ConfigSchema,
    *,
    created_after: float | None = None,
) -> None:
    """Aggregate final local evaluations into one quality report."""
    if config.training.mode.value not in {"split", "splitfed"}:
        return
    rows = []
    corpus_by_client = {
        str(client.client_id): getattr(
            client.dataset.name, "value", client.dataset.name
        )
        for client in getattr(config, "clients", [])
    }
    expected_round = config.training.num_rounds
    missing_client_ids = []
    for client_id in sorted(corpus_by_client, key=int):
        filename = f"Client{client_id}_round_{expected_round}_eval.csv"
        path = metrics_path / filename
        if not path.is_file() or (
            created_after is not None and path.stat().st_mtime < created_after
        ):
            missing_client_ids.append(client_id)
            continue
        labels = []
        probabilities = []
        with path.open(newline="", encoding="utf-8") as source:
            for item in csv.DictReader(source):
                labels.append(int(float(item["labels"])))
                probabilities.append(float(item["probs"]))
        rows.append(
            {
                "client_id": client_id,
                "corpus": corpus_by_client.get(client_id),
                "round": expected_round,
                **evaluate_binary_predictions(labels, probabilities),
            }
        )

    metric_fields = ("client_id", "corpus", "round") + tuple(
        evaluate_binary_predictions([], []).keys()
    )
    with (metrics_path / "splitfed_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.DictWriter(output, fieldnames=metric_fields)
        writer.writeheader()
        writer.writerows(rows)
    total_samples = sum(row["num_samples"] for row in rows)
    save_yaml(
        metrics_path / "splitfed_summary.yaml",
        {
            "schema_version": 1,
            "metrics_schema_version": BINARY_METRICS_SCHEMA_VERSION,
            "mode": config.training.mode.value,
            "round": expected_round,
            "complete": not missing_client_ids,
            "missing_client_ids": missing_client_ids,
            "rows": rows,
            "macro_average": {
                field: _optional_mean(rows, field)
                for field in CORPUS_AGGREGATE_METRICS
            },
            "sample_weighted_average": {
                field: _optional_weighted_mean(rows, field, total_samples)
                for field in CORPUS_AGGREGATE_METRICS
            },
        },
    )


def _optional_mean(rows: list[dict], field: str) -> float | None:
    values = [row[field] for row in rows if row.get(field) is not None]
    if not rows or len(values) != len(rows):
        return None
    return sum(values) / len(values)


def _optional_weighted_mean(
    rows: list[dict], field: str, total_samples: int
) -> float | None:
    has_missing_value = any(row.get(field) is None for row in rows)
    if not rows or not total_samples or has_missing_value:
        return None
    return sum(row[field] * row["num_samples"] for row in rows) / total_samples


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
            "buffer_policy": {
                "name": "weighted_floating_state_largest_nonfloating_v1",
                "version": "1",
            },
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
