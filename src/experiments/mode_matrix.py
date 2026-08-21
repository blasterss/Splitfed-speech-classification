from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import torch.multiprocessing as mp

from ..main import _execute_training
from ..schema import ConfigSchema
from ..splitfed.controller import TrainingController
from ..utils.config import read_yaml, save_yaml
from ..utils.training import set_seed

MODE_VARIANTS = (
    "centralized",
    "federated",
    "split-shared",
    "split-personalized",
    "splitfed",
)


def build_mode_config(
    base: dict,
    variant: str,
    *,
    rounds: int,
    artifact_root: str | None = None,
) -> dict:
    """Derive one strict mode topology from a splitfed base configuration."""
    if variant not in MODE_VARIANTS:
        raise ValueError(f"Unknown mode variant {variant!r}")
    if rounds <= 0:
        raise ValueError("rounds must be positive")

    resolved = copy.deepcopy(base)
    base_name = resolved["experiment"]["name"]
    resolved["experiment"]["name"] = f"{base_name}-{variant}-r{rounds}"
    resolved["training"]["num_rounds"] = rounds
    resolved["training"]["eval_every"] = rounds
    resolved["training"]["fed_every"] = rounds
    if artifact_root is not None:
        resolved["models_save_path"] = str(Path(artifact_root).absolute())

    split_server = copy.deepcopy(base.get("split_server"))
    fed_server = copy.deepcopy(base.get("fed_server"))
    channels = base.get("channels", {})
    split_channels = {
        name: copy.deepcopy(channels[name])
        for name in ("split_uplink", "split_downlink")
    }
    fed_channels = {
        name: copy.deepcopy(channels[name])
        for name in ("federated_uplink", "federated_downlink")
    }

    if variant == "centralized":
        resolved["training"]["mode"] = "centralized"
        resolved["split_server"] = None
        resolved["fed_server"] = None
        resolved["channels"] = {}
    elif variant == "federated":
        resolved["training"]["mode"] = "federated"
        resolved["split_server"] = None
        fed_server["aggregation_freq"] = rounds
        resolved["fed_server"] = fed_server
        resolved["channels"] = fed_channels
    elif variant.startswith("split-"):
        resolved["training"]["mode"] = "split"
        split_server["model_scope"] = variant.removeprefix("split-")
        resolved["split_server"] = split_server
        resolved["fed_server"] = None
        resolved["channels"] = split_channels
    else:
        resolved["training"]["mode"] = "splitfed"
        split_server["model_scope"] = "shared"
        fed_server["aggregation_freq"] = rounds
        resolved["split_server"] = split_server
        resolved["fed_server"] = fed_server
        resolved["channels"] = {**split_channels, **fed_channels}

    return resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--artifact-root")
    parser.add_argument(
        "--mode",
        action="append",
        choices=MODE_VARIANTS,
        dest="modes",
        help="Run only selected variants; repeat as needed.",
    )
    args = parser.parse_args()

    raw = read_yaml(args.config_file)
    modes = args.modes or list(MODE_VARIANTS)
    results = []
    for variant in modes:
        mode_raw = build_mode_config(
            raw,
            variant,
            rounds=args.rounds,
            artifact_root=args.artifact_root,
        )
        config = ConfigSchema(**mode_raw)
        set_seed(config.experiment.seed)
        started = time.monotonic()
        try:
            _execute_training(
                TrainingController(config),
                config,
                configuration_provenance={
                    "profile": None,
                    "cli_overrides": [],
                    "overrides": [
                        {
                            "source": "mode_matrix",
                            "path": "training.mode",
                            "expression": variant,
                        },
                        {
                            "source": "mode_matrix",
                            "path": "training.num_rounds",
                            "expression": str(args.rounds),
                        },
                    ],
                },
            )
        except Exception as exc:
            results.append(
                {
                    "variant": variant,
                    "experiment": config.experiment.name,
                    "status": "failed",
                    "elapsed_seconds": time.monotonic() - started,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        else:
            results.append(
                {
                    "variant": variant,
                    "experiment": config.experiment.name,
                    "status": "passed",
                    "elapsed_seconds": time.monotonic() - started,
                    "error": None,
                }
            )

    artifact_root = args.artifact_root or raw.get("models_save_path")
    if artifact_root:
        root = Path(artifact_root).absolute()
        root.mkdir(parents=True, exist_ok=True)
        save_yaml(
            root / "mode_matrix_summary.yaml",
            {"schema_version": 1, "rounds": args.rounds, "results": results},
        )

    failures = [result for result in results if result["status"] == "failed"]
    if failures:
        failed_modes = ", ".join(result["variant"] for result in failures)
        raise RuntimeError(f"Mode matrix failures: {failed_modes}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
