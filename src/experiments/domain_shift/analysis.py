"""E0 record-level domain discrepancy without a training split."""

from __future__ import annotations

import argparse
import itertools
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from ...dataset.processors.factory import DatasetLoaderFactory
from ...schema import DatasetConfig
from ...utils.config import read_yaml, save_yaml
from .data import (
    _dataset_configs_from_mapping,
    _feature_channel_names,
    _masked_record_means,
    _validate_matrix,
)
from .mmd import mmd_permutation_test
from .wasserstein import relative_wasserstein_1

STAGES = ("raw", "client_local_normalized")


def discrepancy(left, right, *, permutations=500, seed=42):
    """Compute unbiased RBF MMD squared in the supplied feature space."""
    return mmd_permutation_test(
        left, right, permutations=permutations, seed=seed
    )


def _write_summary(frame, groups, metrics, path):
    summary = frame.groupby(groups)[metrics].agg(
        [
            "mean",
            "std",
            "median",
            lambda s: s.quantile(0.025),
            lambda s: s.quantile(0.975),
        ]
    )
    summary.columns = [
        f"{metric}_{stat}".replace("<lambda_0>", "q025").replace(
            "<lambda_1>", "q975"
        )
        for metric, stat in summary.columns
    ]
    summary.reset_index().to_csv(path, index=False)


def run_domain_shift(
    dataset_configs: list[DatasetConfig],
    *,
    artifact_root: str | Path,
    sample_size=480,
    repeats=50,
    permutations=500,
    seed=42,
    wasserstein_sample_size=240,
):
    """Load all records and standardize temporal means within each corpus."""
    if (
        len(dataset_configs) != 3
        or len({c.name for c in dataset_configs}) != 3
    ):
        raise ValueError("E0 requires three distinct corpora")
    if sample_size < 2 or repeats < 1 or permutations < 1:
        raise ValueError(
            "Require sample_size >= 2 and positive repeats/permutations"
        )
    if wasserstein_sample_size < 1:
        raise ValueError("wasserstein_sample_size must be positive")
    representations, manifests, names = {}, {}, None
    feature_rows = []
    for config in dataset_configs:
        loader = DatasetLoaderFactory.create(config)
        records, metadata = loader.load()
        if not records or len(records) != len(metadata):
            raise ValueError("Expected aligned non-empty records and metadata")
        data = np.stack(records).astype(np.float64)
        valid = np.asarray(
            [m.get("valid_frames", data.shape[-1]) for m in metadata]
        )
        raw = _validate_matrix(_masked_record_means(data, valid), "features")
        current_names = _feature_channel_names(
            config.feature_names, raw.shape[1]
        )
        if names is not None and names != current_names:
            raise ValueError("Corpora must have identical feature ordering")
        names = current_names
        required = max(sample_size, 2 * wasserstein_sample_size)
        if len(raw) < required:
            raise ValueError(
                f"{config.name.value}: {len(raw)} records < {required}"
            )
        mean, std = raw.mean(0), raw.std(0)
        std = np.where(std > 0, std, 1.0)
        corpus = config.name.value
        representations[corpus] = {"raw": raw, STAGES[1]: (raw - mean) / std}
        labels = np.asarray([m["label"] for m in metadata])
        manifests[corpus] = {
            "records": len(raw),
            "actors": len({m["actor_id"] for m in metadata}),
            "actor_ids": sorted({str(m["actor_id"]) for m in metadata}),
            "anger": int((labels == 1).sum()),
            "non_anger": int((labels == 0).sum()),
            "extraction": loader.last_load_report,
            "normalization_mean": mean.tolist(),
            "normalization_std": std.tolist(),
        }
        for j, feature in enumerate(names):
            values = raw[:, j]
            feature_rows.append(
                {
                    "dataset": corpus,
                    "feature": feature,
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)),
                    "median": float(np.median(values)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
    rows, wasserstein_rows, selections = [], [], []
    pairs = list(itertools.combinations(representations, 2))
    streams = np.random.SeedSequence(seed).spawn(repeats * len(pairs))
    for repeat in range(repeats):
        for pair_index, (left, right) in enumerate(pairs):
            sampling, testing, transport = streams[
                repeat * len(pairs) + pair_index
            ].spawn(3)
            rng = np.random.default_rng(sampling)
            indices = {
                c: rng.choice(
                    len(representations[c]["raw"]), sample_size, replace=False
                )
                for c in (left, right)
            }
            identity = {
                "repeat": repeat,
                "left_corpus": left,
                "right_corpus": right,
            }
            transport_rng = np.random.default_rng(transport)
            halves = {}
            for corpus in (left, right):
                draw = transport_rng.choice(
                    len(representations[corpus]["raw"]),
                    2 * wasserstein_sample_size,
                    replace=False,
                )
                halves[corpus] = {
                    "a": draw[:wasserstein_sample_size].tolist(),
                    "b": draw[wasserstein_sample_size:].tolist(),
                }
            selections.append(
                {
                    **identity,
                    "indices": {c: v.tolist() for c, v in indices.items()},
                    "wasserstein_indices": halves,
                }
            )
            test_seed = int(testing.generate_state(1)[0])
            for stage in STAGES:
                x, y = (
                    representations[c][stage][indices[c]]
                    for c in (left, right)
                )
                rows.append(
                    {
                        **identity,
                        "normalization": stage,
                        "n_each": sample_size,
                        **discrepancy(
                            x, y, permutations=permutations, seed=test_seed
                        ),
                    }
                )
            samples = [
                representations[c][STAGES[1]][halves[c][half]]
                for c in (left, right)
                for half in ("a", "b")
            ]
            wasserstein_rows.append(
                {
                    **identity,
                    "normalization": STAGES[1],
                    "n_each": wasserstein_sample_size,
                    **relative_wasserstein_1(*samples),
                }
            )
    destination = Path(artifact_root)
    destination.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(destination / "pairwise_repeats.csv", index=False)
    _write_summary(
        frame,
        ["normalization", "left_corpus", "right_corpus"],
        ["mmd_squared"],
        destination / "pairwise_summary.csv",
    )
    transport_frame = pd.DataFrame(wasserstein_rows)
    transport_frame.to_csv(
        destination / "wasserstein_repeats.csv", index=False
    )
    _write_summary(
        transport_frame,
        ["normalization", "left_corpus", "right_corpus"],
        [
            "wasserstein_between",
            "wasserstein_within_left",
            "wasserstein_within_right",
            "wasserstein_within_mean",
            "wasserstein_ratio",
        ],
        destination / "wasserstein_summary.csv",
    )
    pd.DataFrame(
        [
            {
                "dataset": c,
                **{
                    key: m[key]
                    for key in ("records", "actors", "anger", "non_anger")
                },
                "anger_share": m["anger"] / m["records"],
            }
            for c, m in manifests.items()
        ]
    ).to_csv(destination / "corpus_summary.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(
        destination / "feature_summary.csv", index=False
    )
    resolved = {
        "schema_version": 2,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: version(name)
                for name in ("numpy", "POT", "scipy", "pandas")
            },
        },
        "sample_size": sample_size,
        "repeats": repeats,
        "permutations": permutations,
        "seed": seed,
        "wasserstein_sample_size": wasserstein_sample_size,
        "wasserstein_protocol": "disjoint_halves_without_replacement",
        "wasserstein_ratio": "between / mean(within_left, within_right)",
        "dataset_configs": [
            c.model_dump(mode="json", exclude={"test_size", "split_seed"})
            for c in dataset_configs
        ],
        "representation": "valid_frame_temporal_means",
        "normalization": (
            "population_standardization_of_all_record_means_per_corpus"
        ),
        "feature_names": names,
        "manifests": manifests,
        "selections": selections,
        "limitations": [
            "Record sampling and permutations ignore actor dependence; "
            "p-values are descriptive.",
            "Repeat quantiles describe subsampling variation, "
            "not population confidence intervals.",
            "SAVEE MMD at n=480 uses all records on every repeat; "
            "only its ordering varies.",
            "Raw and normalized geometries differ; "
            "do not subtract or report percentage reduction.",
            "R is a finite-sample empirical relative discrepancy, not "
            "a population-normalized distance or causal effect.",
            "A zero within baseline leaves R undefined (NaN).",
            "Within estimates use disjoint records, not disjoint actors.",
        ],
    }
    save_yaml(destination / "resolved_analysis.yaml", resolved)
    return resolved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--sample-size", type=int, default=480)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wasserstein-sample-size", type=int, default=240)
    args = parser.parse_args()
    run_domain_shift(
        _dataset_configs_from_mapping(read_yaml(args.config_file)),
        artifact_root=args.artifact_root,
        sample_size=args.sample_size,
        repeats=args.repeats,
        permutations=args.permutations,
        seed=args.seed,
        wasserstein_sample_size=args.wasserstein_sample_size,
    )


if __name__ == "__main__":
    main()
