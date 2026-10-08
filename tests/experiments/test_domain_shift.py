import numpy as np
import pandas as pd
import pytest

from src.experiments.domain_shift import discrepancy, run_domain_shift
from src.schema import DatasetConfig


def test_reference_transport_and_permutation_floor():
    x = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    result = discrepancy(x, x + [3.0, 4.0], permutations=9)
    from src.experiments.domain_shift.wasserstein import (
        multivariate_wasserstein_1,
    )

    assert multivariate_wasserstein_1(x, x + [3.0, 4.0]) == pytest.approx(5)
    assert result["permutation_p_floor"] == 0.1
    assert result["permutation_p_value"] >= 0.1


def test_all_records_balanced_repeats_and_local_normalization(
    tmp_path, monkeypatch
):
    class Loader:
        last_load_report = {"loaded": 6, "failed": 0}

        def __init__(self, config):
            self.offset = {"CREMA-D": 0, "RAVDESS": 10, "SAVEE": 20}[
                config.name.value
            ]

        def load(self):
            return (
                [
                    np.array([[i + self.offset, i + self.offset + 2, 999.0]])
                    for i in range(6)
                ],
                [
                    {
                        "label": i % 2,
                        "actor_id": "single_actor",
                        "valid_frames": 2,
                    }
                    for i in range(6)
                ],
            )

    monkeypatch.setattr(
        "src.experiments.domain_shift.analysis.DatasetLoaderFactory.create",
        Loader,
    )
    configs = [
        DatasetConfig(name=c, root=str(tmp_path), feature_names=["zcr"])
        for c in ("CREMA-D", "RAVDESS", "SAVEE")
    ]
    result = run_domain_shift(
        configs,
        artifact_root=tmp_path / "out",
        sample_size=6,
        wasserstein_sample_size=3,
        repeats=2,
        permutations=3,
    )
    rows = pd.read_csv(tmp_path / "out/pairwise_repeats.csv")
    assert len(rows) == 12
    assert set(rows.n_each) == {6}
    assert len(result["selections"]) == 6
    assert result["manifests"]["CREMA-D"]["normalization_mean"] == [3.5]
    transport = pd.read_csv(tmp_path / "out/wasserstein_repeats.csv")
    assert len(transport) == 6
    assert set(transport.n_each) == {3}
    assert set(transport.normalization) == {"client_local_normalized"}
    assert "sinkhorn_divergence" not in rows
    assert len(pd.read_csv(tmp_path / "out/corpus_summary.csv")) == 3
    assert len(pd.read_csv(tmp_path / "out/feature_summary.csv")) == 3
    assert result["manifests"]["CREMA-D"]["actors"] == 1
    with pytest.raises(ValueError, match="records <"):
        run_domain_shift(
            configs, artifact_root=tmp_path / "bad", sample_size=7, repeats=1
        )


def test_wasserstein_ratio_reference_and_zero_baseline():
    from src.experiments.domain_shift.wasserstein import relative_wasserstein_1

    arrays = [np.array([[x], [x + 1.0]]) for x in (0, 2, 10, 14)]
    result = relative_wasserstein_1(*arrays)
    assert result["wasserstein_between"] == pytest.approx(10)
    assert result["wasserstein_within_left"] == pytest.approx(2)
    assert result["wasserstein_within_right"] == pytest.approx(4)
    assert result["wasserstein_ratio"] == pytest.approx(10 / 3)
    zero = relative_wasserstein_1(arrays[0], arrays[0], arrays[2], arrays[2])
    assert not zero["ratio_defined"]
    assert np.isnan(zero["wasserstein_ratio"])


def test_mmd_reference_statistic_and_500_permutation_floor():
    from src.experiments.domain_shift.mmd import mmd_permutation_test

    left = np.arange(20, dtype=float).reshape(-1, 1)
    right = left + 100
    values = np.vstack((left, right))
    distances = (values[:, None, :] - values[None, :, :]) ** 2
    distances = distances.sum(axis=2)
    positive = distances[np.triu_indices(len(values), 1)]
    sigma_squared = np.median(positive[positive > 0])
    kernel = np.exp(-distances / (2 * sigma_squared))
    n = len(left)
    expected = (
        (kernel[:n, :n].sum() - np.trace(kernel[:n, :n])) / (n * (n - 1))
        + (kernel[n:, n:].sum() - np.trace(kernel[n:, n:])) / (n * (n - 1))
        - 2 * kernel[:n, n:].mean()
    )
    result = mmd_permutation_test(left, right, permutations=500, seed=42)
    assert result["mmd_squared"] == pytest.approx(expected)
    assert result["rbf_bandwidth"] == pytest.approx(np.sqrt(sigma_squared))
    assert result["permutation_exceedances"] == 0
    assert result["permutation_floor_reached"] is True
    assert result["permutation_p_value"] == pytest.approx(1 / 501)


@pytest.mark.parametrize(
    "left,right",
    [
        (np.ones((1, 2)), np.ones((2, 2))),
        (np.ones((2, 2)), np.ones((2, 3))),
        (np.array([[np.nan, 0], [0, 0]]), np.ones((2, 2))),
    ],
)
def test_mmd_rejects_invalid_inputs(left, right):
    from src.experiments.domain_shift.mmd import mmd_permutation_test

    with pytest.raises(ValueError):
        mmd_permutation_test(left, right, permutations=10, seed=42)


def test_sampling_is_replayable_independent_and_paired_between_spaces(
    tmp_path, monkeypatch
):
    class Loader:
        last_load_report = {"loaded": 12, "failed": 0}

        def __init__(self, config):
            pass

        def load(self):
            return (
                [np.array([[float(i)]]) for i in range(12)],
                [
                    {"label": i % 2, "actor_id": "one", "valid_frames": 1}
                    for i in range(12)
                ],
            )

    comparisons = []

    def capture(left, right, **kwargs):
        comparisons.append((left.copy(), right.copy()))
        return {name: 0.0 for name in ("mmd_squared",)}

    monkeypatch.setattr(
        "src.experiments.domain_shift.analysis.DatasetLoaderFactory.create",
        Loader,
    )
    monkeypatch.setattr(
        "src.experiments.domain_shift.analysis.discrepancy", capture
    )
    configs = [
        DatasetConfig(name=c, root=str(tmp_path), feature_names=["zcr"])
        for c in ("CREMA-D", "RAVDESS", "SAVEE")
    ]
    first = run_domain_shift(
        configs,
        artifact_root=tmp_path / "first",
        sample_size=3,
        wasserstein_sample_size=3,
        repeats=4,
    )
    replay = run_domain_shift(
        configs,
        artifact_root=tmp_path / "replay",
        sample_size=3,
        wasserstein_sample_size=3,
        repeats=4,
    )
    assert first["selections"] == replay["selections"]
    subsets = {
        tuple(sorted(s["indices"]["CREMA-D"]))
        for s in first["selections"]
        if "CREMA-D" in s["indices"]
    }
    assert len(subsets) > 1
    for raw, normalized in zip(
        comparisons[::2], comparisons[1::2], strict=True
    ):
        for x, z in zip(raw, normalized, strict=True):
            np.testing.assert_allclose(z, (x - 5.5) / np.arange(12).std())
    for selection in first["selections"]:
        for halves in selection["wasserstein_indices"].values():
            assert len(halves["a"]) == len(halves["b"]) == 3
            assert set(halves["a"]).isdisjoint(halves["b"])
    transport = pd.read_csv(tmp_path / "first/wasserstein_repeats.csv")
    np.testing.assert_allclose(
        transport.wasserstein_ratio,
        transport.wasserstein_between / transport.wasserstein_within_mean,
    )
