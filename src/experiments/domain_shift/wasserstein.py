"""Exact multivariate Wasserstein-1 with uniform record masses."""

import numpy as np
import ot


def multivariate_wasserstein_1(left, right):
    cost = ot.dist(left, right, metric="euclidean")
    return float(
        ot.emd2(
            np.full(len(left), 1 / len(left)),
            np.full(len(right), 1 / len(right)),
            cost,
        )
    )


def relative_wasserstein_1(left_a, left_b, right_a, right_b):
    """Compare between-corpus W1 against disjoint within-corpus baselines.

    The first half of each corpus defines the between comparison. All four
    empirical distributions must have the same size and feature ordering.
    A zero baseline leaves R undefined rather than silently adding epsilon.
    """
    from .data import _validate_matrix

    arrays = [
        _validate_matrix(values, name)
        for values, name in zip(
            (left_a, left_b, right_a, right_b),
            ("left_a", "left_b", "right_a", "right_b"),
            strict=True,
        )
    ]
    if any(values.shape != arrays[0].shape for values in arrays):
        raise ValueError("W1 between/within requires equal shapes")
    between = multivariate_wasserstein_1(arrays[0], arrays[2])
    within_left = multivariate_wasserstein_1(arrays[0], arrays[1])
    within_right = multivariate_wasserstein_1(arrays[2], arrays[3])
    baseline = (within_left + within_right) / 2
    return {
        "wasserstein_between": between,
        "wasserstein_within_left": within_left,
        "wasserstein_within_right": within_right,
        "wasserstein_within_mean": baseline,
        "wasserstein_ratio": between / baseline if baseline > 0 else np.nan,
        "ratio_defined": baseline > 0,
    }
