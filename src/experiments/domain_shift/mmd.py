"""Unbiased RBF MMD squared with a record-level permutation test."""

import numpy as np

from .data import _validate_matrix


def mmd_permutation_test(
    left: np.ndarray,
    right: np.ndarray,
    *,
    permutations: int,
    seed: int,
) -> dict[str, float]:
    """Return RBF MMD squared, bandwidth and permutation-test resolution."""
    left = _validate_matrix(left, "left")
    right = _validate_matrix(right, "right")
    if left.shape[1] != right.shape[1]:
        raise ValueError("MMD inputs must have identical feature dimensions")
    if len(left) < 2 or len(right) < 2:
        raise ValueError(
            "Unbiased MMD requires at least two samples per group"
        )
    if permutations <= 0:
        raise ValueError("Permutation count must be positive")
    combined = np.concatenate((left, right), axis=0)
    squared_distances = _pairwise_squared_distances(combined)
    bandwidth_squared = _median_positive_distance_squared(squared_distances)
    kernel = np.exp(-squared_distances / (2.0 * bandwidth_squared))
    left_indices = np.arange(len(left))
    right_indices = np.arange(len(left), len(combined))
    observed = _unbiased_mmd_from_kernel(kernel, left_indices, right_indices)

    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(permutations):
        shuffled = rng.permutation(len(combined))
        candidate = _unbiased_mmd_from_kernel(
            kernel,
            shuffled[: len(left)],
            shuffled[len(left) :],
        )
        exceedances += candidate >= observed

    return {
        "mmd_squared": float(observed),
        "rbf_bandwidth": float(np.sqrt(bandwidth_squared)),
        "permutation_p_value": float((exceedances + 1) / (permutations + 1)),
        "permutation_exceedances": int(exceedances),
        "permutation_p_floor": 1 / (permutations + 1),
        "permutation_floor_reached": exceedances == 0,
    }


def _pairwise_squared_distances(values: np.ndarray) -> np.ndarray:
    norms = np.einsum("ij,ij->i", values, values)
    distances = norms[:, None] + norms[None, :] - 2.0 * values @ values.T
    return np.maximum(distances, 0.0)


def _median_positive_distance_squared(
    squared_distances: np.ndarray,
) -> float:
    upper = squared_distances[np.triu_indices(len(squared_distances), k=1)]
    positive = upper[upper > 0]
    if not len(positive):
        return 1.0
    return float(np.median(positive))


def _unbiased_mmd_from_kernel(
    kernel: np.ndarray, left_indices: np.ndarray, right_indices: np.ndarray
) -> float:
    left_kernel = kernel[np.ix_(left_indices, left_indices)]
    right_kernel = kernel[np.ix_(right_indices, right_indices)]
    cross_kernel = kernel[np.ix_(left_indices, right_indices)]
    left_count = len(left_indices)
    right_count = len(right_indices)
    left_term = (left_kernel.sum() - np.trace(left_kernel)) / (
        left_count * (left_count - 1)
    )
    right_term = (right_kernel.sum() - np.trace(right_kernel)) / (
        right_count * (right_count - 1)
    )
    return float(left_term + right_term - 2.0 * cross_kernel.mean())
