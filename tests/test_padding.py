import numpy as np

from src.dataset.dataset import EmotionalDataset, _masked_normalization_stats
from src.dataset.processors.base_processor import BaseDatasetLoader


def test_pad_stacked_aligns_time_dimension_with_zeros():
    features = [
        np.ones((2, 3), dtype=np.float32),
        np.full((2, 5), 2, dtype=np.float32),
    ]

    padded = BaseDatasetLoader._pad_stacked(features)

    assert [feature.shape for feature in padded] == [(2, 5), (2, 5)]
    np.testing.assert_array_equal(padded[0][:, 3:], 0)
    np.testing.assert_array_equal(padded[1], features[1])


def test_pad_multichannel_aligns_last_dimension_with_zeros():
    features = [
        np.ones((2, 3, 4), dtype=np.float32),
        np.full((2, 3, 6), 2, dtype=np.float32),
    ]

    padded = BaseDatasetLoader._pad_multichannel(features)

    assert [feature.shape for feature in padded] == [(2, 3, 6), (2, 3, 6)]
    np.testing.assert_array_equal(padded[0][..., 4:], 0)
    np.testing.assert_array_equal(padded[1], features[1])


def test_padding_empty_input_returns_empty_list():
    assert BaseDatasetLoader._pad_stacked([]) == []
    assert BaseDatasetLoader._pad_multichannel([]) == []


def test_masked_normalization_ignores_padded_frames():
    data = np.asarray(
        [
            [[1.0, 1.0, 0.0, 0.0]],
            [[3.0, 3.0, 3.0, 3.0]],
        ],
        dtype=np.float32,
    )

    mean, std = _masked_normalization_stats(data, np.asarray([2, 4]))

    np.testing.assert_allclose(mean, np.asarray([14.0 / 6.0]))
    valid_values = np.asarray([1.0, 1.0, 3.0, 3.0, 3.0, 3.0])
    np.testing.assert_allclose(std, np.asarray([valid_values.std() + 1e-8]))


def test_normalized_dataset_keeps_padded_frames_zero():
    dataset = EmotionalDataset(
        np.asarray([[[1.0, 1.0, 0.0, 0.0]]], dtype=np.float32),
        np.asarray([0.0]),
        mean=np.asarray([1.0]),
        std=np.asarray([0.5]),
        valid_frames=np.asarray([2]),
    )

    features, _ = dataset[0]

    np.testing.assert_array_equal(features.numpy()[..., 2:], 0)
