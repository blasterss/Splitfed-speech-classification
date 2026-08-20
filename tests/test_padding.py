import numpy as np

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
