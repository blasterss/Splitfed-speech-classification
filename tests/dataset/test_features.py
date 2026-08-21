import numpy as np
import pytest

from src.dataset.feature_extraction import FeatureExtraction


def test_spectral_contrast_propagates_extraction_failure(monkeypatch):
    def fail_extraction(**kwargs):
        raise RuntimeError("spectral failure")

    monkeypatch.setattr(
        "src.dataset.feature_extraction.librosa.feature.spectral_contrast",
        fail_extraction,
    )

    with pytest.raises(RuntimeError, match="spectral failure"):
        FeatureExtraction.get_spectral_contrast(np.ones(1024), 16000)
