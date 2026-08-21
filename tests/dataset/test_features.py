import numpy as np
import pytest

from src.dataset.feature_extraction import FeatureExtraction
from src.dataset.feature_utils import FeatureUtils


def test_spectral_contrast_propagates_extraction_failure(monkeypatch):
    def fail_extraction(**kwargs):
        raise RuntimeError("spectral failure")

    monkeypatch.setattr(
        "src.dataset.feature_extraction.librosa.feature.spectral_contrast",
        fail_extraction,
    )

    with pytest.raises(RuntimeError, match="spectral failure"):
        FeatureExtraction.get_spectral_contrast(np.ones(1024), 16000)


def test_spectral_contrast_supports_six_bands_at_8khz():
    waveform = np.random.default_rng(7).normal(size=8000)

    contrast = FeatureExtraction.get_spectral_contrast(
        waveform,
        8000,
        n_bands=6,
    )

    assert contrast.shape[0] == 7
    assert contrast.shape[1] > 0
    assert np.isfinite(contrast).all()


def test_spectral_contrast_does_not_pass_unsupported_fmax(monkeypatch):
    observed = {}

    def extract(**kwargs):
        observed.update(kwargs)
        return np.ones((7, 3))

    monkeypatch.setattr(
        "src.dataset.feature_extraction.librosa.feature.spectral_contrast",
        extract,
    )

    FeatureExtraction.get_spectral_contrast(np.ones(1024), 8000)

    assert "fmax" not in observed
    assert 0 < observed["fmin"] < 125


@pytest.mark.parametrize(
    ("target_sample_rate", "expected_librosa_rate"),
    [(None, None), (8000, 8000)],
)
def test_audio_loading_resamples_only_when_target_is_configured(
    monkeypatch, target_sample_rate, expected_librosa_rate
):
    observed = {}

    def load(path, *, sr, duration):
        observed.update(path=path, sr=sr, duration=duration)
        return np.ones(16), 16000 if sr is None else sr

    monkeypatch.setattr("src.dataset.feature_utils.librosa.load", load)

    FeatureUtils.load_audio(
        "audio.wav",
        sample_rate=44050,
        target_sample_rate=target_sample_rate,
        duration=1.5,
    )

    assert observed == {
        "path": "audio.wav",
        "sr": expected_librosa_rate,
        "duration": 1.5,
    }
