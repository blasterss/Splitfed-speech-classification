from types import SimpleNamespace

import numpy as np
import pytest

from src.dataset.audio import AudioFileSegmenter


def test_audio_segmenter_unpacks_audio_and_uses_effective_sample_rate(
    monkeypatch,
):
    waveform = np.arange(10)
    observed = {}

    def load_audio(path, *, target_sample_rate):
        observed.update(
            path=path,
            target_sample_rate=target_sample_rate,
        )
        return waveform, 4

    monkeypatch.setattr("src.dataset.audio.segmenter.load_audio", load_audio)

    segmenter = AudioFileSegmenter(
        SimpleNamespace(SAMPLING_RATE=16000),
        "audio.wav",
        window_duration=1,
        hop_duration=0.5,
    )

    assert observed == {
        "path": "audio.wav",
        "target_sample_rate": None,
    }
    assert segmenter.sample_rate == 4
    assert [segment.tolist() for segment in segmenter] == [
        [0, 1, 2, 3],
        [2, 3, 4, 5],
        [4, 5, 6, 7],
        [6, 7, 8, 9],
    ]


@pytest.mark.parametrize("window_duration,hop_duration", [(0, 0.5), (1, 0)])
def test_audio_segmenter_rejects_non_positive_durations(
    window_duration, hop_duration
):
    with pytest.raises(ValueError, match="durations"):
        AudioFileSegmenter(
            SimpleNamespace(SAMPLING_RATE=16000),
            "audio.wav",
            window_duration=window_duration,
            hop_duration=hop_duration,
        )
