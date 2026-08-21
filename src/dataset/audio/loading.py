from pathlib import Path

import librosa


def load_audio(
    path: Path | str,
    *,
    target_sample_rate: float | None,
    duration: float | None = None,
):
    """Load audio at its native rate or resample to the configured rate."""
    return librosa.load(path, sr=target_sample_rate, duration=duration)
