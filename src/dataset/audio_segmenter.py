from .audio import load_audio


class AudioFileSegmenter:
    def __init__(
        self,
        config,
        filepath: str,
        window_duration: float = 1,
        hop_duration: float = 0.5,
    ):
        if window_duration <= 0 or hop_duration <= 0:
            raise ValueError(
                "Segment window and hop durations must be positive"
            )

        self.audio, self.sample_rate = load_audio(
            filepath,
            target_sample_rate=None,
        )
        self.window_size = int(window_duration * self.sample_rate)
        self.hop_size = int(hop_duration * self.sample_rate)
        if self.window_size <= 0 or self.hop_size <= 0:
            raise ValueError("Segment window and hop sizes must be positive")
        self.num_samples = len(self.audio)

    def __iter__(self):
        for start in range(
            0, self.num_samples - self.window_size + 1, self.hop_size
        ):
            end = start + self.window_size
            yield self.audio[start:end]

    def get_segments(self):
        return list(iter(self))
