from .feature_utils import FeatureUtils


class AudioFileSegmenter:
    def __init__(
        self,
        config,
        filepath: str,
        window_duration: float = 1,
        hop_duration: float = 0.5,
    ):
        self.window_size = int(window_duration * config.SAMPLING_RATE)
        self.hop_size = int(hop_duration * config.SAMPLING_RATE)

        self.audio = FeatureUtils.load_audio(filepath)
        self.num_samples = len(self.audio)

    def __iter__(self):
        for start in range(
            0, self.num_samples - self.window_size + 1, self.hop_size
        ):
            end = start + self.window_size
            yield self.audio[start:end]

    def get_segments(self):
        return list(iter(self))
