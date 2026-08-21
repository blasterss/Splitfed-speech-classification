from typing import Literal

import librosa
import numpy as np

from ..schema import FeatureType


class FeatureExtraction:
    """
    Audio feature extraction utility with support for
    stacked and multi-channel feature formats.

    Output formats:
        - 'stacked':
            [n_features, time]
            All features concatenated together.

        - 'multi_channel':
            [n_channels, freq_bins, time]
            Each feature stored in a separate channel.
    """

    # Default extraction configuration
    DEFAULT_CONFIG = {
        "n_mels": 60,
        "n_mfcc": 13,
        "n_fft": 512,
        "hop_length": 128,
        "power": 3,
        "frame_length": 512,
    }

    @staticmethod
    def get_all_features(
        y: np.ndarray,
        sr: int,
        feature_mode: Literal["stacked", "multi_channel"] = "stacked",
        feature_names: list[FeatureType] | None = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Extracts multiple audio features and returns them
        in either stacked or multi-channel format.

        Args:
            y:
                Audio waveform.

            sr:
                Sampling rate.

            feature_mode:
                Output format type.

            feature_names:
                List of feature names to extract.

        Returns:
            Extracted features.
        """

        config = {**FeatureExtraction.DEFAULT_CONFIG, **kwargs}
        feature_names = feature_names or [FeatureType.mel, FeatureType.mfcc]

        channels = []

        if "mel" in feature_names:
            channels.append(
                FeatureExtraction.get_mel_spec(
                    y,
                    sr,
                    n_mels=config["n_mels"],
                    n_fft=config["n_fft"],
                    hop_length=config["hop_length"],
                    power=config["power"],
                )
            )

        if "mfcc" in feature_names:
            channels.append(
                FeatureExtraction.get_mfcc(
                    y,
                    sr,
                    n_mfcc=config["n_mfcc"],
                    n_fft=config["n_fft"],
                    hop_length=config["hop_length"],
                )
            )

        if "rms" in feature_names:
            channels.append(
                FeatureExtraction.get_rms(
                    y,
                    frame_length=config["frame_length"],
                    hop_length=config["hop_length"],
                )
            )

        if "contrast" in feature_names:
            channels.append(
                FeatureExtraction.get_spectral_contrast(
                    y,
                    sr,
                    n_fft=config["n_fft"],
                    hop_length=config["hop_length"],
                )
            )

        if "zcr" in feature_names:
            channels.append(
                FeatureExtraction.get_zero_crossing_rate(
                    y,
                    frame_length=config["frame_length"],
                    hop_length=config["hop_length"],
                )
            )

        # Align all channels by minimum time dimension
        min_time = min(ch.shape[-1] for ch in channels)

        channels = [ch[..., :min_time] for ch in channels]

        if feature_mode == "stacked":
            return np.concatenate(channels, axis=0)

        else:
            return channels

    @staticmethod
    def get_all_features_separate(
        y: np.ndarray, sr: int, **kwargs
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns extracted features separately.

        Returns:
            Tuple of:
                (mel_spec, mfcc, rms)
        """

        config = {**FeatureExtraction.DEFAULT_CONFIG, **kwargs}

        mel_spec = FeatureExtraction.get_mel_spec(
            y,
            sr,
            n_mels=config["n_mels"],
            n_fft=config["n_fft"],
            hop_length=config["hop_length"],
            power=config["power"],
        )

        mfcc = FeatureExtraction.get_mfcc(
            y,
            sr,
            n_mfcc=config["n_mfcc"],
            n_fft=config["n_fft"],
            hop_length=config["hop_length"],
        )

        rms = FeatureExtraction.get_rms(
            y,
            frame_length=config["frame_length"],
            hop_length=config["hop_length"],
        )

        # Align time dimensions
        min_time = min(mel_spec.shape[1], mfcc.shape[1], rms.shape[1])

        mel_spec = mel_spec[:, :min_time]
        mfcc = mfcc[:, :min_time]
        rms = rms[:, :min_time]

        return mel_spec, mfcc, rms

    @staticmethod
    def get_mel_spec(
        y: np.ndarray,
        sr: float,
        n_mels: int = 60,
        n_fft: int = 512,
        hop_length: int = 128,
        power: float = 3,
    ) -> np.ndarray:
        """
        Extracts Mel spectrogram features.
        """

        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            power=power,
        )

        # Convert to decibel scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        return mel_spec_db

    @staticmethod
    def get_mfcc(
        y: np.ndarray,
        sr: float,
        n_mfcc: int = 13,
        n_fft: int = 512,
        hop_length: int = 128,
    ) -> np.ndarray:
        """
        Extracts MFCC coefficients.
        """

        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
        )

        return mfcc

    @staticmethod
    def get_rms(
        y: np.ndarray, frame_length: int = 512, hop_length: int = 128
    ) -> np.ndarray:
        """
        Extracts RMS energy features.
        """

        rms = librosa.feature.rms(
            y=y, frame_length=frame_length, hop_length=hop_length
        )

        return rms

    @staticmethod
    def get_spectral_contrast(
        y: np.ndarray,
        sr: float,
        n_fft: int = 512,
        hop_length: int = 128,
        n_bands: int = 6,
        fmin: float = None,
        fmax: float = None,
    ) -> np.ndarray:
        """
        Extracts Spectral Contrast features
        with automatic frequency adjustment.

        Args:
            y:
                Audio waveform.

            sr:
                Sampling rate.

            n_fft:
                FFT window size.

            hop_length:
                Hop size between frames.

            n_bands:
                Number of frequency bands.

            fmin:
                Minimum frequency.
                Defaults to 200 Hz.

            fmax:
                Maximum frequency.
                Defaults to sr/2 - 10.
        """

        # Nyquist frequency
        nyquist = sr / 2

        if fmin is None:
            fmin = 200.0

        if fmax is None:
            # Keep margin below Nyquist
            fmax = min(nyquist - 10, 8000.0)

        # Validate frequency range
        if fmax > nyquist:
            fmax = nyquist - 10

        if fmin >= fmax:
            fmin = 100.0
            fmax = nyquist - 10

        # librosa derives octave bands from fmin and n_bands and does not
        # accept an fmax argument. Lower fmin when needed so the requested
        # number of bands remains below the effective upper frequency.
        max_safe_fmin = fmax / (2 ** (n_bands - 1))
        fmin = min(fmin, max_safe_fmin * 0.99)
        if fmin <= 0:
            raise ValueError("Spectral contrast requires a positive frequency")

        return librosa.feature.spectral_contrast(
            y=y,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_bands=n_bands,
            fmin=fmin,
        )

    @staticmethod
    def get_zero_crossing_rate(
        y: np.ndarray, frame_length: int = 512, hop_length: int = 128
    ) -> np.ndarray:
        """
        Extracts Zero Crossing Rate (ZCR).
        """

        zcr = librosa.feature.zero_crossing_rate(
            y=y, frame_length=frame_length, hop_length=hop_length
        )

        return zcr

    @staticmethod
    def get_pitch(
        y: np.ndarray,
        sr: float,
        hop_length: int = 128,
        fmin: float = 65,
        fmax: float = 350,
    ) -> np.ndarray:
        """
        Extracts pitch (fundamental frequency).
        """

        pitches, magnitudes = librosa.piptrack(
            y=y, sr=sr, hop_length=hop_length, fmin=fmin, fmax=fmax
        )

        # Select pitch with highest magnitude
        # for each frame
        pitch_values = []

        for i in range(pitches.shape[1]):
            frame_pitches = pitches[:, i]
            frame_mags = magnitudes[:, i]

            if frame_mags.max() > 0:
                pitch = frame_pitches[np.argmax(frame_mags)]
            else:
                pitch = 0

            pitch_values.append(pitch)

        return np.array(pitch_values).reshape(1, -1)
