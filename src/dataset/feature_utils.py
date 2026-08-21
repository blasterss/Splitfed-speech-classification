import random
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


class FeatureUtils:
    @staticmethod
    def load_audio(
        path: Path,
        sample_rate: float,
        target_sample_rate: float | None,
        duration: float | None = None,
    ):
        """
        Loads an audio file with optional resampling
        and duration trimming.
        """

        return librosa.load(path, sr=target_sample_rate, duration=duration)

    @staticmethod
    def show_waveform(y: np.ndarray, sr: float):
        """
        Displays an audio waveform.
        """

        plt.figure(figsize=(12, 4))

        librosa.display.waveshow(y, sr=sr)

        plt.title("Audio Waveform")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")

        plt.show()

    @staticmethod
    def show_spec(spec, sr: float, title: str = "Spectrogram"):
        """
        Displays a spectrogram.
        """

        fig = plt.figure(figsize=(12, 5))

        img = librosa.display.specshow(
            spec, sr=sr, x_axis="time", y_axis="mel", cmap="magma_r"
        )

        fig.colorbar(img, format="%+2.0f dB")

        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.title(title)

        plt.show()

    @staticmethod
    def show_random_sample(
        dataloader: DataLoader,
        sr: float = 16000,
        channel_names: list[str] | None = None,
    ):
        """
        Displays a random sample from the dataloader.
        """

        target = random.randint(0, len(dataloader) - 1)

        for i, (features, labels) in enumerate(dataloader):
            if i == target:
                idx = random.randint(0, len(features) - 1)

                FeatureUtils._visualize(
                    features[idx], labels[idx].item(), sr, channel_names
                )

                break

    @staticmethod
    def show_sample_by_index(
        dataloader: DataLoader,
        batch_idx: int,
        sample_idx: int,
        sr: float = 16000,
        channel_names: list[str] | None = None,
    ):
        """
        Displays a specific sample from a batch.
        """

        for i, (features, labels) in enumerate(dataloader):
            if i == batch_idx:
                FeatureUtils._visualize(
                    features[sample_idx],
                    labels[sample_idx].item(),
                    sr,
                    channel_names,
                )

                return

        raise ValueError(f"batch_idx {batch_idx} out of range")

    @staticmethod
    def _visualize(
        feature: torch.Tensor,
        label: int,
        sr: float,
        channel_names: list[str] | None = None,
    ):
        """
        Internal visualization helper for features.
        """

        title = f"Label: {label} ({'Conflict' if label == 1 else 'Other'})"

        # Multi-channel format [C, H, W]
        if feature.dim() == 3:
            n_channels = feature.shape[0]

            names = channel_names or [
                f"Channel {i}" for i in range(n_channels)
            ]

            n_cols = min(4, n_channels)
            n_rows = (n_channels + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))

            axes = np.array(axes).flatten()

            for ch, ax in enumerate(axes):
                if ch >= n_channels:
                    ax.set_visible(False)
                    continue

                data = feature[ch].cpu().numpy()

                # Flatten shape (1, T) -> (T,)
                if data.shape[0] == 1:
                    data = data.squeeze(0)

                # Spectrogram visualization
                if data.ndim == 2 and data.shape[0] > 1:
                    librosa.display.specshow(
                        data,
                        sr=sr,
                        x_axis="time",
                        y_axis="mel",
                        cmap="viridis",
                        ax=ax,
                    )

                # 1D signal visualization
                else:
                    ax.plot(data.squeeze())
                    ax.set_xlabel("Time")

                ax.set_title(names[ch], fontsize=10)

            plt.suptitle(title, fontsize=14, fontweight="bold")

        # Stacked format [C, W]
        else:
            fig, ax = plt.subplots(figsize=(15, 5))

            data = feature.cpu().numpy()

            if data.ndim == 2:
                librosa.display.specshow(
                    data,
                    sr=sr,
                    x_axis="time",
                    y_axis="linear",
                    cmap="viridis",
                    ax=ax,
                )

            else:
                ax.plot(data)

            ax.set_title(title)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def show_batch(
        dataloader: DataLoader,
        sr: float,
        channel_names: list[str] | None = None,
    ):
        """
        Displays multiple samples from a batch.
        """

        for features, labels in dataloader:
            n_samples = min(32, len(features))

            n_cols = 8
            n_rows = (n_samples + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))

            axes = np.array(axes).flatten()

            for i in range(len(axes)):
                ax = axes[i]

                if i >= n_samples:
                    ax.set_visible(False)
                    continue

                data = features[i]

                # Average channels for visualization
                to_show = (
                    data.mean(0).cpu().numpy()
                    if data.dim() == 3
                    else data.cpu().numpy()
                )

                if to_show.ndim == 2:
                    librosa.display.specshow(
                        to_show,
                        sr=sr,
                        x_axis="time",
                        y_axis="mel",
                        cmap="viridis",
                        ax=ax,
                    )

                else:
                    ax.plot(to_show)

                ax.set_title(f"L: {labels[i].item()}", fontsize=8)

                ax.set_xticks([])
                ax.set_yticks([])

            plt.tight_layout()
            plt.show()

            break

    @staticmethod
    def dataset_stats(
        data: torch.Tensor,
        labels: torch.Tensor,
        channel_names: list[str] | None = None,
    ):
        """
        Prints dataset statistics.

        Args:
            data:
                Tensor of shape [N, C, 1, T]

            labels:
                Tensor of shape [N]
        """

        n_samples = data.shape[0]
        n_channels = data.shape[1]

        names = channel_names or [f"Channel {i}" for i in range(n_channels)]

        print(f"Samples : {n_samples}")
        print(f"Shape   : {tuple(data.shape[1:])}")

        unique, counts = torch.unique(labels, return_counts=True)
        label_counts = {
            key.item(): value.item()
            for key, value in zip(unique, counts, strict=True)
        }
        print(f"Labels  : {label_counts}\n")

        print(
            f"{'Channel':<20} "
            f"{'Mean':>8} "
            f"{'Std':>8} "
            f"{'Min':>8} "
            f"{'Max':>8} "
            f"{'Zeros%':>8}"
        )

        print("-" * 60)

        for ch in range(n_channels):
            # Flatten channel data
            ch_data = data[:, ch].flatten()

            zero_pct = (ch_data == 0).float().mean().item() * 100

            print(
                f"{names[ch]:<20}"
                f"{ch_data.mean().item():>8.3f}"
                f"{ch_data.std().item():>8.3f}"
                f"{ch_data.min().item():>8.3f}"
                f"{ch_data.max().item():>8.3f}"
                f"{zero_pct:>7.1f}%"
            )
