import torch
from torch import nn
from torch.distributions import Laplace, Normal

from .base_model import Baseblock1d


class PrivacyLayer(nn.Module):
    """
    Privacy-preserving layer that optionally applies:
        - L2 norm clipping
        - Additive Gaussian or Laplace noise
    """

    def __init__(
        self,
        noise_std: float = 0.05,
        noise_type: str = "gauss",
        clip_norm: bool = False,
        clip_value: float = 1.0,
    ):
        super().__init__()

        self.noise_std = noise_std
        self.noise_type = noise_type.lower()

        self.clip_norm = clip_norm
        self.clip_value = clip_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies clipping and noise injection.
        """

        if not self.training:
            return x

        # Optional L2 norm clipping
        if self.clip_norm:
            norm = x.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-6)

            x = x * torch.clamp(self.clip_value / norm, max=1.0)

        # Gaussian noise
        if self.noise_type == "gauss":
            distribution = Normal(
                loc=torch.zeros_like(x),
                scale=torch.full_like(x, self.noise_std),
            )

        # Laplace noise
        elif self.noise_type == "laplace":
            distribution = Laplace(
                loc=torch.zeros_like(x),
                scale=torch.full_like(x, self.noise_std),
            )

        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")

        return x + distribution.sample()


class ClientSideModel(nn.Module):
    """
    Client-side neural network model.

    Architecture:
        - Two residual 1D convolution blocks
        - Adaptive average pooling
        - Optional privacy-preserving layer
    """

    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 64,
        kernel_size: int = 7,
        padding: int = 3,
        noise: bool = True,
        noise_std: float | None = 0.05,
        noise_type: str | None = "gauss",
    ):
        super().__init__()

        # First residual block
        self.block1 = Baseblock1d(
            input_channels=input_channels,
            output_channels=output_channels // 2,
            kernel_size=kernel_size,
            padding=padding,
        )

        # Second residual block
        self.block2 = Baseblock1d(
            input_channels=output_channels // 2,
            output_channels=output_channels,
            kernel_size=kernel_size - 2,
            padding=padding - 1,
        )

        # Temporal pooling
        self.pool = nn.AdaptiveAvgPool1d(128)

        # Optional privacy layer
        if noise:
            self.privacy = PrivacyLayer(
                noise_std=noise_std, noise_type=noise_type
            )

        else:
            self.privacy = None

    def forward(self, x):
        """
        Forward pass through client-side model.
        """

        x = self.block1(x)
        x = self.block2(x)

        x = self.pool(x)

        # Apply privacy mechanism
        if self.privacy:
            x = self.privacy(x)

        return x
