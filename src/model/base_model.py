from torch import nn
import torch.nn.functional as F


class Baseblock1d(nn.Module):
    """
    Residual 1D convolutional block.

    Consists of:
        - Identity shortcut branch
        - Two convolutional layers
        - Residual connection with ReLU activation
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        padding: int,
        stride: int = 1,
    ):
        super().__init__()

        # Identity shortcut branch
        self.identity = nn.Sequential(
            nn.Conv1d(
                input_channels, output_channels, kernel_size=1, stride=stride
            ),
            nn.BatchNorm1d(output_channels),
        )

        # First convolution block
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                input_channels,
                output_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
            ),
            nn.BatchNorm1d(output_channels),
            nn.ReLU(inplace=True),
        )

        # Second convolution block
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                output_channels,
                output_channels,
                kernel_size=kernel_size - 2,
                padding=padding - 1,
                stride=stride,
            ),
            nn.BatchNorm1d(output_channels),
        )

    def forward(self, x):
        """
        Forward pass through residual block.
        """

        identity = self.identity(x)

        x = self.conv1(x)
        x = self.conv2(x)

        return F.relu(x + identity)


class Baseblock(nn.Module):
    """
    Residual 2D convolutional block.

    Consists of:
        - Identity shortcut branch
        - Two convolutional layers
        - Residual connection with ReLU activation
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        padding: int,
        stride: int = 1,
    ):
        super().__init__()

        # Identity shortcut branch
        self.identity = nn.Sequential(
            nn.Conv2d(
                input_channels, output_channels, kernel_size=1, stride=stride
            ),
            nn.BatchNorm2d(output_channels),
        )

        # First convolution block
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

        # Second convolution block
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size=kernel_size - 2,
                padding=padding - 1,
                stride=stride,
            ),
            nn.BatchNorm2d(output_channels),
        )

    def forward(self, x):
        """
        Forward pass through residual block.
        """

        identity = self.identity(x)

        x = self.conv1(x)
        x = self.conv2(x)

        return F.relu(x + identity)
