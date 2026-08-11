import torch
from torch import nn
from typing import Optional, Literal

from ..logger import logger
from .base_model import Baseblock, Baseblock1d


class ServerSideModel(nn.Module):
    """
    Server-side model for binary emotion classification.

    Supports two architecture types:
        - 'cnn_gap':
            CNN + Global Average Pooling
            (simpler, faster, less prone to overfitting)

        - 'cnn_birnn':
            CNN + BiRNN
            (captures long-range temporal dependencies)
    """

    def __init__(
        self,
        input_channels: int = 64,
        output_channels: int = 256,
        model_type: Literal["cnn_gap", "cnn_birnn"] = "cnn_gap",
        # RNN parameters (used only for cnn_birnn)
        rnn_hidden: int = 128,
        rnn_layers: int = 3,
        rnn_dropout: float = 0.2,
        rnn_type: str = "LSTM",
        # Dropout parameters
        cnn_dropout: float = 0.2,
        fc_dropout: float = 0.3,
    ):
        super().__init__()

        self.model_type = model_type
        self.input_channels = input_channels
        self.output_channels = output_channels

        # --- Convolutional feature extractor ---
        self.conv_block = nn.Sequential(
            Baseblock1d(
                input_channels,
                output_channels // 2,
                kernel_size=5,
                padding=2,
                stride=1,
            ),
            nn.Dropout(cnn_dropout) if cnn_dropout > 0 else nn.Identity(),
            Baseblock1d(
                output_channels // 2,
                output_channels,
                kernel_size=5,
                padding=2,
                stride=1,
            ),
            nn.Dropout(cnn_dropout) if cnn_dropout > 0 else nn.Identity(),
        )

        if model_type == "cnn_gap":
            # Global average pooling over time axis
            self.temporal_pool = nn.AdaptiveAvgPool1d(1)

            # Classifier head
            self.classifier = nn.Sequential(
                nn.Linear(output_channels, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(fc_dropout),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(fc_dropout / 2),
                nn.Linear(64, 1),
            )

            self.rnn1 = None
            self.norm = None

        else:
            # --- BiRNN architecture ---
            rnn_class = nn.GRU if rnn_type.lower() == "gru" else nn.LSTM

            self.rnn1 = rnn_class(
                input_size=output_channels,
                hidden_size=rnn_hidden,
                num_layers=rnn_layers,
                batch_first=True,
                bidirectional=True,
                dropout=rnn_dropout if rnn_layers > 1 else 0.0,
            )

            self.norm = nn.LayerNorm(rnn_hidden * 2)

            self.classifier = nn.Sequential(
                nn.Linear(rnn_hidden * 2, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(fc_dropout),
                nn.Linear(128, 1),
            )

        self._initialize_weights()

        logger.info(
            f"ServerSideModel initialized: "
            f"type={model_type}, "
            f"channels={input_channels}->{output_channels}"
        )

    def _initialize_weights(self):
        """
        Initializes model weights for stable training.
        """

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Expected input shape:
            [B, C, T]
        """

        if x.dim() != 3:
            raise ValueError(f"Expected [B, C, T], got {x.shape}")

        x = self.conv_block(x)

        if self.model_type == "cnn_gap":
            x = self.temporal_pool(x)
            x = x.squeeze(-1)

        else:
            x = x.permute(0, 2, 1)
            x, _ = self.rnn1(x)
            x = self.norm(x)
            x = torch.mean(x, dim=1)

        logits = self.classifier(x)
        return logits

    def get_feature_dim(self) -> int:
        """
        Returns feature dimension before classifier.
        """

        if self.model_type == "cnn_gap":
            return self.output_channels

        return self.rnn1.hidden_size * 2

    def set_dropout(self, dropout_rate: float):
        """
        Dynamically updates dropout rates.
        """

        for module in self.classifier:
            if isinstance(module, nn.Dropout):
                module.p = dropout_rate

        if self.rnn1 is not None and hasattr(self.rnn1, "dropout"):
            self.rnn1.dropout = dropout_rate
