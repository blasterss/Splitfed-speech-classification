import torch
import torch.nn as nn

from .client_side_model import ClientSideModel
from .server_side_model import ServerSideModel


class SpeechRecognitionModel(nn.Module):
    """
    End-to-end speech emotion recognition model.

    Architecture is split into:
        - Client-side feature extractor (lightweight CNN)
        - Server-side classifier (CNN + GAP or CNN + BiRNN)
    """

    def __init__(
        self,
        input_channels: int,
        server_side_model_type: str,
        noise_std: float = 0.0,
        noise_type: str | None = None,
    ):
        super().__init__()

        # Fixed intermediate representation size
        self.client_output_dim = 64

        # --- Client-side model ---
        self.client_side_model = ClientSideModel(
            input_channels=input_channels,
            output_channels=self.client_output_dim,
            noise=noise_type is not None,
            noise_std=noise_std,
            noise_type=noise_type,
        )

        # --- Server-side model ---
        self.server_side_model = ServerSideModel(
            input_channels=self.client_output_dim,
            output_channels=256,
            model_type=server_side_model_type,
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass:
            raw audio features -> client model -> server model -> logits
        """

        x = self.client_side_model(x)
        x = self.server_side_model(x)

        return x
