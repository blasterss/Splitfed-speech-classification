"""Split-server model and optimizer ownership operations."""

import torch

from ...model.server_side_model import ServerSideModel
from ...schema import SplitServerConfig
from ...utils.training import build_optimizer, set_seed


def build_personalized_models(
    client_ids: list,
    config: SplitServerConfig,
    device: torch.device,
) -> tuple[dict, dict]:
    """Create independently owned models and optimizers per client."""
    models = {}
    optimizers = {}
    for client_id in client_ids:
        set_seed(config.seed)
        model = ServerSideModel(model_type="cnn_birnn").to(device)
        models[client_id] = model
        optimizers[client_id] = build_optimizer(
            model.parameters(), config.model
        )
        optimizers[client_id].zero_grad()
    return models, optimizers
