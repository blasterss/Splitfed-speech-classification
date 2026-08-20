"""Split-server model and optimizer ownership operations."""

import torch
import torch.optim as optim

from ...model.server_side_model import ServerSideModel
from ...schema import SplitServerConfig
from ...utils.training import set_seed


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
        optimizers[client_id] = optim.Adam(
            model.parameters(), lr=config.model.learning_rate
        )
        optimizers[client_id].zero_grad()
    return models, optimizers


def step_accumulated_gradients(
    parameters,
    optimizer: optim.Optimizer,
    batch_count: int,
) -> None:
    """Apply one optimizer step after averaging accumulated gradients."""
    if batch_count <= 0:
        raise ValueError("batch_count must be positive")
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.div_(batch_count)
    optimizer.step()
    optimizer.zero_grad()
