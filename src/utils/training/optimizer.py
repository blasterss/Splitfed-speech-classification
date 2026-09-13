"""Typed optimizer construction shared by all training topologies."""

import torch

from ...schema import OptimizerType


def build_optimizer(parameters, config) -> torch.optim.Optimizer:
    """Build the optimizer selected by a validated model config."""
    if config.optimizer is OptimizerType.adam:
        return torch.optim.Adam(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    if config.optimizer is OptimizerType.sgd:
        return torch.optim.SGD(
            parameters,
            lr=config.learning_rate,
            momentum=config.momentum,
            nesterov=config.nesterov,
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {config.optimizer}")
