import pytest
import torch

from src.schema import ClientModelConfig
from src.utils.training import build_optimizer


@pytest.mark.parametrize("optimizer_name", ["adam", "sgd"])
def test_build_optimizer_uses_typed_configuration(optimizer_name):
    model = torch.nn.Linear(2, 1)
    config = ClientModelConfig(
        optimizer=optimizer_name,
        lr=0.2,
        momentum=0.5 if optimizer_name == "sgd" else 0.0,
        weight_decay=0.01,
    )

    optimizer = build_optimizer(model.parameters(), config)

    assert optimizer.param_groups[0]["lr"] == 0.2
    assert optimizer.param_groups[0]["weight_decay"] == 0.01
    if optimizer_name == "sgd":
        assert isinstance(optimizer, torch.optim.SGD)
        assert optimizer.param_groups[0]["momentum"] == 0.5
    else:
        assert isinstance(optimizer, torch.optim.Adam)
