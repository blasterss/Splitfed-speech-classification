import torch

from src.schema import AggregationStrategy
from src.splitfed.fed_server import FedServer


def test_weighted_fedavg_weights_floating_parameters_by_dataset_size():
    client_params = [
        {"weight": torch.tensor([0.0])},
        {"weight": torch.tensor([10.0])},
    ]

    aggregated = FedServer.aggregate(client_params, [1, 3])

    assert torch.equal(aggregated["weight"], torch.tensor([7.5]))


def test_uniform_fedavg_does_not_weight_by_dataset_size():
    client_params = [
        {"weight": torch.tensor([0.0])},
        {"weight": torch.tensor([10.0])},
    ]

    aggregated = FedServer.aggregate(
        client_params, [1, 99], AggregationStrategy.fedavg
    )

    assert torch.equal(aggregated["weight"], torch.tensor([5.0]))


def test_fedavg_preserves_integer_buffer_from_largest_client():
    client_params = [
        {"weight": torch.tensor([1.0]), "counter": torch.tensor(2)},
        {"weight": torch.tensor([5.0]), "counter": torch.tensor(8)},
    ]

    aggregated = FedServer.aggregate(client_params, [1, 3])

    assert torch.equal(aggregated["weight"], torch.tensor([4.0]))
    assert aggregated["counter"].dtype == torch.int64
    assert torch.equal(aggregated["counter"], torch.tensor(8))


def test_fedavg_computes_on_requested_device_and_returns_cpu(monkeypatch):
    transfers = []
    original_to = torch.Tensor.to

    def record_to(tensor, *args, **kwargs):
        if args:
            transfers.append(str(args[0]))
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", record_to)
    client_params = [
        {"weight": torch.tensor([2.0])},
        {"weight": torch.tensor([6.0])},
    ]

    aggregated = FedServer.aggregate(
        client_params, [1, 1], device=torch.device("cpu")
    )

    assert transfers.count("cpu") >= 2
    assert aggregated["weight"].device.type == "cpu"
    assert torch.equal(aggregated["weight"], torch.tensor([4.0]))
