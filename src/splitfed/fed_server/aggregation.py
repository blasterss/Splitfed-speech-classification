"""Federated model and metric aggregation policies."""

import copy

import torch

from ...schema import AggregationStrategy


def aggregate_states(
    client_params_list: list[dict],
    client_sizes: list[int],
    strategy: AggregationStrategy = AggregationStrategy.weighted_fedavg,
    device: torch.device | str = "cpu",
) -> dict:
    """Aggregate validated client state dictionaries."""
    if not client_params_list or len(client_params_list) != len(client_sizes):
        raise ValueError(
            "FedAvg requires matching non-empty parameter and size lists"
        )
    if any(size <= 0 for size in client_sizes):
        raise ValueError("FedAvg client sample counts must be positive")

    if strategy is AggregationStrategy.fedavg:
        weights = [1.0 / len(client_params_list)] * len(client_params_list)
    elif strategy in (
        AggregationStrategy.weighted_fedavg,
        AggregationStrategy.mergesfl_batch_weighted_v1,
    ):
        total_samples = sum(client_sizes)
        weights = [size / total_samples for size in client_sizes]
    else:
        raise ValueError(f"Unsupported aggregation strategy: {strategy}")

    aggregation_device = torch.device(device)
    new_params = copy.deepcopy(client_params_list[0])
    largest_client = max(
        range(len(client_sizes)), key=client_sizes.__getitem__
    )

    for key in new_params:
        value = client_params_list[0][key]
        if torch.is_tensor(value) and (
            torch.is_floating_point(value) or torch.is_complex(value)
        ):
            new_params[key] = sum(
                client_params_list[index][key].to(aggregation_device)
                * weights[index]
                for index in range(len(client_params_list))
            ).cpu()
        else:
            new_params[key] = client_params_list[largest_client][key]

    return new_params


def aggregate_metrics(
    eval_metrics: list[tuple[int, dict[str, float]]],
) -> dict[str, float]:
    """Compute sample-weighted evaluation metrics across clients."""
    total_num = sum(num for num, _ in eval_metrics)
    if total_num == 0:
        return {}

    all_keys = {key for _, metrics in eval_metrics for key in metrics}
    return {
        key: sum(
            metrics[key] * num
            for num, metrics in eval_metrics
            if key in metrics
        )
        / total_num
        for key in all_keys
    }
