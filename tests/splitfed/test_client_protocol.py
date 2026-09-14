import time
from types import SimpleNamespace

import pytest
import torch

from src.schema import TrainingMode, WorkloadPolicy
from src.splitfed.client import (
    Client,
    _cpu_state_dict_snapshot,
    _extract_payload,
    _validate_global_update,
    _validate_round_ack,
)
from src.transport.message import Message

FUTURE_DEADLINE = time.time() + 3600


def _response(**overrides):
    values = {
        "type": "gradients",
        "sender": "split_server",
        "round": 2,
        "step": 3,
        "deadline_at": FUTURE_DEADLINE,
        "payload": {"gradients": torch.ones(1)},
    }
    values.update(overrides)
    return Message(**values)


def test_planned_batch_scales_client_learning_rate():
    client = Client.__new__(Client)
    client.client_id = 0
    client.device = torch.device("cpu")
    client.cfg = SimpleNamespace(
        runtime=SimpleNamespace(seed=42, drop_last=True)
    )
    client.dataset = SimpleNamespace(
        train_dataset=[(torch.ones(1), torch.tensor(0.0))] * 4
    )
    parameter = torch.nn.Parameter(torch.ones(1))
    client.optimizer = torch.optim.SGD([parameter], lr=0.2)
    client._base_learning_rates = (0.2,)

    client.configure_round(
        round_idx=1,
        batch_size=2,
        local_steps=1,
        learning_rate_scale=0.5,
    )

    assert client.optimizer.param_groups[0]["lr"] == 0.1


def test_extract_payload_accepts_correlated_split_response():
    response = _response(request_id="split-request")
    tensor = _extract_payload(
        response,
        "gradients",
        "gradients",
        "client-0",
        2,
        3,
        "split-request",
    )

    assert torch.equal(tensor, torch.ones(1))


@pytest.mark.parametrize(
    "overrides",
    [
        {"sender": "client-1"},
        {"type": "logits"},
        {"round": 1},
        {"step": 4},
    ],
)
def test_extract_payload_rejects_uncorrelated_response(overrides):
    response = _response(request_id="split-request", **overrides)
    assert (
        _extract_payload(
            response,
            "gradients",
            "gradients",
            "client-0",
            2,
            3,
            "split-request",
        )
        is None
    )


def test_extract_payload_rejects_mismatched_request_id():
    assert (
        _extract_payload(
            _response(request_id="other-request"),
            "gradients",
            "gradients",
            "client-0",
            2,
            3,
            "expected-request",
        )
        is None
    )


def test_extract_payload_rejects_expired_response():
    assert (
        _extract_payload(
            _response(
                request_id="split-request",
                deadline_at=time.time() - 1,
            ),
            "gradients",
            "gradients",
            "client-0",
            2,
            3,
            "split-request",
        )
        is None
    )


def _global_response(state_dict, **overrides):
    values = {
        "type": "global_update",
        "sender": "fed_server",
        "round": 2,
        "step": 1,
        "payload": state_dict,
        "request_id": "fed-request",
        "deadline_at": FUTURE_DEADLINE,
    }
    values.update(overrides)
    return Message(**values)


def test_global_update_accepts_matching_state_schema():
    expected = {"weight": torch.ones(2, dtype=torch.float32)}
    received = {"weight": torch.zeros(2, dtype=torch.float32)}

    assert (
        _validate_global_update(
            _global_response(received, request_id="fed-request"),
            expected,
            2,
            "fed-request",
        )
        is received
    )


@pytest.mark.parametrize(
    "response,match",
    [
        (_global_response({}, sender="client-1"), "sender"),
        (_global_response({}, round=1), "round"),
        (_global_response({"other": torch.ones(2)}), "keys"),
        (_global_response({"weight": torch.ones(3)}), "shape"),
        (
            _global_response({"weight": torch.ones(2, dtype=torch.float64)}),
            "dtype",
        ),
    ],
)
def test_global_update_rejects_invalid_envelope_or_state(response, match):
    expected = {"weight": torch.ones(2, dtype=torch.float32)}

    with pytest.raises(ValueError, match=match):
        _validate_global_update(response, expected, 2, "fed-request")


def test_global_update_rejects_mismatched_request_id():
    expected = {"weight": torch.ones(2, dtype=torch.float32)}

    with pytest.raises(ValueError, match="request_id"):
        _validate_global_update(
            _global_response(expected, request_id="other-request"),
            expected,
            2,
            "fed-request",
        )


def test_round_ack_requires_exact_correlation_and_boolean_payload():
    response = Message(
        type="ack",
        sender="split_server",
        round=2,
        step=4,
        request_id="round-end",
        deadline_at=FUTURE_DEADLINE,
        payload={"server_aggregated": True},
    )

    assert _validate_round_ack(
        response,
        client_id="client-0",
        round_idx=2,
        step=4,
        request_id="round-end",
    )

    response.request_id = "wrong"
    with pytest.raises(ValueError, match="uncorrelated"):
        _validate_round_ack(
            response,
            client_id="client-0",
            round_idx=2,
            step=4,
            request_id="round-end",
        )


class RecordingChannel:
    def __init__(self, response=None, request_id_source=None):
        self.response = response
        self.request_id_source = request_id_source
        self.messages = []

    def send(self, message):
        self.messages.append(message)

    def recv(self):
        if self.request_id_source is not None:
            self.response.request_id = self.request_id_source.messages[
                -1
            ].request_id
        return self.response


def test_client_treats_correlated_split_error_as_fatal():
    client = Client.__new__(Client)
    client.client_id = "client-0"
    client.mode = TrainingMode.splitfed
    client.device = torch.device("cpu")
    client.cfg = SimpleNamespace(runtime=SimpleNamespace(local_steps=1))
    client.model = torch.nn.Linear(1, 1)
    client.optimizer = torch.optim.SGD(client.model.parameters(), lr=0.1)
    client.train_loader = [(torch.ones(1, 1), torch.ones(1))]
    client.to_server = RecordingChannel()
    client.from_server = RecordingChannel(
        Message(
            type="error",
            sender="split_server",
            round=1,
            step=1,
            deadline_at=FUTURE_DEADLINE,
            payload={"reason": "split_batch_timeout"},
        )
    )

    with pytest.raises(RuntimeError, match="invalid split gradient"):
        client.train_one_round(1)


def test_federated_client_trains_complete_model_without_split_channels():
    client = Client.__new__(Client)
    client.client_id = "client-0"
    client.mode = TrainingMode.federated
    client.device = torch.device("cpu")
    client.cfg = SimpleNamespace(runtime=SimpleNamespace(local_steps=1))
    client.model = torch.nn.Linear(2, 1)
    client.criterion = torch.nn.BCEWithLogitsLoss()
    client.optimizer = torch.optim.SGD(client.model.parameters(), lr=0.1)
    client.train_loader = [(torch.ones(2, 2), torch.tensor([0.0, 1.0]))]
    initial_weight = client.model.weight.detach().clone()

    client.train_one_round(1)

    assert not torch.equal(client.model.weight.detach(), initial_weight)


def test_cpu_state_dict_snapshot_is_detached_from_client_model():
    model = torch.nn.Linear(2, 1)
    snapshot = _cpu_state_dict_snapshot(model)

    assert all(value.device.type == "cpu" for value in snapshot.values())
    assert all(
        not value.requires_grad and value.is_contiguous()
        for value in snapshot.values()
    )

    original_weight = snapshot["weight"].clone()
    with torch.no_grad():
        model.weight.add_(1)

    assert torch.equal(snapshot["weight"], original_weight)


def test_federated_payload_is_cpu_snapshot_of_client_model():
    client = Client.__new__(Client)
    client.client_id = "client-0"
    client.model = torch.nn.Linear(2, 1)
    client.dataset = SimpleNamespace(train_dataset=[0])
    client.agg_to_server = RecordingChannel()
    client.agg_from_server = RecordingChannel(
        _global_response(
            client.model.state_dict(),
        ),
        request_id_source=client.agg_to_server,
    )

    client.federative_aggregate(2)

    payload = client.agg_to_server.messages[0].payload["state_dict"]
    assert all(value.device.type == "cpu" for value in payload.values())
    original_weight = payload["weight"].clone()
    with torch.no_grad():
        client.model.weight.add_(1)
    assert torch.equal(payload["weight"], original_weight)


def test_full_epoch_workload_does_not_stop_at_local_steps():
    client = Client.__new__(Client)
    client.cfg = SimpleNamespace(
        runtime=SimpleNamespace(
            local_steps=1,
            workload_policy=WorkloadPolicy.full_epoch_v1,
        )
    )

    assert not client._round_is_complete(step=2)


def test_max_steps_workload_stops_at_configured_limit():
    client = Client.__new__(Client)
    client.cfg = SimpleNamespace(
        runtime=SimpleNamespace(
            local_steps=2,
            workload_policy=WorkloadPolicy.max_steps_v1,
        )
    )

    assert client._round_is_complete(step=2)


def test_fixed_steps_workload_cycles_loader_to_exact_limit():
    client = Client.__new__(Client)
    client.client_id = "client-0"
    client.cfg = SimpleNamespace(
        runtime=SimpleNamespace(
            local_steps=5,
            workload_policy=WorkloadPolicy.fixed_steps_v1,
        )
    )
    client.train_loader = ["batch-1", "batch-2"]

    assert list(client._round_batches()) == [
        "batch-1",
        "batch-2",
        "batch-1",
        "batch-2",
        "batch-1",
    ]
    assert client._round_is_complete(step=5)


def test_fixed_steps_workload_rejects_empty_loader():
    client = Client.__new__(Client)
    client.client_id = "client-0"
    client.cfg = SimpleNamespace(
        runtime=SimpleNamespace(
            local_steps=1,
            workload_policy=WorkloadPolicy.fixed_steps_v1,
        )
    )
    client.train_loader = []

    with pytest.raises(RuntimeError, match="non-empty train loader"):
        list(client._round_batches())


def test_federated_evaluation_writes_to_experiment_metrics_path(tmp_path):
    client = Client.__new__(Client)
    client.client_id = "client-0"
    client.mode = TrainingMode.federated
    client.device = torch.device("cpu")
    client.dataset = SimpleNamespace(test_dataset=[object()])
    client.test_loader = [(torch.ones(1, 2), torch.tensor([0]))]
    client.model = torch.nn.Linear(2, 1)
    client.metrics_path = tmp_path / "artifacts" / "run" / "metrics"
    client.model.weight.data.zero_()
    client.model.bias.data.fill_(-1)

    metrics = client.evaluate()

    assert metrics == {
        "accuracy": 1.0,
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "num_samples": 1,
        "num_positive_labels": 0,
        "num_positive_predictions": 0,
    }
    assert (client.metrics_path / "Clientclient-0_round_0_eval.csv").is_file()
