from types import SimpleNamespace

import pytest
import torch

from src.schema import TrainingMode
from src.splitfed.client import (
    Client,
    _extract_payload,
    _validate_global_update,
)
from src.transport.message import Message


def _response(**overrides):
    values = {
        "type": "gradients",
        "sender": "split_server",
        "round": 2,
        "step": 3,
        "payload": {"gradients": torch.ones(1)},
    }
    values.update(overrides)
    return Message(**values)


def test_extract_payload_accepts_correlated_split_response():
    tensor = _extract_payload(
        _response(),
        "gradients",
        "gradients",
        "client-0",
        2,
        3,
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
    assert (
        _extract_payload(
            _response(**overrides),
            "gradients",
            "gradients",
            "client-0",
            2,
            3,
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
    }
    values.update(overrides)
    return Message(**values)


def test_global_update_accepts_matching_state_schema():
    expected = {"weight": torch.ones(2, dtype=torch.float32)}
    received = {"weight": torch.zeros(2, dtype=torch.float32)}

    assert (
        _validate_global_update(_global_response(received), expected, 2)
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
        _validate_global_update(response, expected, 2)


class RecordingChannel:
    def __init__(self, response=None):
        self.response = response
        self.messages = []

    def send(self, message):
        self.messages.append(message)

    def recv(self):
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


def test_federated_evaluation_creates_results_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = Client.__new__(Client)
    client.client_id = "client-0"
    client.mode = TrainingMode.federated
    client.device = torch.device("cpu")
    client.dataset = SimpleNamespace(test_dataset=[object()])
    client.test_loader = [(torch.ones(1, 2), torch.tensor([0]))]
    client.model = torch.nn.Linear(2, 1)
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
    assert (
        tmp_path / "experiments/results/Clientclient-0_round_0_eval.csv"
    ).is_file()
