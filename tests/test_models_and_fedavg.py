import torch

from src.model.client_side_model import ClientSideModel, PrivacyLayer
from src.model.server_side_model import ServerSideModel
from src.model.speech_model import SpeechRecognitionModel
from src.splitfed.fed_server import FedServer
from src.utils.state import deserialize_state_dict, serialize_state_dict


def test_client_model_produces_split_activations_on_cpu():
    model = ClientSideModel(
        input_channels=3,
        output_channels=8,
        noise=False,
    )

    activations = model(torch.randn(2, 3, 64))

    assert activations.shape == (2, 8, 128)
    assert activations.device.type == "cpu"


def test_server_models_produce_binary_logits_on_cpu():
    activations = torch.randn(2, 8, 128)

    for model_type in ("cnn_gap", "cnn_birnn"):
        model = ServerSideModel(
            input_channels=8,
            output_channels=16,
            model_type=model_type,
            rnn_hidden=8,
            rnn_layers=1,
            cnn_dropout=0,
            fc_dropout=0,
        )

        logits = model(activations)

        assert logits.shape == (2, 1)
        assert logits.device.type == "cpu"


def test_fedavg_weights_floating_parameters_by_dataset_size():
    client_params = [
        {"weight": torch.tensor([0.0])},
        {"weight": torch.tensor([10.0])},
    ]

    aggregated = FedServer.aggregate(client_params, [1, 3])

    assert torch.equal(aggregated["weight"], torch.tensor([7.5]))


def test_fedavg_preserves_integer_buffer_from_largest_client():
    client_params = [
        {"weight": torch.tensor([1.0]), "counter": torch.tensor(2)},
        {"weight": torch.tensor([5.0]), "counter": torch.tensor(8)},
    ]

    aggregated = FedServer.aggregate(client_params, [1, 3])

    assert torch.equal(aggregated["weight"], torch.tensor([4.0]))
    assert aggregated["counter"].dtype == torch.int64
    assert torch.equal(aggregated["counter"], torch.tensor(8))


def test_state_dict_bytes_round_trip_without_shared_tensor_storage():
    state_dict = {
        "weight": torch.tensor([1.5], dtype=torch.float32),
        "counter": torch.tensor(2, dtype=torch.int64),
    }

    restored = deserialize_state_dict(serialize_state_dict(state_dict))

    assert restored.keys() == state_dict.keys()
    assert all(
        torch.equal(restored[key], value) for key, value in state_dict.items()
    )


def test_complete_speech_model_produces_binary_logits():
    model = SpeechRecognitionModel(
        input_channels=3,
        server_side_model_type="cnn_gap",
    )

    logits = model(torch.randn(2, 3, 64))

    assert logits.shape == (2, 1)


def test_privacy_noise_is_disabled_during_evaluation():
    layer = PrivacyLayer(noise_std=1.0, noise_type="gauss")
    inputs = torch.ones(2, 3)
    layer.eval()

    first = layer(inputs)
    second = layer(inputs)

    assert torch.equal(first, inputs)
    assert torch.equal(second, inputs)
