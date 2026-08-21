import torch

from src.model.client_side_model import ClientSideModel, PrivacyLayer
from src.model.server_side_model import ServerSideModel
from src.model.speech_model import SpeechRecognitionModel


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
