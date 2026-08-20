import pytest
import torch

from src.utils.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_round_trip_validates_mode_scope_and_state_schema(tmp_path):
    path = tmp_path / "model.pt"
    expected = {"weight": torch.ones(2, dtype=torch.float32)}
    save_checkpoint(
        path,
        mode="split",
        server_model_scope="personalized",
        client_id="client-0",
        model_state_dict=expected,
    )

    restored = load_checkpoint(
        path,
        expected_mode="split",
        expected_server_model_scope="personalized",
        expected_client_id="client-0",
        expected_state_dict=expected,
    )

    assert torch.equal(restored["weight"], expected["weight"])
    assert not path.with_suffix(".pt.tmp").exists()


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"expected_mode": "splitfed"}, "mode mismatch"),
        ({"expected_server_model_scope": "shared"}, "scope mismatch"),
        ({"expected_client_id": "client-1"}, "client_id mismatch"),
    ],
)
def test_checkpoint_rejects_incompatible_ownership(tmp_path, overrides, match):
    path = tmp_path / "model.pt"
    state = {"weight": torch.ones(2)}
    save_checkpoint(
        path,
        mode="split",
        server_model_scope="personalized",
        client_id="client-0",
        model_state_dict=state,
    )
    expected = {
        "expected_mode": "split",
        "expected_server_model_scope": "personalized",
        "expected_client_id": "client-0",
        "expected_state_dict": state,
    }
    expected.update(overrides)

    with pytest.raises(ValueError, match=match):
        load_checkpoint(path, **expected)


@pytest.mark.parametrize(
    "saved,expected,match",
    [
        ({"other": torch.ones(2)}, {"weight": torch.ones(2)}, "keys"),
        ({"weight": torch.ones(3)}, {"weight": torch.ones(2)}, "shape"),
        (
            {"weight": torch.ones(2, dtype=torch.float64)},
            {"weight": torch.ones(2, dtype=torch.float32)},
            "dtype",
        ),
    ],
)
def test_checkpoint_rejects_incompatible_tensor_schema(
    tmp_path, saved, expected, match
):
    path = tmp_path / "model.pt"
    save_checkpoint(path, mode="centralized", model_state_dict=saved)

    with pytest.raises(ValueError, match=match):
        load_checkpoint(
            path,
            expected_mode="centralized",
            expected_state_dict=expected,
        )
