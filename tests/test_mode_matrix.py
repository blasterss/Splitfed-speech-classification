import pytest

from src.experiments.mode_matrix import MODE_VARIANTS, build_mode_config


def _base():
    return {
        "models_save_path": "artifacts",
        "experiment": {"name": "matrix"},
        "training": {
            "mode": "splitfed",
            "num_rounds": 30,
            "eval_every": 10,
            "fed_every": 10,
        },
        "split_server": {"model_scope": "shared"},
        "fed_server": {"aggregation_freq": 10},
        "channels": {
            "split_uplink": {"name": "split_uplink"},
            "split_downlink": {"name": "split_downlink"},
            "federated_uplink": {"name": "federated_uplink"},
            "federated_downlink": {"name": "federated_downlink"},
        },
    }


@pytest.mark.parametrize("variant", MODE_VARIANTS)
def test_mode_matrix_derives_owned_roles_channels_and_final_frequency(variant):
    resolved = build_mode_config(_base(), variant, rounds=3)

    assert resolved["training"]["num_rounds"] == 3
    assert resolved["training"]["eval_every"] == 3
    assert resolved["experiment"]["name"] == f"matrix-{variant}-r3"

    if variant == "centralized":
        assert resolved["split_server"] is None
        assert resolved["fed_server"] is None
        assert resolved["channels"] == {}
    elif variant == "federated":
        assert resolved["split_server"] is None
        assert set(resolved["channels"]) == {
            "federated_uplink",
            "federated_downlink",
        }
    elif variant.startswith("split-"):
        assert resolved["split_server"]["model_scope"] == variant.removeprefix(
            "split-"
        )
        assert resolved["fed_server"] is None
        assert set(resolved["channels"]) == {
            "split_uplink",
            "split_downlink",
        }
    else:
        assert resolved["split_server"]["model_scope"] == "shared"
        assert set(resolved["channels"]) == {
            "split_uplink",
            "split_downlink",
            "federated_uplink",
            "federated_downlink",
        }

    if resolved["fed_server"] is not None:
        assert resolved["training"]["fed_every"] == 3
        assert resolved["fed_server"]["aggregation_freq"] == 3


def test_mode_matrix_rejects_unknown_variant_and_nonpositive_rounds():
    with pytest.raises(ValueError, match="Unknown"):
        build_mode_config(_base(), "missing", rounds=1)
    with pytest.raises(ValueError, match="positive"):
        build_mode_config(_base(), "centralized", rounds=0)
