import pytest

from src.main import apply_cli_overrides


def test_cli_overrides_parse_typed_values_and_list_paths():
    raw_config = {
        "training": {"num_rounds": 10},
        "clients": [{"runtime": {"batch_size": 8}}],
    }

    resolved = apply_cli_overrides(
        raw_config,
        ["training.num_rounds=2", "clients.0.runtime.batch_size=4"],
    )

    assert resolved["training"]["num_rounds"] == 2
    assert resolved["clients"][0]["runtime"]["batch_size"] == 4
    assert raw_config["training"]["num_rounds"] == 10


@pytest.mark.parametrize(
    "override,match",
    [
        ("training.missing=1", "Unknown override path"),
        ("clients.first.runtime.batch_size=4", "numeric list index"),
        ("training.num_rounds", "expected PATH=VALUE"),
    ],
)
def test_cli_overrides_reject_invalid_or_unknown_paths(override, match):
    raw_config = {
        "training": {"num_rounds": 10},
        "clients": [{"runtime": {"batch_size": 8}}],
    }

    with pytest.raises(ValueError, match=match):
        apply_cli_overrides(raw_config, [override])
