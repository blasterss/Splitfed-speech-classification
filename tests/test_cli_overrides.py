import pytest

from src.config_profiles import (
    PROFILE_REGISTRY,
    ExperimentProfileDefinition,
    ExperimentProfileRegistry,
)
from src.main import apply_cli_overrides, resolve_raw_config


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


def test_profile_yaml_and_cli_merge_have_explicit_precedence():
    raw_config = {
        "experiment": {"name": "test", "transport": "queue", "seed": 7},
        "training": {"eval_every": 3},
    }

    resolved, provenance = resolve_raw_config(
        raw_config,
        profile_name="smoke",
        overrides=["training.eval_every=2"],
    )

    assert resolved["training"] == {
        "num_rounds": 1,
        "eval_every": 2,
        "barrier_timeout_sec": 30.0,
    }
    assert resolved["experiment"]["profile"] == "smoke"
    assert provenance == {
        "profile": {"name": "smoke", "version": "1", "source": "cli"},
        "cli_overrides": ["training.eval_every=2"],
        "overrides": [
            {
                "source": "cli",
                "path": "training.eval_every",
                "expression": "training.eval_every=2",
            }
        ],
    }


def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError, match="Unknown experiment profile"):
        resolve_raw_config({}, profile_name="missing", overrides=[])


def test_yaml_can_select_registered_profile():
    raw_config = {"experiment": {"profile": "smoke"}, "training": {}}

    resolved, provenance = resolve_raw_config(
        raw_config, profile_name=None, overrides=[]
    )

    assert resolved["training"]["num_rounds"] == 1
    assert provenance["profile"] == {
        "name": "smoke",
        "version": "1",
        "source": "yaml",
    }


def test_profile_registry_exposes_typed_versioned_definitions():
    profile = PROFILE_REGISTRY["smoke"]

    assert isinstance(profile, ExperimentProfileDefinition)
    assert profile.name == "smoke"
    assert profile.version == "1"


def test_profile_registry_rejects_duplicate_names():
    duplicate = ExperimentProfileDefinition("unit", "1", {})

    with pytest.raises(ValueError, match="Duplicate experiment profile"):
        ExperimentProfileRegistry([duplicate, duplicate])
