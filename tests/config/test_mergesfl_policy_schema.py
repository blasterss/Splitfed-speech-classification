import pytest
from pydantic import ValidationError

from src.schema import MergeSFLPolicyConfig


def test_mergesfl_policy_config_accepts_feasible_bounds():
    config = MergeSFLPolicyConfig(
        max_batch_size=64,
        local_steps=42,
        ingress_budget_bytes=4096,
        feature_bytes_per_sample=128,
        min_clients=2,
        max_clients=3,
    )

    assert config.name == "mergesfl_algorithm1_v1"
    assert config.local_steps == 42


def test_mergesfl_policy_config_rejects_infeasible_minimum_cohort():
    with pytest.raises(ValidationError, match="cannot admit min_clients"):
        MergeSFLPolicyConfig(
            max_batch_size=64,
            local_steps=42,
            ingress_budget_bytes=127,
            feature_bytes_per_sample=64,
            min_clients=2,
            max_clients=3,
        )
