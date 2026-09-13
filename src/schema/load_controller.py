"""Configuration for client admission and round-planning policies."""

from typing import Literal

from pydantic import Field, model_validator

from .base import StrictConfigModel


class WorkerTimingConfig(StrictConfigModel):
    """Bootstrap per-sample timing used before runtime telemetry exists."""

    compute_seconds_per_sample: float = Field(gt=0)
    transfer_seconds_per_sample: float = Field(gt=0)


class MergeSFLPolicyConfig(StrictConfigModel):
    """Versioned configuration for reconstructed MergeSFL Algorithm 1."""

    name: Literal["mergesfl_algorithm1_v1"] = "mergesfl_algorithm1_v1"
    max_batch_size: int = Field(gt=0)
    local_steps: int = Field(gt=0)
    ema_alpha: float = Field(default=0.8, ge=0, le=1)
    ingress_budget_bytes: int = Field(gt=0)
    feature_bytes_per_sample: int = Field(gt=0)
    kl_threshold: float = Field(default=0.05, ge=0)
    min_clients: int = Field(gt=0)
    max_clients: int = Field(gt=0)
    ga_seed: int = 0
    population_size: int = Field(default=20, ge=2)
    generations: int = Field(default=20, gt=0)
    mutation_probability: float = Field(default=0.05, ge=0, le=1)
    telemetry_max_age_sec: float = Field(default=300.0, gt=0)
    round_timeout_sec: float = Field(default=300.0, gt=0)
    initial_worker_states: dict[int, WorkerTimingConfig] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def validate_policy_constraints(self):
        if self.min_clients > self.max_clients:
            raise ValueError("min_clients cannot exceed max_clients")
        minimum_bytes = self.min_clients * self.feature_bytes_per_sample
        if self.ingress_budget_bytes < minimum_bytes:
            raise ValueError(
                "ingress budget cannot admit min_clients with batch size one"
            )
        return self
