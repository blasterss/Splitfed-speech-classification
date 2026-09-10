"""Dataset and client configuration models."""

from pathlib import Path

from pydantic import Field, field_validator

from .base import StrictConfigModel
from .enums import (
    DatasetType,
    FeatureType,
    NoiseType,
    OptimizerType,
    WorkloadPolicy,
)


class ClientModelConfig(StrictConfigModel):
    """Configuration for the client-side model."""

    optimizer: OptimizerType = Field(
        default="adam", description="Optimization algorithm."
    )
    learning_rate: float = Field(
        gt=0, alias="lr", description="Learning rate."
    )


class ClientRuntimeConfig(StrictConfigModel):
    """Runtime and local training parameters for a client."""

    local_steps: int = Field(
        gt=0,
        description="Number of local steps between synchronizations.",
    )
    workload_policy: WorkloadPolicy = Field(
        default=WorkloadPolicy.max_steps_v1,
        description=(
            "Whether a round caps at local_steps, exhausts the loader, or "
            "cycles the loader until exactly local_steps are completed."
        ),
    )
    batch_size: int = Field(gt=0, description="Local training batch size.")
    seed: int = Field(description="Seed for client-side randomness.")
    device: str = Field(default="cpu", description="Execution device.")


class NoiseConfig(StrictConfigModel):
    """Optional input perturbation configuration."""

    type: NoiseType = Field(description="Perturbation distribution.")
    std: float = Field(gt=0, description="Standard deviation of the noise.")


class DatasetConfig(StrictConfigModel):
    """Configuration for one actor-disjoint corpus view."""

    name: DatasetType = Field(description="Dataset type.")
    root: str = Field(description="Dataset root on the client device.")
    feature_names: list[FeatureType] = Field(
        default_factory=lambda: [
            FeatureType.mfcc,
            FeatureType.rms,
            FeatureType.zcr,
        ],
        min_length=1,
        description="Ordered non-empty feature list.",
    )
    target_sample_rate: int | None = Field(
        default=None,
        gt=0,
        description="Optional target sample rate for resampling.",
    )
    reduced: bool = Field(
        default=False, description="Whether to use a reduced dataset."
    )
    reduced_size: int | None = Field(
        default=None,
        gt=0,
        description="Maximum files to discover in reduced mode.",
    )
    test_size: float = Field(
        default=0.2,
        gt=0,
        lt=1,
        description="Fraction reserved for the actor-disjoint test split.",
    )
    split_seed: int = Field(
        default=42,
        description="Seed for the actor-disjoint train/test split.",
    )

    @field_validator("root")
    @classmethod
    def validate_root_exists(cls, value: str) -> str:
        path = Path(value)
        if not path.exists():
            raise ValueError(f"Path does not exist: {value}")
        return str(path.absolute())

    @field_validator("feature_names")
    @classmethod
    def validate_unique_feature_names(cls, value):
        if len(value) != len(set(value)):
            raise ValueError("feature_names must not contain duplicates")
        return value


class ClientConfig(StrictConfigModel):
    """Full configuration for a federated learning client."""

    client_id: int = Field(ge=0, description="Unique client identifier.")
    dataset: DatasetConfig = Field(description="Client dataset configuration.")
    model: ClientModelConfig = Field(description="Client model configuration.")
    runtime: ClientRuntimeConfig = Field(
        description="Client runtime settings."
    )
    noise: NoiseConfig | None = Field(
        default=None, description="Optional perturbation configuration."
    )
