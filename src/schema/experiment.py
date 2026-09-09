"""Experiment and global training configuration models."""

from pydantic import Field, field_validator

from .base import StrictConfigModel
from .enums import ExperimentProfile, TrainingMode, TransportType


class TrainingConfig(StrictConfigModel):
    """Global training process parameters."""

    num_rounds: int = Field(gt=0, description="Total training rounds.")
    mode: TrainingMode = Field(default=TrainingMode.splitfed)
    seed: int = Field(description="Seed for experiment reproducibility.")
    eval_every: int = Field(gt=0, description="Evaluation cadence.")
    fed_every: int = Field(gt=0, description="Aggregation cadence.")
    aggregate_final: bool = Field(
        default=True,
        description="Whether to aggregate after the final training round.",
    )
    barrier_timeout_sec: float = Field(
        default=60.0,
        gt=0,
        description="Maximum client lifecycle barrier wait in seconds.",
    )


class ExperimentConfig(StrictConfigModel):
    """Experiment metadata and execution transport."""

    name: str = Field(description="Experiment name.")
    profile: ExperimentProfile | None = Field(
        default=None,
        description="Resolved built-in experiment profile.",
    )
    description: str | None = Field(
        default=None, description="Optional experiment description."
    )
    analysis_only: bool = Field(
        default=False,
        description="Reject training dispatch for dataset-analysis configs.",
    )
    cross_corpus_evaluation: bool = Field(
        default=False,
        description="Evaluate a final complete-model checkpoint per corpus.",
    )
    transport: TransportType = Field(description="Selected transport.")
    seed: int = Field(description="Seed for experiment reproducibility.")

    @field_validator("name")
    @classmethod
    def validate_artifact_component(cls, value: str) -> str:
        if not value or value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError(
                "experiment.name must be a non-empty path component"
            )
        return value
