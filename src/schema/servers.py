"""Split and federated server configuration models."""

from typing import Literal

from pydantic import Field, model_validator

from .base import StrictConfigModel
from .enums import (
    AggregationStrategy,
    OptimizerType,
    ServerModelScope,
    SplitServerStrategy,
)


class SplitServerModelConfig(StrictConfigModel):
    """Configuration for the server-side split model."""

    pos_weight: float = Field(
        default=3.0,
        gt=0,
        description="BCEWithLogitsLoss positive-class weight.",
    )
    optimizer: OptimizerType = Field(
        default="adam", description="Optimization algorithm."
    )
    learning_rate: float = Field(
        gt=0, alias="lr", description="Server-side learning rate."
    )
    device: str = Field(default="cuda", description="Execution device.")
    gradient_accumulation_steps: Literal[1] = Field(
        default=1,
        description="One optimizer update per strategy batch.",
    )
    batch_timeout_sec: float = Field(
        default=30.0,
        gt=0,
        description="Maximum wait for missing peers in a split batch.",
    )


class SplitServerConfig(StrictConfigModel):
    """Configuration for the split-learning server."""

    model: SplitServerModelConfig = Field(description="Server model settings.")
    model_scope: ServerModelScope = Field(default=ServerModelScope.shared)
    training_strategy: SplitServerStrategy = Field(
        default=SplitServerStrategy.concat_v1,
        description="Shared server-model update ordering.",
    )
    seed: int = Field(description="Seed for server-side randomness.")
    split_uplink_channel: str = Field(
        description="Channel carrying client activations."
    )
    split_downlink_channel: str = Field(
        description="Channel carrying activation gradients."
    )

    @model_validator(mode="after")
    def validate_strategy_scope(self):
        if (
            self.model_scope is ServerModelScope.personalized
            and self.training_strategy is not SplitServerStrategy.concat_v1
        ):
            raise ValueError(
                "split-server training_strategy applies only to shared "
                "server models"
            )
        return self


class FedServerConfig(StrictConfigModel):
    """Configuration for the federated aggregation server."""

    strategy: AggregationStrategy = Field(
        description="Parameter aggregation algorithm."
    )
    seed: int = Field(description="Seed for server-side randomness.")
    device: str = Field(default="cuda", description="Execution device.")
    aggregation_freq: int = Field(
        gt=0, description="Aggregation frequency in training rounds."
    )
    min_clients: int = Field(gt=0, description="Required aggregation quorum.")
    quorum_timeout_sec: float = Field(
        default=30.0,
        gt=0,
        description="Maximum aggregation window after the first update.",
    )
    federated_uplink_channel: str = Field(
        description="Channel carrying client model updates."
    )
    federated_downlink_channel: str = Field(
        description="Channel carrying the global client model."
    )
