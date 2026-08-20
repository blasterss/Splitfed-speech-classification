"""
schema.py

Unified Pydantic configuration schema for the Split Federated Learning (SFL) system.
Used to describe the experiment topology, clients, servers, and communication channels.
"""

from enum import Enum
from typing import List, Optional, Dict, Union
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


# ============================================================
# ENUMS
# ============================================================


class TransportType(str, Enum):
    """Transport layer type for inter-component communication."""

    queue = "queue"
    grpc = "grpc"


class DatasetType(str, Enum):
    """Supported datasets."""

    crema_d = "CREMA-D"
    ravdess = "RAVDESS"
    savee = "SAVEE"


class AggregationStrategy(str, Enum):
    """Parameter aggregation strategy on the federated server."""

    fedavg = "fedavg"
    weighted_fedavg = "weighted_fedavg"


class FeatureType(str, Enum):
    """Types of extracted features."""

    mel = "mel"
    mfcc = "mfcc"
    rms = "rms"
    zcr = "zcr"
    contrast = "contrast"


# ============================================================
# CHANNEL CONFIGS
# ============================================================


class ChannelConfig(BaseModel):
    """
    Base configuration for a communication channel between system components.
    Used as an abstraction for both local and network-based interaction.
    """

    transport: TransportType = Field(
        description=(
            "Transport type (queue — local, grpc — network interaction)."
        )
    )

    name: str = Field(description="Unique logical name of the channel.")

    buffer_size: int = Field(
        default=0, ge=0, description="Channel buffer size (0 — unlimited)."
    )

    compression: Optional[str] = Field(
        default=None,
        description="Compression type for transmitted data (if supported).",
    )


class QueueChannelConfig(ChannelConfig):
    """
    Communication channel based on multiprocessing.Queue / Pipe.
    Used for local simulation of a distributed environment.
    """

    maxsize: int = Field(
        default=0, ge=0, description="Maximum queue size (0 — unlimited)."
    )

    timeout: float = Field(
        default=60.0,
        ge=0,
        description="Timeout for waiting on a queue message (in seconds).",
    )


class GRPCChannelConfig(ChannelConfig):
    """
    gRPC channel configuration for network-based component interaction.
    """

    address: str = Field(
        description="Network address of the gRPC service in host:port format."
    )

    use_tls: bool = Field(
        default=True, description="Use TLS for a secured connection."
    )

    timeout_sec: int = Field(
        default=30,
        gt=0,
        description="Timeout for waiting on a gRPC response (in seconds).",
    )


# ============================================================
# CLIENT CONFIGS
# ============================================================


class ClientModelConfig(BaseModel):
    """Configuration for the client-side model."""

    # input_dim: int = Field(
    #     gt=0,
    #     description="Number of input channels for the client model."
    # )
    # output_dim: int = Field(
    #     gt=0,
    #     description="Number of output channels for the client model."
    # )
    # kernel_size: int = Field(
    #     gt=0,
    #     description="Convolution kernel size for the client model."
    # )
    # padding: int = Field(
    #     ge=0,
    #     description="Padding size for the client model."
    # )

    optimizer: str = Field(
        default="adam", description="Optimization algorithm."
    )

    learning_rate: float = Field(
        gt=0, alias="lr", description="Learning rate."
    )


class ClientRuntimeConfig(BaseModel):
    """Runtime and local training parameters for a client."""

    local_steps: int = Field(
        gt=0,
        description="Number of local training steps between synchronizations.",
    )

    batch_size: int = Field(gt=0, description="Batch size for local training.")

    seed: int = Field(
        description="Seed for client-side randomness (e.g., data shuffling)."
    )

    device: str = Field(
        default="cpu", description="Execution device (cpu or cuda)."
    )


class NoiseConfig(BaseModel):
    """Noise injection configuration (for secure / robust SFL)."""

    type: str = Field(description="Noise type (gauss, laplace, etc.).")

    std: float = Field(gt=0, description="Standard deviation of the noise.")


class DatasetConfig(BaseModel):
    """
    Configuration for the dataset used by a client.
    """

    name: DatasetType = Field(description="Type of dataset to use.")

    root: str = Field(description="Path to the dataset on the client device.")

    feature_names: Optional[List[FeatureType]] = Field(
        default=[FeatureType.mfcc, FeatureType.rms, FeatureType.zcr],
        description=(
            "List of feature names to extract. "
            "If not specified, all supported features will be extracted."
        ),
    )
    target_sample_rate: Optional[int] = Field(
        default=None,
        gt=0,
        description=(
            "Target sample rate. "
            "If specified and different from the original sample rate, "
            "the data will be resampled."
        ),
    )

    reduced: bool = Field(
        default=False,
        description="Whether to use a reduced dataset (for testing).",
    )

    reduced_size: Optional[int] = Field(
        default=None,
        gt=0,
        description=(
            "Maximum number of files in reduced mode. "
            "If not specified, the default value is used."
        ),
    )

    test_size: float = Field(
        default=0.2,
        gt=0,
        lt=1,
        description="Fraction of data reserved for the test split.",
    )

    @field_validator("root")
    @classmethod
    def validate_root_exists(cls, v):
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Path does not exist: {v}")
        return str(path.absolute())


class ClientConfig(BaseModel):
    """Full configuration for a federated learning client."""

    client_id: int = Field(ge=0, description="Unique client identifier.")

    dataset: DatasetConfig = Field(description="Client dataset configuration.")

    model: ClientModelConfig = Field(
        description="Client-side model configuration."
    )

    runtime: ClientRuntimeConfig = Field(
        description="Local training and runtime parameters."
    )

    noise: Optional[NoiseConfig] = Field(
        default=None, description="Noise injection configuration (optional)."
    )


# ============================================================
# SPLIT SERVER CONFIGS
# ============================================================


class SplitServerModelConfig(BaseModel):
    """Configuration for the server-side split model."""

    pos_weight: float = Field(
        default=3.0,
        gt=0,
        description="pos_weight parameter for BCEWithLogitsLoss on the server.",
    )

    optimizer: str = Field(
        default="adam",
        description="Optimization algorithm for the server-side model.",
    )

    learning_rate: float = Field(
        gt=0,
        alias="lr",
        description="Learning rate for the server-side model.",
    )

    device: str = Field(
        default="cuda",
        description="Execution device for the server-side model.",
    )

    gradient_accumulation_steps: int = Field(
        default=4,
        gt=0,
        description="Server batches averaged per optimizer update.",
    )


class SplitServerConfig(BaseModel):
    """Configuration for the split-learning server."""

    model: SplitServerModelConfig = Field(
        description="Server-side model configuration."
    )

    seed: int = Field(
        description="Seed for server-side randomness (e.g., client sampling)."
    )

    split_uplink_channel: str = Field(
        description="Channel name for transmitting activations from clients to the server."
    )

    split_downlink_channel: str = Field(
        description="Channel name for transmitting gradients from the server to clients."
    )


# ============================================================
# FEDERATED SERVER CONFIGS
# ============================================================


class FedServerConfig(BaseModel):
    """Configuration for the federated aggregation server."""

    strategy: AggregationStrategy = Field(
        description="Parameter aggregation algorithm."
    )

    seed: int = Field(
        description="Seed for server-side randomness (e.g., client sampling)."
    )

    device: str = Field(
        default="cuda",
        description="Execution device for the server-side model.",
    )

    aggregation_freq: int = Field(
        gt=0, description="Aggregation frequency (in training rounds)."
    )

    min_clients: int = Field(
        gt=0,
        description="Minimum number of clients required to perform aggregation.",
    )

    federated_uplink_channel: str = Field(
        description="Channel name for receiving parameters from client models."
    )

    federated_downlink_channel: str = Field(
        description="Channel name for broadcasting global model parameters."
    )


# ============================================================
# EXPERIMENT & TRAINING CONFIGS
# ============================================================


class TrainingConfig(BaseModel):
    """Global training process parameters."""

    num_rounds: int = Field(
        gt=0, description="Total number of training rounds."
    )

    seed: int = Field(description="Seed for experiment reproducibility.")

    eval_every: int = Field(
        gt=0, description="Validation frequency (in rounds)."
    )

    fed_every: int = Field(
        gt=0, description="Federated aggregation frequency (in rounds)."
    )

    barrier_timeout_sec: float = Field(
        default=60.0,
        gt=0,
        description="Maximum wait for client lifecycle barriers in seconds.",
    )


class ExperimentConfig(BaseModel):
    """Experiment metadata and execution mode."""

    name: str = Field(description="Experiment name.")

    description: Optional[str] = Field(
        default=None,
        description="Optional free-form text description of the experiment.",
    )

    transport: TransportType = Field(description="Transport type in use.")

    seed: int = Field(description="Seed for experiment reproducibility.")


# ============================================================
# ROOT CONFIG
# ============================================================


class ConfigSchema(BaseModel):
    """
    Root configuration schema for a Split Federated Learning experiment.
    """

    data_path: str = Field(
        description="Path to the directory containing the source data."
    )

    models_save_path: Optional[str] = Field(
        default=None,
        description=(
            "Path for saving trained models. "
            "If not specified, models will not be saved."
        ),
    )

    experiment: ExperimentConfig = Field(
        description="General experiment parameters."
    )

    training: TrainingConfig = Field(
        description="Training process parameters."
    )

    clients: List[ClientConfig] = Field(
        description="List of client configurations."
    )

    split_server: SplitServerConfig = Field(
        description="Split-learning server configuration."
    )

    fed_server: FedServerConfig = Field(
        description="Federated server configuration."
    )

    channels: Dict[str, Union[QueueChannelConfig, GRPCChannelConfig]] = Field(
        description="Dictionary of communication channels used in the system."
    )
