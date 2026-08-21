"""
schema.py

Unified Pydantic configuration schema for the Split Federated Learning system.
Describes experiment topology, clients, servers, and communication channels.
"""

from enum import Enum
from pathlib import Path

import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

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


class OptimizerType(str, Enum):
    adam = "adam"


class NoiseType(str, Enum):
    gauss = "gauss"
    laplace = "laplace"


class TrainingMode(str, Enum):
    centralized = "centralized"
    federated = "federated"
    split = "split"
    splitfed = "splitfed"


class ServerModelScope(str, Enum):
    shared = "shared"
    personalized = "personalized"


class ExperimentProfile(str, Enum):
    smoke = "smoke"
    unit = "unit"


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


class StrictConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChannelConfig(StrictConfigModel):
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

    compression: str | None = Field(
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
        gt=0,
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


class ClientModelConfig(StrictConfigModel):
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
        description="Number of local training steps between synchronizations.",
    )

    batch_size: int = Field(gt=0, description="Batch size for local training.")

    seed: int = Field(
        description="Seed for client-side randomness (e.g., data shuffling)."
    )

    device: str = Field(
        default="cpu", description="Execution device (cpu or cuda)."
    )


class NoiseConfig(StrictConfigModel):
    """Noise injection configuration (for secure / robust SFL)."""

    type: NoiseType = Field(description="Supported perturbation distribution.")

    std: float = Field(gt=0, description="Standard deviation of the noise.")


class DatasetConfig(StrictConfigModel):
    """
    Configuration for the dataset used by a client.
    """

    name: DatasetType = Field(description="Type of dataset to use.")

    root: str = Field(description="Path to the dataset on the client device.")

    feature_names: list[FeatureType] | None = Field(
        default=[FeatureType.mfcc, FeatureType.rms, FeatureType.zcr],
        min_length=1,
        description=(
            "List of feature names to extract. "
            "If not specified, all supported features will be extracted."
        ),
    )
    target_sample_rate: int | None = Field(
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

    reduced_size: int | None = Field(
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

    split_seed: int = Field(
        default=42,
        description="Seed used only for the actor-disjoint train/test split.",
    )

    @field_validator("root")
    @classmethod
    def validate_root_exists(cls, v):
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Path does not exist: {v}")
        return str(path.absolute())


class ClientConfig(StrictConfigModel):
    """Full configuration for a federated learning client."""

    client_id: int = Field(ge=0, description="Unique client identifier.")

    dataset: DatasetConfig = Field(description="Client dataset configuration.")

    model: ClientModelConfig = Field(
        description="Client-side model configuration."
    )

    runtime: ClientRuntimeConfig = Field(
        description="Local training and runtime parameters."
    )

    noise: NoiseConfig | None = Field(
        default=None, description="Noise injection configuration (optional)."
    )


# ============================================================
# SPLIT SERVER CONFIGS
# ============================================================


class SplitServerModelConfig(StrictConfigModel):
    """Configuration for the server-side split model."""

    pos_weight: float = Field(
        default=3.0,
        gt=0,
        description=(
            "pos_weight parameter for BCEWithLogitsLoss on the server."
        ),
    )

    optimizer: OptimizerType = Field(
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

    batch_timeout_sec: float = Field(
        default=30.0,
        gt=0,
        description="Maximum wait for missing peers in a split batch.",
    )


class SplitServerConfig(StrictConfigModel):
    """Configuration for the split-learning server."""

    model: SplitServerModelConfig = Field(
        description="Server-side model configuration."
    )

    model_scope: ServerModelScope = Field(default=ServerModelScope.shared)

    seed: int = Field(
        description="Seed for server-side randomness (e.g., client sampling)."
    )

    split_uplink_channel: str = Field(
        description=(
            "Channel name for transmitting activations from clients to server."
        )
    )

    split_downlink_channel: str = Field(
        description=(
            "Channel name for transmitting gradients from server to clients."
        )
    )


# ============================================================
# FEDERATED SERVER CONFIGS
# ============================================================


class FedServerConfig(StrictConfigModel):
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
        description=(
            "Minimum number of clients required to perform aggregation."
        ),
    )

    quorum_timeout_sec: float = Field(
        default=30.0,
        gt=0,
        description="Maximum aggregation window after the first update.",
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


class TrainingConfig(StrictConfigModel):
    """Global training process parameters."""

    num_rounds: int = Field(
        gt=0, description="Total number of training rounds."
    )

    mode: TrainingMode = Field(default=TrainingMode.splitfed)

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


class ExperimentConfig(StrictConfigModel):
    """Experiment metadata and execution mode."""

    name: str = Field(description="Experiment name.")

    profile: ExperimentProfile | None = Field(
        default=None,
        description="Resolved name of the built-in experiment profile.",
    )

    description: str | None = Field(
        default=None,
        description="Optional free-form text description of the experiment.",
    )

    transport: TransportType = Field(description="Transport type in use.")

    seed: int = Field(description="Seed for experiment reproducibility.")

    @field_validator("name")
    @classmethod
    def validate_artifact_component(cls, value: str) -> str:
        if not value or value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError(
                "experiment.name must be a non-empty path component"
            )
        return value


# ============================================================
# ROOT CONFIG
# ============================================================


class ConfigSchema(StrictConfigModel):
    """
    Root configuration schema for a Split Federated Learning experiment.
    """

    data_path: str = Field(
        description="Path to the directory containing the source data."
    )

    models_save_path: str | None = Field(
        default=None,
        description=(
            "Artifact root for experiment metadata, checkpoints and metrics. "
            "If not specified, run artifacts will not be saved."
        ),
    )

    experiment: ExperimentConfig = Field(
        description="General experiment parameters."
    )

    training: TrainingConfig = Field(
        description="Training process parameters."
    )

    clients: list[ClientConfig] = Field(
        min_length=1,
        description="List of client configurations.",
    )

    split_server: SplitServerConfig | None = Field(
        default=None, description="Split-learning server configuration."
    )

    fed_server: FedServerConfig | None = Field(
        default=None, description="Federated server configuration."
    )

    channels: dict[str, QueueChannelConfig | GRPCChannelConfig] = Field(
        description="Dictionary of communication channels used in the system."
    )

    @model_validator(mode="after")
    def validate_topology(self):
        if self.experiment.transport is not TransportType.queue:
            raise ValueError(
                "Only queue transport is operational; grpc is a stub"
            )
        for channel_name, channel in self.channels.items():
            if channel.transport is not TransportType.queue:
                raise ValueError(
                    f"Channel {channel_name!r} selects non-operational grpc"
                )
            if channel.compression is not None:
                raise ValueError(
                    f"Channel {channel_name!r} compression is not implemented"
                )

        device_fields = [
            (
                f"clients[{client.client_id}].runtime.device",
                client.runtime.device,
            )
            for client in self.clients
        ]
        if self.split_server:
            device_fields.append(
                ("split_server.model.device", self.split_server.model.device)
            )
        if self.fed_server:
            device_fields.append(("fed_server.device", self.fed_server.device))
        for field, device in device_fields:
            _validate_device_available(device, field)

        client_ids = [client.client_id for client in self.clients]
        if len(client_ids) != len(set(client_ids)):
            raise ValueError("Client IDs must be unique")

        mode = self.training.mode
        needs_split = mode in (TrainingMode.split, TrainingMode.splitfed)
        needs_fed = mode in (TrainingMode.federated, TrainingMode.splitfed)

        if mode is TrainingMode.centralized:
            owner = self.clients[0]
            owner_signature = (
                owner.model,
                owner.runtime.batch_size,
                owner.runtime.device,
                owner.noise,
                owner.dataset.feature_names,
                owner.dataset.target_sample_rate,
            )
            for client in self.clients[1:]:
                signature = (
                    client.model,
                    client.runtime.batch_size,
                    client.runtime.device,
                    client.noise,
                    client.dataset.feature_names,
                    client.dataset.target_sample_rate,
                )
                if signature != owner_signature:
                    raise ValueError(
                        "centralized dataset views must share model, batch "
                        "size, device, noise, feature ordering and sample rate"
                    )

        if needs_split != (self.split_server is not None):
            requirement = "required" if needs_split else "not allowed"
            raise ValueError(
                f"split_server is {requirement} for mode {mode.value}"
            )
        if needs_fed != (self.fed_server is not None):
            requirement = "required" if needs_fed else "not allowed"
            raise ValueError(
                f"fed_server is {requirement} for mode {mode.value}"
            )

        if (
            mode is TrainingMode.splitfed
            and self.split_server.model_scope is not ServerModelScope.shared
        ):
            raise ValueError("splitfed requires shared server model scope")

        if self.fed_server and self.fed_server.min_clients > len(self.clients):
            raise ValueError(
                "fed_server.min_clients cannot exceed client count"
            )
        if (
            self.fed_server
            and self.fed_server.aggregation_freq != self.training.fed_every
        ):
            raise ValueError(
                "fed_server.aggregation_freq must equal training.fed_every"
            )

        required_channels = set()
        channel_references = set()
        if self.split_server:
            required_channels.update({"split_uplink", "split_downlink"})
            channel_references.update(
                {
                    self.split_server.split_uplink_channel,
                    self.split_server.split_downlink_channel,
                }
            )
            if (
                self.split_server.split_uplink_channel != "split_uplink"
                or self.split_server.split_downlink_channel != "split_downlink"
            ):
                raise ValueError(
                    "split_server channel references must match canonical "
                    "split_uplink/split_downlink roles"
                )
        if self.fed_server:
            required_channels.update(
                {"federated_uplink", "federated_downlink"}
            )
            channel_references.update(
                {
                    self.fed_server.federated_uplink_channel,
                    self.fed_server.federated_downlink_channel,
                }
            )
            if (
                self.fed_server.federated_uplink_channel != "federated_uplink"
                or self.fed_server.federated_downlink_channel
                != "federated_downlink"
            ):
                raise ValueError(
                    "fed_server channel references must match canonical "
                    "federated_uplink/federated_downlink roles"
                )

        configured_channels = set(self.channels)
        if configured_channels != required_channels:
            raise ValueError(
                f"Mode {mode.value} requires exactly channels "
                f"{sorted(required_channels)}"
            )

        missing_channels = channel_references - self.channels.keys()
        if missing_channels:
            missing = ", ".join(sorted(missing_channels))
            raise ValueError(
                f"Undefined server channel reference(s): {missing}"
            )

        return self


def _validate_device_available(device: str, field: str) -> None:
    if device == "cpu":
        return
    if device == "cuda":
        index = 0
    elif device.startswith("cuda:") and device[5:].isdigit():
        index = int(device[5:])
    else:
        raise ValueError(
            f"{field} must be 'cpu', 'cuda', or an indexed CUDA device"
        )
    if not torch.cuda.is_available():
        raise ValueError(f"{field} requests CUDA but CUDA is unavailable")
    if index >= torch.cuda.device_count():
        raise ValueError(
            f"{field} requests cuda:{index}, but only "
            f"{torch.cuda.device_count()} CUDA device(s) are available"
        )
