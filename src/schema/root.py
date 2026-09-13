"""Root configuration model and cross-block topology validation."""

import torch
from pydantic import Field, model_validator

from .base import StrictConfigModel
from .channels import GRPCChannelConfig, QueueChannelConfig
from .clients import ClientConfig
from .enums import (
    AggregationStrategy,
    ServerModelScope,
    SplitServerStrategy,
    TrainingMode,
    WorkloadPolicy,
)
from .experiment import ExperimentConfig, TrainingConfig
from .load_controller import MergeSFLPolicyConfig
from .servers import FedServerConfig, SplitServerConfig


class ConfigSchema(StrictConfigModel):
    """Root configuration schema for one training experiment."""

    data_path: str = Field(
        description="Path to the directory containing source data."
    )
    models_save_path: str | None = Field(
        default=None,
        description="Artifact root; no artifacts are saved when omitted.",
    )
    experiment: ExperimentConfig = Field(description="Experiment metadata.")
    training: TrainingConfig = Field(description="Training parameters.")
    clients: list[ClientConfig] = Field(
        min_length=1, description="Configured clients."
    )
    split_server: SplitServerConfig | None = Field(
        default=None, description="Optional split-learning server."
    )
    fed_server: FedServerConfig | None = Field(
        default=None, description="Optional federated server."
    )
    load_controller: MergeSFLPolicyConfig | None = Field(
        default=None, description="Optional dynamic client round policy."
    )
    channels: dict[str, QueueChannelConfig | GRPCChannelConfig] = Field(
        description="Logical communication channels."
    )

    @model_validator(mode="after")
    def validate_topology(self):
        self._validate_transport()
        self._validate_devices()
        self._validate_clients()
        self._validate_mode_ownership()
        self._validate_split_strategy()
        self._validate_load_controller()
        self._validate_experiment_contract()
        self._validate_channels()
        return self

    def _validate_transport(self) -> None:
        client_ids = {client.client_id for client in self.clients}
        addresses = []
        for channel_name, channel in self.channels.items():
            if channel.transport is not self.experiment.transport:
                raise ValueError(
                    f"Channel {channel_name!r} transport must match experiment"
                )
            if channel.compression is not None:
                raise ValueError(
                    f"Channel {channel_name!r} compression is not implemented"
                )
            if isinstance(channel, GRPCChannelConfig):
                if channel.use_tls:
                    raise ValueError(
                        "gRPC TLS credentials are not implemented; "
                        "use_tls must be false"
                    )
                if set(channel.addresses) != client_ids:
                    raise ValueError(
                        f"Channel {channel_name!r} gRPC addresses must match "
                        "configured client IDs"
                    )
                for client_id, address in channel.addresses.items():
                    if not address.strip():
                        raise ValueError(
                            f"Channel {channel_name!r} has an empty gRPC "
                            f"address for client {client_id!r}"
                        )
                    addresses.append(address)
        if len(addresses) != len(set(addresses)):
            raise ValueError("gRPC channel addresses must be unique")

    def _validate_devices(self) -> None:
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

    def _validate_clients(self) -> None:
        client_ids = [client.client_id for client in self.clients]
        if len(client_ids) != len(set(client_ids)):
            raise ValueError("Client IDs must be unique")

        if self.training.mode is not TrainingMode.centralized:
            return
        owner = self.clients[0]
        owner_signature = _centralized_client_signature(owner)
        for client in self.clients[1:]:
            if _centralized_client_signature(client) != owner_signature:
                raise ValueError(
                    "centralized dataset views must share model, batch "
                    "size, device, noise, feature ordering and sample rate"
                )

    def _validate_mode_ownership(self) -> None:
        mode = self.training.mode
        needs_split = mode in (TrainingMode.split, TrainingMode.splitfed)
        needs_fed = mode in (TrainingMode.federated, TrainingMode.splitfed)

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

    def _validate_split_strategy(self) -> None:
        if (
            self.split_server is None
            or self.split_server.training_strategy
            is not SplitServerStrategy.mergesfl_v1
        ):
            return
        if any(
            client.runtime.workload_policy
            is not WorkloadPolicy.fixed_steps_v1
            for client in self.clients
        ):
            raise ValueError(
                "mergesfl_v1 requires fixed_steps_v1 for every client"
            )
        if any(not client.runtime.drop_last for client in self.clients):
            raise ValueError("mergesfl_v1 requires drop_last=true")
        local_steps = {client.runtime.local_steps for client in self.clients}
        if len(local_steps) != 1:
            raise ValueError(
                "mergesfl_v1 requires equal local_steps for every client"
            )

    def _validate_load_controller(self) -> None:
        strategy = (
            self.split_server.training_strategy
            if self.split_server is not None
            else None
        )
        algorithm1 = strategy is SplitServerStrategy.mergesfl_algorithm1_v1
        if algorithm1 != (self.load_controller is not None):
            raise ValueError(
                "mergesfl_algorithm1_v1 requires exactly one load_controller"
            )
        if not algorithm1:
            return
        assert self.load_controller is not None
        assert self.split_server is not None
        assert self.fed_server is not None
        if self.training.mode is not TrainingMode.splitfed:
            raise ValueError("MergeSFL Algorithm 1 requires splitfed mode")
        if self.split_server.model_scope is not ServerModelScope.shared:
            raise ValueError("MergeSFL Algorithm 1 requires shared server")
        if (
            self.fed_server.strategy
            is not AggregationStrategy.mergesfl_batch_weighted_v1
        ):
            raise ValueError(
                "MergeSFL Algorithm 1 requires batch-weighted aggregation"
            )
        if self.fed_server.min_clients != self.load_controller.min_clients:
            raise ValueError(
                "MergeSFL policy and FedServer min_clients must match"
            )
        if self.load_controller.max_clients > len(self.clients):
            raise ValueError("MergeSFL max_clients exceeds client count")
        if self.training.fed_every != 1:
            raise ValueError("MergeSFL Algorithm 1 requires fed_every=1")
        if not self.training.aggregate_final:
            raise ValueError("MergeSFL Algorithm 1 requires aggregate_final")
        if any(not client.runtime.drop_last for client in self.clients):
            raise ValueError("MergeSFL Algorithm 1 requires drop_last=true")
        client_ids = {client.client_id for client in self.clients}
        if set(self.load_controller.initial_worker_states) != client_ids:
            raise ValueError(
                "MergeSFL initial_worker_states must match client IDs"
            )

    def _validate_channels(self) -> None:
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
                f"Mode {self.training.mode.value} requires exactly channels "
                f"{sorted(required_channels)}"
            )
        missing_channels = channel_references - self.channels.keys()
        if missing_channels:
            missing = ", ".join(sorted(missing_channels))
            raise ValueError(
                f"Undefined server channel reference(s): {missing}"
            )

    def _validate_experiment_contract(self) -> None:
        if self.experiment.analysis_only:
            if self.experiment.cross_corpus_evaluation:
                raise ValueError(
                    "analysis_only cannot enable cross_corpus_evaluation"
                )
            return
        if not self.experiment.cross_corpus_evaluation:
            return
        if self.training.mode not in (
            TrainingMode.centralized,
            TrainingMode.federated,
        ):
            raise ValueError(
                "cross_corpus_evaluation requires a complete-model "
                "centralized or federated mode"
            )
        if self.models_save_path is None:
            raise ValueError(
                "cross_corpus_evaluation requires models_save_path"
            )
        if self.training.mode is TrainingMode.federated and (
            not self.training.aggregate_final
            or self.training.num_rounds % self.training.fed_every != 0
        ):
            raise ValueError(
                "federated cross_corpus_evaluation requires final-round "
                "aggregation"
            )
        corpus_names = [client.dataset.name for client in self.clients]
        if len(corpus_names) != len(set(corpus_names)):
            raise ValueError(
                "cross_corpus_evaluation requires unique corpus names"
            )


def _centralized_client_signature(client: ClientConfig) -> tuple:
    return (
        client.model,
        client.runtime.batch_size,
        client.runtime.device,
        client.noise,
        client.dataset.feature_names,
        client.dataset.target_sample_rate,
    )


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
