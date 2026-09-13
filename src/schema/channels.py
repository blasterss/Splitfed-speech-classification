"""Communication channel configuration models."""

from pydantic import Field

from .base import StrictConfigModel
from .enums import TransportType


class ChannelConfig(StrictConfigModel):
    """Base configuration for a logical communication channel."""

    transport: TransportType = Field(
        description="Transport type (queue or grpc)."
    )
    name: str = Field(description="Unique logical name of the channel.")
    buffer_size: int = Field(
        default=0, ge=0, description="Channel buffer size (0 is unlimited)."
    )
    compression: str | None = Field(
        default=None,
        description="Compression type for transmitted data, if supported.",
    )


class QueueChannelConfig(ChannelConfig):
    """Local multiprocessing queue channel configuration."""

    maxsize: int = Field(
        default=0, ge=0, description="Maximum queue size (0 is unlimited)."
    )
    timeout: float = Field(
        default=60.0,
        gt=0,
        description="Timeout for waiting on a queue message in seconds.",
    )


class GRPCChannelConfig(ChannelConfig):
    """Per-client endpoint configuration for a gRPC logical channel."""

    addresses: dict[int, str] = Field(
        min_length=1,
        description="Receiver address keyed by configured client ID.",
    )
    use_tls: bool = Field(default=False, description="Whether to use TLS.")
    timeout_sec: float = Field(
        default=30.0,
        gt=0,
        description="Timeout for waiting on a gRPC response in seconds.",
    )
