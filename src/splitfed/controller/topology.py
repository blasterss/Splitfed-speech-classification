"""Runtime channel and server topology construction."""

from ...logger import get_logger
from ...schema import ConfigSchema, TrainingMode
from ...transport.base import ChannelFactory
from ..fed_server import FedServer
from ..split_server import SplitServer

logger = get_logger(__name__)

SPLIT_UPLINK = "split_uplink"
SPLIT_DOWNLINK = "split_downlink"
FED_UPLINK = "federated_uplink"
FED_DOWNLINK = "federated_downlink"


def build_channels(config: ConfigSchema, mp_context, stop_event) -> dict:
    """Create only the per-client channels required by the selected mode."""
    required = _required_channel_names(config.training.mode)
    channels: dict[str, dict] = {}

    for client_cfg in config.clients:
        client_id = client_cfg.client_id
        client_channels = {}
        channels[client_id] = client_channels

        for name, params in config.channels.items():
            if name not in required:
                logger.warning(
                    "Unexpected channel name '%s' in config – skipping", name
                )
                continue
            client_channels[name] = ChannelFactory.create(
                params,
                mp_context=mp_context,
                stop_event=stop_event,
            )

        missing = required - client_channels.keys()
        if missing:
            raise ValueError(
                f"Client '{client_id}' is missing channel definitions: "
                f"{missing}"
            )

    return channels


def build_servers(
    config: ConfigSchema,
    channels: dict,
    stop_event,
    mp_context,
    failure_queue,
    resource_metrics_queue=None,
) -> tuple[SplitServer | None, FedServer | None]:
    """Construct the servers enabled by validated mode configuration."""
    split_server = None
    if config.split_server is not None:
        split_channels = {
            client_id: {
                "uplink": client_channels[SPLIT_UPLINK],
                "downlink": client_channels[SPLIT_DOWNLINK],
            }
            for client_id, client_channels in channels.items()
        }
        split_server = SplitServer(
            config=config.split_server,
            client_channels=split_channels,
            stop_event=stop_event,
            mp_context=mp_context,
            failure_queue=failure_queue,
            resource_metrics_queue=resource_metrics_queue,
            training_config=config.training,
            fed_server_config=config.fed_server,
        )
    fed_server = None
    if config.fed_server is not None:
        fed_channels = {
            client_id: {
                "uplink": client_channels[FED_UPLINK],
                "downlink": client_channels[FED_DOWNLINK],
            }
            for client_id, client_channels in channels.items()
        }
        fed_server = FedServer(
            config=config.fed_server,
            client_channels=fed_channels,
            num_clients=len(config.clients),
            stop_event=stop_event,
            mp_context=mp_context,
            failure_queue=failure_queue,
            resource_metrics_queue=resource_metrics_queue,
        )
        fed_server.training_mode = config.training.mode.value

    return split_server, fed_server


def _required_channel_names(mode: TrainingMode) -> set[str]:
    required = set()
    if mode in (TrainingMode.split, TrainingMode.splitfed):
        required.update({SPLIT_UPLINK, SPLIT_DOWNLINK})
    if mode in (TrainingMode.federated, TrainingMode.splitfed):
        required.update({FED_UPLINK, FED_DOWNLINK})
    return required
