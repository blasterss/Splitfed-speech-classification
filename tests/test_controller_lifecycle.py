from src.schema import ConfigSchema
from src.splitfed.controller import TrainingController


def make_config(dataset_root):
    channel_names = (
        "split_uplink",
        "split_downlink",
        "federated_uplink",
        "federated_downlink",
    )
    channels = {
        name: {
            "transport": "queue",
            "name": name,
            "maxsize": 2,
            "timeout": 1,
        }
        for name in channel_names
    }

    return ConfigSchema(
        data_path=str(dataset_root),
        models_save_path=str(dataset_root / "checkpoints"),
        experiment={"name": "test", "transport": "queue", "seed": 42},
        training={
            "num_rounds": 1,
            "seed": 42,
            "eval_every": 1,
            "fed_every": 1,
        },
        clients=[
            {
                "client_id": 0,
                "dataset": {
                    "name": "SAVEE",
                    "root": str(dataset_root),
                    "feature_names": ["mfcc", "rms", "zcr"],
                    "target_sample_rate": 8000,
                    "reduced": True,
                    "reduced_size": 1,
                    "test_size": 0.2,
                },
                "model": {"lr": 0.001, "optimizer": "adam"},
                "runtime": {
                    "local_steps": 1,
                    "batch_size": 1,
                    "seed": 42,
                    "device": "cpu",
                },
                "noise": {"type": "gauss", "std": 1e-10},
            }
        ],
        split_server={
            "model": {
                "pos_weight": 1,
                "optimizer": "adam",
                "lr": 0.001,
                "device": "cpu",
            },
            "seed": 42,
            "split_uplink_channel": "split_uplink",
            "split_downlink_channel": "split_downlink",
        },
        fed_server={
            "strategy": "fedavg",
            "seed": 42,
            "device": "cpu",
            "aggregation_freq": 1,
            "min_clients": 1,
            "federated_uplink_channel": "federated_uplink",
            "federated_downlink_channel": "federated_downlink",
        },
        channels=channels,
    )


def test_controller_setup_and_teardown_manage_runtime_resources(tmp_path):
    controller = TrainingController(make_config(tmp_path))

    controller.setup()

    assert controller._manager is not None
    assert set(controller.channels[0]) == {
        "split_uplink",
        "split_downlink",
        "federated_uplink",
        "federated_downlink",
    }
    assert controller.split_server is not None
    assert controller.fed_server is not None

    controller.teardown()

    assert controller._manager is None
