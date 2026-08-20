import queue

import torch
import torch.multiprocessing as mp
from torch.utils.data import TensorDataset

from src.schema import ConfigSchema
from src.splitfed.centralized import (
    CentralizedTrainer,
    _centralized_training_worker,
    _validate_centralized_shapes,
)
from src.utils.state import deserialize_state_dict, serialize_state_dict


class _CompletedProcess:
    exitcode = 0

    def is_alive(self):
        return False

    def join(self, timeout=None):
        pass


def _centralized_config(tmp_path):
    return ConfigSchema(
        data_path=str(tmp_path),
        models_save_path=str(tmp_path / "checkpoints"),
        experiment={"name": "centralized", "transport": "queue", "seed": 7},
        training={
            "mode": "centralized",
            "num_rounds": 1,
            "seed": 7,
            "eval_every": 1,
            "fed_every": 1,
        },
        clients=[
            {
                "client_id": 0,
                "dataset": {
                    "name": "SAVEE",
                    "root": str(tmp_path),
                    "feature_names": ["mfcc"],
                    "test_size": 0.5,
                },
                "model": {"lr": 0.001, "optimizer": "adam"},
                "runtime": {
                    "local_steps": 1,
                    "batch_size": 2,
                    "seed": 7,
                    "device": "cpu",
                },
            }
        ],
        split_server=None,
        fed_server=None,
        channels={},
    )


def test_centralized_shapes_reject_incompatible_dataset_views():
    first = TensorDataset(torch.zeros(2, 3, 64), torch.zeros(2))
    second = TensorDataset(torch.zeros(2, 4, 64), torch.zeros(2))

    try:
        _validate_centralized_shapes([first, second])
    except ValueError as error:
        assert "identical feature shapes" in str(error)
    else:
        raise AssertionError("Expected incompatible shapes to be rejected")


def test_centralized_wait_drains_state_before_join():
    trainer = CentralizedTrainer.__new__(CentralizedTrainer)
    trainer._process = _CompletedProcess()
    trainer._last_state_dict = None
    trainer._result_queue = queue.Queue()
    trainer._result_queue.put(
        serialize_state_dict({"weight": torch.tensor([1.0])})
    )

    trainer.wait(poll_timeout=0)

    assert torch.equal(trainer._last_state_dict["weight"], torch.tensor([1.0]))


def test_spawned_centralized_complete_model_cycle(tmp_path):
    context = mp.get_context("spawn")
    stop_event = context.Event()
    result_queue = context.Queue(maxsize=1)
    config = _centralized_config(tmp_path)
    datasets = [
        (
            TensorDataset(
                torch.randn(4, 3, 64), torch.tensor([0.0, 1.0, 0.0, 1.0])
            ),
            TensorDataset(torch.randn(2, 3, 64), torch.tensor([0.0, 1.0])),
        )
    ]
    process = context.Process(
        target=_centralized_training_worker,
        args=(config, stop_event, result_queue, datasets),
        name="CentralizedSmoke",
    )

    try:
        process.start()
        payload = result_queue.get(timeout=30)
        process.join(timeout=10)
    finally:
        stop_event.set()
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)

    state = deserialize_state_dict(payload)
    assert not process.is_alive()
    assert process.exitcode == 0
    assert any(key.startswith("client_side_model.") for key in state)
    assert any(key.startswith("server_side_model.") for key in state)
