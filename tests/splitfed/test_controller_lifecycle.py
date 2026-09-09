import copy
import queue
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from src.schema import ConfigSchema, TrainingMode
from src.splitfed.common.lifecycle import (
    _cancel_training,
    _raise_for_failed_processes,
    _raise_for_failed_servers,
    _shutdown_processes,
    _wait_for_training_processes,
)
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
    assert controller._stop_event is controller.split_server._stop_event
    assert controller._stop_event is controller.fed_server._stop_event

    controller.teardown()

    assert controller._manager is None


def test_config_rejects_duplicate_client_ids_before_setup(tmp_path):
    raw = make_config(tmp_path).model_dump(by_alias=True)
    raw["clients"].append(copy.deepcopy(raw["clients"][0]))

    with pytest.raises(ValidationError, match="Client IDs must be unique"):
        ConfigSchema(**raw)


def test_config_rejects_quorum_larger_than_client_count(tmp_path):
    raw = make_config(tmp_path).model_dump(by_alias=True)
    raw["fed_server"]["min_clients"] = 2

    with pytest.raises(ValidationError, match="min_clients"):
        ConfigSchema(**raw)


def test_config_rejects_missing_server_channel_reference(tmp_path):
    raw = make_config(tmp_path).model_dump(by_alias=True)
    raw["channels"].pop("split_uplink")

    with pytest.raises(ValidationError, match="split_uplink"):
        ConfigSchema(**raw)


def test_config_rejects_swapped_split_channel_roles(tmp_path):
    raw = make_config(tmp_path).model_dump(by_alias=True)
    raw["split_server"]["split_uplink_channel"] = "split_downlink"
    raw["split_server"]["split_downlink_channel"] = "split_uplink"

    with pytest.raises(ValidationError, match="canonical"):
        ConfigSchema(**raw)


def test_config_rejects_swapped_federated_channel_roles(tmp_path):
    raw = make_config(tmp_path).model_dump(by_alias=True)
    raw["fed_server"]["federated_uplink_channel"] = "federated_downlink"
    raw["fed_server"]["federated_downlink_channel"] = "federated_uplink"

    with pytest.raises(ValidationError, match="canonical"):
        ConfigSchema(**raw)


def test_config_rejects_grpc_stub_before_setup(tmp_path):
    raw = make_config(tmp_path).model_dump(by_alias=True)
    raw["experiment"]["transport"] = "grpc"

    with pytest.raises(ValidationError, match="grpc is a stub"):
        ConfigSchema(**raw)


def test_config_rejects_ignored_queue_compression(tmp_path):
    raw = make_config(tmp_path).model_dump(by_alias=True)
    raw["channels"]["split_uplink"]["compression"] = "gzip"

    with pytest.raises(ValidationError, match="compression"):
        ConfigSchema(**raw)


@pytest.mark.parametrize(
    ("path", "unknown_key"),
    [
        ((), "unknown_root_option"),
        (("training",), "fed_evey"),
        (("clients", 0, "runtime"), "batch_sze"),
    ],
)
def test_config_rejects_unknown_fields(tmp_path, path, unknown_key):
    raw = make_config(tmp_path).model_dump(by_alias=True)
    target = raw
    for part in path:
        target = target[part]
    target[unknown_key] = 1

    with pytest.raises(ValidationError, match=unknown_key):
        ConfigSchema(**raw)


def test_federated_mode_rejects_split_server_and_split_channels(tmp_path):
    raw = make_config(tmp_path).model_dump(by_alias=True)
    raw["training"]["mode"] = "federated"

    with pytest.raises(ValidationError, match="split_server"):
        ConfigSchema(**raw)


def test_federated_cadence_fields_must_match(tmp_path):
    raw = make_config(tmp_path).model_dump(by_alias=True)
    raw["training"]["fed_every"] = 2
    raw["fed_server"]["aggregation_freq"] = 1

    with pytest.raises(ValidationError, match="aggregation_freq"):
        ConfigSchema(**raw)


def test_federated_mode_accepts_only_federated_topology(tmp_path):
    raw = make_config(tmp_path).model_dump(by_alias=True)
    raw["training"]["mode"] = "federated"
    raw["split_server"] = None
    raw["channels"].pop("split_uplink")
    raw["channels"].pop("split_downlink")

    config = ConfigSchema(**raw)

    assert config.split_server is None


def test_federated_setup_creates_only_federated_channels_and_server(tmp_path):
    raw = make_config(tmp_path).model_dump(by_alias=True)
    raw["training"]["mode"] = "federated"
    raw["split_server"] = None
    raw["channels"].pop("split_uplink")
    raw["channels"].pop("split_downlink")
    controller = TrainingController(ConfigSchema(**raw))

    controller.setup()

    assert set(controller.channels[0]) == {
        "federated_uplink",
        "federated_downlink",
    }
    assert controller.split_server is None
    assert controller.fed_server is not None
    controller.teardown()


def test_centralized_setup_creates_no_channels_or_servers(tmp_path):
    raw = make_config(tmp_path).model_dump(by_alias=True)
    raw["training"]["mode"] = "centralized"
    raw["split_server"] = None
    raw["fed_server"] = None
    raw["channels"] = {}
    controller = TrainingController(ConfigSchema(**raw))

    controller.setup()

    assert controller.channels == {0: {}}
    assert controller.split_server is None
    assert controller.fed_server is None
    assert controller.centralized_trainer is not None
    controller.teardown()


def test_centralized_rejects_inconsistent_training_views(tmp_path):
    raw = make_config(tmp_path).model_dump(by_alias=True)
    raw["training"]["mode"] = "centralized"
    raw["split_server"] = None
    raw["fed_server"] = None
    raw["channels"] = {}
    second = copy.deepcopy(raw["clients"][0])
    second["client_id"] = 1
    second["runtime"]["batch_size"] = 2
    raw["clients"].append(second)

    with pytest.raises(ValidationError, match="centralized dataset views"):
        ConfigSchema(**raw)


def test_split_personalized_rejects_federated_server_and_channels(tmp_path):
    raw = make_config(tmp_path).model_dump(by_alias=True)
    raw["training"]["mode"] = "split"
    raw["split_server"]["model_scope"] = "personalized"
    raw["fed_server"] = None
    raw["channels"].pop("federated_uplink")
    raw["channels"].pop("federated_downlink")

    config = ConfigSchema(**raw)

    assert config.split_server.model_scope.value == "personalized"
    assert config.fed_server is None


def test_splitfed_accepts_personalized_server_scope(tmp_path):
    raw = make_config(tmp_path).model_dump(by_alias=True)
    raw["split_server"]["model_scope"] = "personalized"

    config = ConfigSchema(**raw)

    assert config.training.mode is TrainingMode.splitfed
    assert config.split_server.model_scope.value == "personalized"


class FakeProcess:
    def __init__(self, name, exitcode):
        self.name = name
        self.exitcode = exitcode


class PollProcess(FakeProcess):
    def __init__(self, name, exitcode=0, alive=True):
        super().__init__(name, exitcode)
        self.alive = alive
        self.terminate_called = False
        self.kill_called = False

    def join(self, timeout=None):
        self.alive = False

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.terminate_called = True
        self.alive = False

    def kill(self):
        self.kill_called = True
        self.alive = False


class StubbornProcess(PollProcess):
    def join(self, timeout=None):
        pass


class ReapedOnlyAfterKillProcess(PollProcess):
    def __init__(self, name):
        super().__init__(name, alive=True)
        self.join_calls = 0

    def join(self, timeout=None):
        self.join_calls += 1
        if self.kill_called:
            self.alive = False

    def terminate(self):
        self.terminate_called = True

    def kill(self):
        self.kill_called = True


class FakeServer:
    def __init__(self, exitcode):
        self.exitcode = exitcode


class LifecycleServer(FakeServer):
    def __init__(self):
        super().__init__(None)
        self.start_called = False
        self.stop_called = False

    def start(self):
        self.start_called = True

    def stop(self):
        self.stop_called = True


class FakeStopEvent:
    def __init__(self):
        self.set_called = False

    def set(self):
        self.set_called = True


class FakeBarrier:
    def __init__(self, error=None):
        self.abort_called = False
        self.error = error

    def abort(self):
        self.abort_called = True
        if self.error is not None:
            raise self.error


class FakeManager:
    def __init__(self, barriers):
        self.barriers = iter(barriers)

    def Barrier(self, parties):
        return next(self.barriers)


class FailingStartProcess:
    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        raise RuntimeError("spawn failed")


def test_failed_client_process_is_propagated():
    processes = [FakeProcess("Client-0", 0), FakeProcess("Client-1", 1)]

    try:
        _raise_for_failed_processes(processes)
    except RuntimeError as error:
        assert str(error) == "Client process failure: Client-1 (exitcode=1)"
    else:
        raise AssertionError("Expected failed client process to be propagated")


def test_successful_client_processes_do_not_raise():
    _raise_for_failed_processes([FakeProcess("Client-0", 0)])


def test_training_process_wait_checks_server_exitcodes():
    _wait_for_training_processes(
        [PollProcess("Client-0")], (FakeServer(0),), poll_timeout=0
    )


def test_failed_server_process_is_propagated():
    with pytest.raises(
        RuntimeError,
        match=r"Server process failure: FakeServer \(exitcode=1\)",
    ):
        _raise_for_failed_servers((FakeServer(1),))


def test_shutdown_processes_terminates_alive_processes():
    process = StubbornProcess("Client-0", alive=True)

    _shutdown_processes([process], join_timeout=0)

    assert process.terminate_called
    assert not process.is_alive()


def test_shutdown_processes_joins_after_killing_stubborn_process():
    process = ReapedOnlyAfterKillProcess("Client-0")

    _shutdown_processes([process], join_timeout=0)

    assert process.terminate_called
    assert process.kill_called
    assert process.join_calls == 3
    assert not process.is_alive()


def test_cancel_training_sets_stop_event_and_aborts_all_barriers():
    stop_event = FakeStopEvent()
    broken_barrier = FakeBarrier(RuntimeError("already broken"))
    waiting_barrier = FakeBarrier()

    _cancel_training(stop_event, (broken_barrier, waiting_barrier))

    assert stop_event.set_called
    assert broken_barrier.abort_called
    assert waiting_barrier.abort_called


def test_controller_drains_bounded_dataset_reports_by_client_id():
    controller = TrainingController.__new__(TrainingController)
    controller._dataset_report_queue = queue.Queue()
    controller.dataset_manifests = {}
    controller._dataset_report_queue.put({"client_id": 2, "dataset": "SAVEE"})

    controller._drain_dataset_reports()

    assert controller.dataset_manifests == {
        2: {"client_id": 2, "dataset": "SAVEE"}
    }


def test_controller_keeps_worker_first_failure_over_fallback():
    controller = TrainingController.__new__(TrainingController)
    controller.first_failure = None
    controller._failure_queue = queue.Queue(maxsize=1)
    controller._failure_queue.put(
        {
            "schema_version": 1,
            "component": "client",
            "message": "dataset load failed",
        }
    )

    controller._capture_first_failure(RuntimeError("exitcode=1"))
    controller._capture_first_failure(RuntimeError("later failure"))

    assert controller.first_failure == {
        "schema_version": 1,
        "component": "client",
        "message": "dataset load failed",
    }


def test_start_training_cancels_barriers_when_client_spawn_fails(monkeypatch):
    stop_event = FakeStopEvent()
    ready_barrier = FakeBarrier()
    eval_barrier = FakeBarrier()
    split_server = LifecycleServer()
    fed_server = LifecycleServer()
    controller = TrainingController.__new__(TrainingController)
    controller.cfg = SimpleNamespace(
        training=SimpleNamespace(mode=TrainingMode.splitfed)
    )
    controller.client_cfgs = [SimpleNamespace(client_id=0)]
    controller.split_server = split_server
    controller.fed_server = fed_server
    controller.channels = {
        0: {
            controller.SPLIT_UPLINK: object(),
            controller.SPLIT_DOWNLINK: object(),
            controller.FED_UPLINK: object(),
            controller.FED_DOWNLINK: object(),
        }
    }
    controller._client_processes = []
    controller._manager = FakeManager((ready_barrier, eval_barrier))
    controller._stop_event = stop_event
    controller._stop_events = {}
    monkeypatch.setattr(
        "src.splitfed.controller.mp.Process", FailingStartProcess
    )

    with pytest.raises(RuntimeError, match="spawn failed"):
        controller.start_training()

    assert stop_event.set_called
    assert ready_barrier.abort_called
    assert eval_barrier.abort_called
    assert split_server.stop_called
    assert fed_server.stop_called
