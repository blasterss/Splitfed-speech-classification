from types import SimpleNamespace

import pytest

from src.main import _execute_training


class FakeController:
    def __init__(self, *, setup_error=None, training_error=None):
        self.setup_error = setup_error
        self.training_error = training_error
        self.calls = []
        self.split_server = None
        self.fed_server = None
        self.centralized_trainer = None

    def setup(self):
        self.calls.append("setup")
        if self.setup_error is not None:
            raise self.setup_error

    def start_training(self):
        self.calls.append("start_training")
        if self.training_error is not None:
            raise self.training_error

    def teardown(self):
        self.calls.append("teardown")


def _config_without_artifacts():
    return SimpleNamespace(models_save_path=None)


def test_execute_training_tears_down_after_success():
    controller = FakeController()

    _execute_training(controller, _config_without_artifacts())

    assert controller.calls == ["setup", "start_training", "teardown"]


@pytest.mark.parametrize("failure_stage", ["setup", "training"])
def test_execute_training_tears_down_after_failure(failure_stage):
    error = RuntimeError(f"{failure_stage} failed")
    controller = FakeController(
        setup_error=error if failure_stage == "setup" else None,
        training_error=error if failure_stage == "training" else None,
    )

    with pytest.raises(RuntimeError, match=f"{failure_stage} failed"):
        _execute_training(controller, _config_without_artifacts())

    assert controller.calls[-1] == "teardown"


class FakeServer:
    def __init__(self, calls, name, error=None):
        self.calls = calls
        self.name = name
        self.error = error

    def stop(self):
        self.calls.append(f"stop-{self.name}")
        if self.error is not None:
            raise self.error


def test_execute_training_attempts_all_server_stops_before_teardown():
    controller = FakeController()
    controller.split_server = FakeServer(
        controller.calls, "split", RuntimeError("split stop failed")
    )
    controller.fed_server = FakeServer(controller.calls, "fed")

    with pytest.raises(RuntimeError, match="split stop failed"):
        _execute_training(controller, _config_without_artifacts())

    assert controller.calls == [
        "setup",
        "start_training",
        "stop-split",
        "stop-fed",
        "teardown",
    ]
