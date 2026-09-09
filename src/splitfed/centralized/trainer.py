import queue
from pathlib import Path

import torch.multiprocessing as mp

from ...logger import logger
from ...schema import ConfigSchema
from ...utils.persistence import deserialize_state_dict, save_checkpoint
from .worker import _centralized_training_worker


class CentralizedTrainer:
    """Own the single complete model used by the centralized baseline."""

    def __init__(
        self,
        config: ConfigSchema,
        stop_event=None,
        mp_context=None,
        dataset_report_queue=None,
        resource_metrics_queue=None,
    ):
        self.config = config
        self._mp_context = mp_context or mp.get_context("spawn")
        self._stop_event = (
            stop_event if stop_event is not None else self._mp_context.Event()
        )
        self._result_queue: mp.Queue = self._mp_context.Queue(maxsize=1)
        self._process: mp.Process | None = None
        self._last_exitcode: int | None = None
        self._last_state_dict: dict | None = None
        self._dataset_report_queue = dataset_report_queue
        self._resource_metrics_queue = resource_metrics_queue

    def start(self) -> None:
        self._stop_event.clear()
        self._process = self._mp_context.Process(
            target=_centralized_training_worker,
            args=(
                self.config,
                self._stop_event,
                self._result_queue,
                None,
                self._dataset_report_queue,
                self._resource_metrics_queue,
            ),
            daemon=False,
            name="CentralizedTrainer",
        )
        self._last_exitcode = None
        self._process.start()
        logger.info(
            "Centralized trainer process started (pid=%d)", self._process.pid
        )

    def stop(self) -> None:
        if self._process is not None:
            self._stop_event.set()
            self._process.join(timeout=30)
            if self._process.is_alive():
                self._stop_event.set()
                self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=5)
                if self._process.is_alive():
                    raise RuntimeError(
                        "Centralized trainer remained alive after kill"
                    )
            self._last_exitcode = self._process.exitcode
            self._process = None
        if self._last_state_dict is None and self._last_exitcode == 0:
            try:
                payload = self._result_queue.get(timeout=5)
                self._last_state_dict = deserialize_state_dict(payload)
            except queue.Empty:
                pass

    def wait(self, poll_timeout: float = 0.5) -> None:
        """Drain the result before join so large states cannot block exit."""
        if self._process is None:
            raise RuntimeError("Centralized trainer has not been started")
        while self._last_state_dict is None:
            try:
                payload = self._result_queue.get(timeout=poll_timeout)
                self._last_state_dict = deserialize_state_dict(payload)
            except queue.Empty:
                if not self._process.is_alive():
                    self._process.join(timeout=1)
                    raise RuntimeError(
                        "Centralized trainer failed before producing state "
                        f"(exitcode={self._process.exitcode})"
                    ) from None
        self._process.join(timeout=30)
        if self._process.is_alive():
            raise RuntimeError(
                "Centralized trainer did not exit after publishing state"
            )
        if self._process.exitcode != 0:
            raise RuntimeError(
                "Centralized trainer process failure: "
                f"exitcode={self._process.exitcode}"
            )

    @property
    def exitcode(self) -> int | None:
        if self._process is not None:
            return self._process.exitcode
        return self._last_exitcode

    def get_state_dict(self) -> dict:
        if self._last_state_dict is not None:
            return self._last_state_dict
        try:
            return deserialize_state_dict(self._result_queue.get_nowait())
        except queue.Empty:
            raise RuntimeError(
                "No centralized state_dict is available after training"
            ) from None

    def save(self, path: str | Path) -> None:
        save_path = Path(path) / "centralized_model.pt"
        save_checkpoint(
            save_path,
            mode="centralized",
            model_state_dict=self.get_state_dict(),
        )
        logger.info("Centralized model saved to '%s'", save_path)
