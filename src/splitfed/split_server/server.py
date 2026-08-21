import queue
from pathlib import Path

import torch.multiprocessing as mp

from ...logger import logger
from ...schema import ServerModelScope, SplitServerConfig
from ...transport.base import Channel
from ...utils.persistence import deserialize_state_dict, save_checkpoint
from .worker import (
    _split_server_worker_batch,
    _split_server_worker_personalized,
)

logger = logger.getChild("SplitServer")


class SplitServer:
    """
    Split-learning server.

    Runs in its own process. Receives activations from clients,
    completes the forward pass, computes loss / gradients, and
    sends gradients back (train) or logits back (eval).

    Channel layout (one pair per client):
        uplink   – server reads  (client writes activations)
        downlink – server writes (client reads  gradients / logits)

    After stop(), call get_state_dict() to retrieve the trained weights,
    or save() to write them directly to disk.
    """

    def __init__(
        self,
        config: SplitServerConfig,
        client_channels: dict[str, dict[str, Channel]],
        stop_event=None,
        mp_context=None,
        failure_queue=None,
    ):
        self.config = config
        self.client_channels = client_channels

        self._mp_context = mp_context or mp.get_context("spawn")
        self._stop_event = (
            stop_event if stop_event is not None else self._mp_context.Event()
        )
        self._result_queue: mp.Queue = self._mp_context.Queue(maxsize=1)
        self._process: mp.Process | None = None
        self._last_exitcode: int | None = None
        self._last_state_dict: dict | None = None
        self._failure_queue = failure_queue

    def start(self) -> None:
        """Spawn the server worker process."""
        self._stop_event.clear()
        worker = (
            _split_server_worker_personalized
            if self.config.model_scope is ServerModelScope.personalized
            else _split_server_worker_batch
        )
        self._process = self._mp_context.Process(
            target=worker,
            args=(
                self.config,
                self.client_channels,
                self._stop_event,
                self._result_queue,
                self._failure_queue,
            ),
            daemon=True,
            name="SplitServer",
        )
        self._last_exitcode = None
        self._process.start()
        logger.info("SplitServer process started (pid=%d)", self._process.pid)

    def stop(self) -> None:
        """Signal the worker to finish and wait for it to exit."""
        self._stop_event.set()
        if self._process is not None:
            try:
                payload = self._result_queue.get(timeout=30)
                self._last_state_dict = deserialize_state_dict(payload)
            except queue.Empty:
                logger.warning("SplitServer produced no final state_dict")
            self._process.join(timeout=30)
            if self._process.is_alive():
                logger.warning(
                    "SplitServer worker did not exit within 30 s — "
                    "terminating."
                )
                self._process.terminate()
                self._process.join(timeout=5)
                if self._process.is_alive():
                    self._process.kill()
                    self._process.join(timeout=5)
                    if self._process.is_alive():
                        raise RuntimeError(
                            "SplitServer worker remained alive after kill"
                        )

            self._last_exitcode = self._process.exitcode

            self._process = None
        logger.info("SplitServer process stopped")

    @property
    def exitcode(self) -> int | None:
        if self._process is not None:
            return self._process.exitcode
        return self._last_exitcode

    def get_state_dict(self) -> dict:
        """
        Return the trained server-side model weights.
        Must be called *after* stop().
        """
        try:
            if self._last_state_dict is not None:
                return self._last_state_dict
            return deserialize_state_dict(self._result_queue.get_nowait())
        except queue.Empty:
            raise RuntimeError(
                "No state_dict available. Either stop() has not been called "
                "yet "
                "or the worker exited abnormally."
            ) from None

    def save(self, path: str) -> None:
        """Retrieve the trained weights and save them to disk."""
        state_dict = self.get_state_dict()
        if self.config.model_scope is ServerModelScope.personalized:
            for client_id, client_state in state_dict.items():
                save_path = Path(path) / f"split_server_client_{client_id}.pt"
                save_checkpoint(
                    save_path,
                    mode=getattr(self, "training_mode", "split"),
                    server_model_scope="personalized",
                    client_id=client_id,
                    model_state_dict=client_state,
                )
                logger.info(
                    "Personalized SplitServer model saved to '%s'", save_path
                )
        else:
            save_path = Path(path) / "split_server.pt"
            save_checkpoint(
                save_path,
                mode=getattr(self, "training_mode", "splitfed"),
                server_model_scope="shared",
                model_state_dict=state_dict,
            )
            logger.info("SplitServer model saved to '%s'", save_path)
