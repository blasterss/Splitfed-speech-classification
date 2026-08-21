import os
import signal

import torch.multiprocessing as mp

from src.utils.runtime import ignore_parent_interrupts


def _interrupt_immune_worker(ready, stop) -> None:
    ignore_parent_interrupts()
    ready.set()
    stop.wait(timeout=10)


def test_spawn_worker_leaves_sigint_to_supervisor():
    context = mp.get_context("spawn")
    ready = context.Event()
    stop = context.Event()
    process = context.Process(
        target=_interrupt_immune_worker,
        args=(ready, stop),
        name="InterruptImmuneWorker",
    )
    process.start()

    try:
        assert ready.wait(timeout=10)
        os.kill(process.pid, signal.SIGINT)
        process.join(timeout=0.2)
        assert process.is_alive()

        stop.set()
        process.join(timeout=10)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=10)
