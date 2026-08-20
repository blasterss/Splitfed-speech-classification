"""Process supervision and cancellation helpers for training runtimes."""

import torch.multiprocessing as mp

from ...logger import get_logger

logger = get_logger(__name__)


def _raise_for_failed_processes(processes: list[mp.Process]) -> None:
    failures = [
        f"{process.name} (exitcode={process.exitcode})"
        for process in processes
        if process.exitcode not in (None, 0)
    ]
    if failures:
        message = "Client process failure: " + ", ".join(failures)
        logger.error(message)
        raise RuntimeError(message)


def _raise_for_failed_servers(servers: tuple[object, ...]) -> None:
    failures = [
        f"{type(server).__name__} (exitcode={server.exitcode})"
        for server in servers
        if server.exitcode not in (None, 0)
    ]
    if failures:
        message = "Server process failure: " + ", ".join(failures)
        logger.error(message)
        raise RuntimeError(message)


def _wait_for_training_processes(
    processes: list[mp.Process],
    servers: tuple[object, ...],
    poll_timeout: float,
) -> None:
    remaining = list(processes)
    while remaining:
        _raise_for_failed_servers(servers)
        for process in remaining[:]:
            process.join(timeout=poll_timeout)
            if not process.is_alive():
                remaining.remove(process)
        _raise_for_failed_processes(
            [process for process in processes if not process.is_alive()]
        )
    _raise_for_failed_servers(servers)


def _cancel_training(stop_event, barriers: tuple[object, ...]) -> None:
    stop_event.set()
    for barrier in barriers:
        try:
            barrier.abort()
        except Exception as exc:
            logger.debug("Could not abort training barrier: %s", exc)


def _shutdown_processes(
    processes: list[mp.Process], join_timeout: float
) -> None:
    for process in processes:
        process.join(timeout=join_timeout)
    alive_processes = [process for process in processes if process.is_alive()]
    for process in alive_processes:
        process.terminate()
    for process in alive_processes:
        process.join(timeout=join_timeout)
    killed_processes = []
    for process in alive_processes:
        if process.is_alive():
            process.kill()
            killed_processes.append(process)
    for process in killed_processes:
        process.join(timeout=join_timeout)
    unreaped = [
        process.name for process in killed_processes if process.is_alive()
    ]
    if unreaped:
        raise RuntimeError(
            "Processes remained alive after kill: " + ", ".join(unreaped)
        )
