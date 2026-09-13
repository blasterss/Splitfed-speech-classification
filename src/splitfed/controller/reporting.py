"""Controller-owned queue cleanup and diagnostic report collection."""

import queue
import time

from ...utils.runtime import FailureRecord


def close_process_queue(process_queue) -> None:
    """Close a process queue and join its feeder thread when present."""
    if process_queue is None:
        return
    close = getattr(process_queue, "close", None)
    if close is not None:
        close()
    join_thread = getattr(process_queue, "join_thread", None)
    if join_thread is not None:
        join_thread()


def drain_dataset_reports(report_queue, manifests: dict[int, dict]) -> None:
    """Drain validated, unique per-client dataset reports into manifests."""
    if report_queue is None:
        return
    while True:
        try:
            report = report_queue.get_nowait()
        except queue.Empty:
            break
        client_id = report.get("client_id")
        if not isinstance(client_id, int):
            raise ValueError("Dataset report has invalid client_id")
        if client_id in manifests:
            raise ValueError(
                f"Duplicate dataset report for client {client_id}"
            )
        manifests[client_id] = report


def collect_dataset_reports(
    report_queue,
    manifests: dict[int, dict],
    *,
    expected_client_ids: set[int],
    timeout: float,
) -> None:
    """Wait for the exact client manifest set before planning starts."""
    deadline = time.monotonic() + timeout
    while set(manifests) != expected_client_ids:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            missing = sorted(expected_client_ids - set(manifests))
            raise TimeoutError(f"missing dataset reports from {missing}")
        try:
            report = report_queue.get(timeout=remaining)
        except queue.Empty as exc:
            missing = sorted(expected_client_ids - set(manifests))
            raise TimeoutError(
                f"missing dataset reports from {missing}"
            ) from exc
        client_id = report.get("client_id")
        if client_id not in expected_client_ids:
            raise ValueError("Dataset report has unknown client_id")
        if client_id in manifests:
            raise ValueError(
                f"Duplicate dataset report for client {client_id}"
            )
        manifests[client_id] = report


def capture_first_failure(
    current_failure: dict | None,
    failure_queue,
    fallback: BaseException,
) -> dict:
    """Prefer worker context, otherwise build a controller failure record."""
    if current_failure is not None:
        return current_failure
    if failure_queue is not None:
        try:
            return failure_queue.get_nowait()
        except queue.Empty:
            pass
    return FailureRecord.from_exception(
        component="controller", exception=fallback
    ).as_dict()
