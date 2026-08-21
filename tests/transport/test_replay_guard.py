import pytest

from src.transport.replay import ReplayGuard


def test_replay_guard_rejects_duplicate_request_id():
    guard = ReplayGuard(capacity=2)
    guard.accept("request-a")
    guard.accept("request-b")

    with pytest.raises(ValueError, match="Replay"):
        guard.accept("request-a")


def test_replay_guard_evicts_oldest_id_at_capacity():
    guard = ReplayGuard(capacity=2)
    guard.accept("request-a")
    guard.accept("request-b")
    guard.accept("request-c")

    guard.accept("request-a")


def test_replay_guard_requires_positive_capacity():
    with pytest.raises(ValueError, match="capacity"):
        ReplayGuard(capacity=0)
