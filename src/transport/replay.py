from collections import deque

DEFAULT_REPLAY_CAPACITY = 10_000


class ReplayGuard:
    """Bounded FIFO cache of request IDs already accepted by a worker."""

    def __init__(self, capacity: int = DEFAULT_REPLAY_CAPACITY):
        if capacity <= 0:
            raise ValueError("ReplayGuard capacity must be positive")
        self._capacity = capacity
        self._order: deque[str] = deque()
        self._seen: set[str] = set()

    def accept(self, request_id: str) -> None:
        """Remember a new request ID or reject an ID still in the cache."""
        if request_id in self._seen:
            raise ValueError(f"Replay detected for request_id {request_id!r}")
        if len(self._order) == self._capacity:
            self._seen.remove(self._order.popleft())
        self._order.append(request_id)
        self._seen.add(request_id)
