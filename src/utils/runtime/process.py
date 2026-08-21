import signal


def ignore_parent_interrupts() -> None:
    """Leave terminal interrupt handling to the supervising process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
