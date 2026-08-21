"""Process signal policy and structured worker failure reporting."""

from .failures import FailureRecord, publish_failure
from .process import ignore_parent_interrupts

__all__ = [
    "FailureRecord",
    "ignore_parent_interrupts",
    "publish_failure",
]
