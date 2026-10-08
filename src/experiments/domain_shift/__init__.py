"""Corpus domain-shift analysis, separate from model training."""

from .analysis import discrepancy, run_domain_shift

__all__ = ["discrepancy", "run_domain_shift"]
