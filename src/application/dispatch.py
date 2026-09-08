"""Dispatch a validated configuration to its owning experiment runner."""

from ..experiments.local_cross_corpus import run_local_cross_corpus
from ..schema import ConfigSchema, TrainingMode
from ..splitfed.controller import TrainingController
from .lifecycle import execute_training


def execute_configured_experiment(
    config: ConfigSchema,
    *,
    configuration_provenance: dict | None = None,
) -> None:
    """Execute the runner selected explicitly by ``training.mode``."""
    if config.training.mode is TrainingMode.local:
        run_local_cross_corpus(config)
        return

    execute_training(
        TrainingController(config=config),
        config,
        configuration_provenance=configuration_provenance,
    )
