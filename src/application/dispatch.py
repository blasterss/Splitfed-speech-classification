"""Dispatch a validated configuration to its owning experiment runner."""

from ..experiments.checkpoint_evaluation import evaluate_full_model_checkpoint
from ..experiments.local_cross_corpus import run_local_cross_corpus
from ..schema import ConfigSchema, TrainingMode
from ..splitfed.controller import TrainingController
from ..utils.persistence import ArtifactPaths
from .lifecycle import execute_training


def execute_configured_experiment(
    config: ConfigSchema,
    *,
    configuration_provenance: dict | None = None,
) -> None:
    """Execute the runner selected explicitly by ``training.mode``."""
    if config.experiment.analysis_only:
        raise ValueError(
            "This configuration is analysis-only and cannot be dispatched "
            "to the training runtime"
        )
    if config.training.mode is TrainingMode.local:
        run_local_cross_corpus(config)
        return

    execute_training(
        TrainingController(config=config),
        config,
        configuration_provenance=configuration_provenance,
    )
    if config.experiment.cross_corpus_evaluation:
        _evaluate_complete_model_checkpoint(config)


def _evaluate_complete_model_checkpoint(config: ConfigSchema) -> None:
    """Evaluate the final complete model using the shared E1 contract."""
    checkpoint_names = {
        TrainingMode.centralized: "centralized_model.pt",
        TrainingMode.federated: "global_client_model.pt",
    }
    checkpoint_name = checkpoint_names.get(config.training.mode)
    if checkpoint_name is None or config.models_save_path is None:
        raise ValueError(
            "Cross-corpus checkpoint evaluation requires an artifact-backed "
            "complete-model mode"
        )
    paths = ArtifactPaths.from_root(
        config.models_save_path,
        config.experiment.name,
    )
    evaluate_full_model_checkpoint(
        config,
        paths.checkpoints / checkpoint_name,
        expected_mode=config.training.mode.value,
        artifact_root=paths.metrics / "cross_corpus",
    )
