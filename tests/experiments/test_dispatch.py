from types import SimpleNamespace

import pytest

from src.application.dispatch import execute_configured_experiment
from src.schema import TrainingMode


def _config(tmp_path, *, mode, analysis_only=False, evaluate=False):
    return SimpleNamespace(
        experiment=SimpleNamespace(
            name="e1",
            analysis_only=analysis_only,
            cross_corpus_evaluation=evaluate,
        ),
        training=SimpleNamespace(mode=mode),
        models_save_path=str(tmp_path),
    )


def test_analysis_only_config_cannot_start_training(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(
        "src.application.dispatch.execute_training",
        lambda *args, **kwargs: called.append("training"),
    )

    with pytest.raises(ValueError, match="analysis-only"):
        execute_configured_experiment(
            _config(
                tmp_path,
                mode=TrainingMode.centralized,
                analysis_only=True,
            )
        )

    assert called == []


def test_centralized_e1_evaluates_saved_checkpoint(tmp_path, monkeypatch):
    calls = []
    config = _config(
        tmp_path,
        mode=TrainingMode.centralized,
        evaluate=True,
    )
    monkeypatch.setattr(
        "src.application.dispatch.TrainingController", lambda config: object()
    )
    monkeypatch.setattr(
        "src.application.dispatch.execute_training",
        lambda *args, **kwargs: calls.append("training"),
    )
    monkeypatch.setattr(
        "src.application.dispatch.evaluate_full_model_checkpoint",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    execute_configured_experiment(config)

    assert calls[0] == "training"
    args, kwargs = calls[1]
    assert args[0] is config
    assert args[1] == (
        tmp_path / "e1" / "checkpoints" / "centralized_model.pt"
    )
    assert kwargs["expected_mode"] == "centralized"
    assert kwargs["artifact_root"] == (
        tmp_path / "e1" / "metrics" / "cross_corpus"
    )
