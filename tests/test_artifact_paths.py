from src.utils.artifacts import ArtifactPaths


def test_artifact_paths_are_scoped_by_experiment_name(tmp_path):
    paths = ArtifactPaths.from_root(tmp_path / "artifacts", "smoke-run")

    assert paths.root == tmp_path / "artifacts" / "smoke-run"
    assert paths.metadata == paths.root / "metadata"
    assert paths.checkpoints == paths.root / "checkpoints"
    assert paths.metrics == paths.root / "metrics"
    assert paths.diagnostics == paths.root / "diagnostics"


def test_artifact_paths_create_all_owned_directories(tmp_path):
    paths = ArtifactPaths.from_root(tmp_path / "artifacts", "smoke-run")

    paths.mkdir()

    assert paths.metadata.is_dir()
    assert paths.checkpoints.is_dir()
    assert paths.metrics.is_dir()
    assert paths.diagnostics.is_dir()
