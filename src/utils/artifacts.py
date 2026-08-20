from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactPaths:
    """Owned directories for one experiment run."""

    root: Path
    metadata: Path
    checkpoints: Path
    metrics: Path
    diagnostics: Path

    @classmethod
    def from_root(cls, artifact_root: str | Path, experiment_name: str):
        root = Path(artifact_root) / experiment_name
        return cls(
            root=root,
            metadata=root / "metadata",
            checkpoints=root / "checkpoints",
            metrics=root / "metrics",
            diagnostics=root / "diagnostics",
        )

    def mkdir(self) -> None:
        for path in (
            self.metadata,
            self.checkpoints,
            self.metrics,
            self.diagnostics,
        ):
            path.mkdir(parents=True, exist_ok=True)
