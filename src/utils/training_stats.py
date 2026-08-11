from dataclasses import dataclass, field

from ..logger import logger

logger = logger.getChild("TrainingStats")


@dataclass
class _RoundStats:
    losses: list = field(default_factory=list)
    n_batches: int = 0

    def update(self, loss: float) -> None:
        self.losses.append(loss)
        self.n_batches += 1

    def log_and_reset(self, round_idx: int) -> None:
        if not self.losses:
            return
        avg = sum(self.losses) / len(self.losses)
        mn = min(self.losses)
        mx = max(self.losses)
        logger.info(
            "Round %d | loss avg=%.6f  min=%.6f  max=%.6f  batches=%d",
            round_idx,
            avg,
            mn,
            mx,
            self.n_batches,
        )
        self.losses.clear()
        self.n_batches = 0
