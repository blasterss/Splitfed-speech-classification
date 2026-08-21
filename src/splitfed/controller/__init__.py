"""Training controller component."""

import torch.multiprocessing as mp

from .controller import TrainingController

__all__ = ["TrainingController", "mp"]
