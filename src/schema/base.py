"""Base configuration model policy."""

from pydantic import BaseModel, ConfigDict


class StrictConfigModel(BaseModel):
    """Reject unknown configuration fields across every schema block."""

    model_config = ConfigDict(extra="forbid")
