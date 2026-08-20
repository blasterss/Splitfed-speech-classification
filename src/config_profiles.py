"""Typed, versioned built-in experiment profile registry."""

from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentProfileDefinition:
    """One typed built-in profile and its configuration defaults."""

    name: str
    version: str
    config: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Profile name must not be empty")
        if not self.version:
            raise ValueError("Profile version must not be empty")
        object.__setattr__(self, "config", deepcopy(dict(self.config)))

    def config_copy(self) -> dict:
        """Return defaults safe for merge-time mutation."""
        return deepcopy(dict(self.config))


class ExperimentProfileRegistry(Mapping[str, ExperimentProfileDefinition]):
    """Validated registry that rejects ambiguous duplicate profile names."""

    def __init__(self, definitions: list[ExperimentProfileDefinition]):
        profiles: dict[str, ExperimentProfileDefinition] = {}
        for definition in definitions:
            if definition.name in profiles:
                raise ValueError(
                    f"Duplicate experiment profile {definition.name!r}"
                )
            profiles[definition.name] = definition
        self._profiles = profiles

    def __getitem__(self, name: str) -> ExperimentProfileDefinition:
        return self._profiles[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._profiles)

    def __len__(self) -> int:
        return len(self._profiles)


PROFILE_REGISTRY = ExperimentProfileRegistry(
    [
        ExperimentProfileDefinition(
            name="smoke",
            version="1",
            config={
                "training": {
                    "num_rounds": 1,
                    "eval_every": 1,
                    "barrier_timeout_sec": 30.0,
                }
            },
        )
    ]
)
