"""Configuration layering and typed CLI override handling."""

import copy

import yaml

from ..config_profiles import PROFILE_REGISTRY


def apply_cli_overrides(raw_config: dict, overrides: list[str]) -> dict:
    """Apply typed, existing-path-only CLI overrides to a raw config."""
    resolved = copy.deepcopy(raw_config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(
                f"Invalid override {override!r}; expected PATH=VALUE"
            )
        raw_path, raw_value = override.split("=", 1)
        parts = raw_path.split(".")
        if not raw_path or any(not part for part in parts):
            raise ValueError(f"Invalid override path {raw_path!r}")

        target = resolved
        for part in parts[:-1]:
            target = _descend_override_path(target, part, raw_path)

        leaf = parts[-1]
        value = yaml.safe_load(raw_value)
        if isinstance(target, dict):
            if leaf not in target:
                raise ValueError(f"Unknown override path {raw_path!r}")
            target[leaf] = value
        elif isinstance(target, list):
            index = _override_list_index(leaf, raw_path)
            if index >= len(target):
                raise ValueError(f"Unknown override path {raw_path!r}")
            target[index] = value
        else:
            raise ValueError(f"Unknown override path {raw_path!r}")
    return resolved


def resolve_raw_config(
    raw_config: dict,
    *,
    profile_name: str | None,
    overrides: list[str],
) -> tuple[dict, dict]:
    """Resolve profile, YAML and CLI layers in increasing precedence."""
    yaml_profile = raw_config.get("experiment", {}).get("profile")
    selected_profile = profile_name or yaml_profile
    profile_provenance = None
    if selected_profile is None:
        merged = copy.deepcopy(raw_config)
    else:
        try:
            profile = PROFILE_REGISTRY[selected_profile]
        except KeyError as exc:
            raise ValueError(
                f"Unknown experiment profile {selected_profile!r}"
            ) from exc
        merged = _deep_merge(profile.config_copy(), raw_config)
        merged.setdefault("experiment", {})["profile"] = selected_profile
        profile_provenance = {
            "name": selected_profile,
            "version": profile.version,
            "source": "cli" if profile_name is not None else "yaml",
        }

    resolved = apply_cli_overrides(merged, overrides)
    return resolved, {
        "profile": profile_provenance,
        "cli_overrides": list(overrides),
        "overrides": [
            {
                "source": "cli",
                "path": override.split("=", 1)[0],
                "expression": override,
            }
            for override in overrides
        ],
    }


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge mappings while replacing non-mapping values."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _descend_override_path(target, part: str, raw_path: str):
    if isinstance(target, dict):
        if part not in target:
            raise ValueError(f"Unknown override path {raw_path!r}")
        return target[part]
    if isinstance(target, list):
        index = _override_list_index(part, raw_path)
        if index >= len(target):
            raise ValueError(f"Unknown override path {raw_path!r}")
        return target[index]
    raise ValueError(f"Unknown override path {raw_path!r}")


def _override_list_index(part: str, raw_path: str) -> int:
    try:
        index = int(part)
    except ValueError as exc:
        raise ValueError(
            f"Override path {raw_path!r} requires a numeric list index"
        ) from exc
    if index < 0:
        raise ValueError(f"Unknown override path {raw_path!r}")
    return index
