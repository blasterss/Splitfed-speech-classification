from pathlib import Path

import torch

CHECKPOINT_SCHEMA_VERSION = 1


def save_checkpoint(
    path: str | Path,
    *,
    mode: str,
    model_state_dict: dict,
    server_model_scope: str | None = None,
    client_id: str | int | None = None,
) -> None:
    """Atomically save a versioned checkpoint with an ownership contract."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "mode": mode,
        "server_model_scope": server_model_scope,
        "client_id": client_id,
        "model_state_dict": model_state_dict,
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(envelope, temporary)
    temporary.replace(destination)


def load_checkpoint(
    path: str | Path,
    *,
    expected_mode: str,
    expected_state_dict: dict,
    expected_server_model_scope: str | None = None,
    expected_client_id: str | int | None = None,
) -> dict:
    """Validate ownership and tensor schema before returning model state."""
    envelope = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(envelope, dict):
        raise ValueError("Checkpoint must contain a dictionary envelope")
    if envelope.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Unsupported checkpoint schema_version")
    _require_identity(envelope, "mode", expected_mode)
    _require_identity(
        envelope, "server_model_scope", expected_server_model_scope
    )
    _require_identity(envelope, "client_id", expected_client_id)

    state = envelope.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint model_state_dict must be a dictionary")
    if state.keys() != expected_state_dict.keys():
        raise ValueError(
            "Checkpoint state_dict keys do not match model schema"
        )
    for key, expected in expected_state_dict.items():
        received = state[key]
        if not isinstance(received, torch.Tensor):
            raise ValueError(f"Checkpoint value for {key!r} is not a tensor")
        if received.shape != expected.shape:
            raise ValueError(f"Checkpoint shape mismatch for {key!r}")
        if received.dtype != expected.dtype:
            raise ValueError(f"Checkpoint dtype mismatch for {key!r}")
    return state


def _require_identity(envelope: dict, field: str, expected: object) -> None:
    if envelope.get(field) != expected:
        raise ValueError(
            f"Checkpoint {field} mismatch: expected {expected!r}, "
            f"got {envelope.get(field)!r}"
        )
