from io import BytesIO

import torch


def serialize_state_dict(state_dict: dict) -> bytes:
    buffer = BytesIO()
    torch.save(state_dict, buffer)
    return buffer.getvalue()


def deserialize_state_dict(payload: bytes) -> dict:
    if not isinstance(payload, bytes):
        raise TypeError("Serialized state_dict payload must be bytes")
    state_dict = torch.load(
        BytesIO(payload),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(state_dict, dict):
        raise ValueError("Serialized state_dict did not contain a dictionary")
    return state_dict
