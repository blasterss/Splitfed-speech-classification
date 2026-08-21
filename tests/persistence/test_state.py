import torch

from src.utils.persistence import deserialize_state_dict, serialize_state_dict


def test_state_dict_bytes_round_trip_without_shared_tensor_storage():
    state_dict = {
        "weight": torch.tensor([1.5], dtype=torch.float32),
        "counter": torch.tensor(2, dtype=torch.int64),
    }

    restored = deserialize_state_dict(serialize_state_dict(state_dict))

    assert restored.keys() == state_dict.keys()
    assert all(
        torch.equal(restored[key], value) for key, value in state_dict.items()
    )
