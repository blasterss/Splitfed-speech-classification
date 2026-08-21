from src.main import _save_dataset_manifest
from src.utils.config import read_yaml


def test_dataset_manifest_records_missing_clients(tmp_path):
    _save_dataset_manifest(
        tmp_path,
        expected_client_ids=[0, 1],
        client_manifests={
            0: {
                "client_id": 0,
                "dataset": "SAVEE",
                "coverage": {"train": {"samples": 2}},
            }
        },
    )

    manifest = read_yaml(tmp_path / "dataset_manifest.yaml")
    assert manifest["schema_version"] == 1
    assert not manifest["complete"]
    assert manifest["missing_client_ids"] == [1]
    assert manifest["clients"][0]["client_id"] == 0
