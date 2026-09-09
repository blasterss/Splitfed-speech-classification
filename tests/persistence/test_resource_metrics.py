import csv

from src.application.artifacts import _save_resource_metrics
from src.utils.config import read_yaml


def test_save_resource_metrics_writes_rows_and_summary(tmp_path):
    metrics = [
        {
            "schema_version": 1,
            "role": "client",
            "client_id": 0,
            "round": 1,
            "phase": "train",
            "wall_time_seconds": 2.0,
            "cpu_user_seconds": 1.0,
            "cpu_system_seconds": 0.5,
            "peak_rss_bytes": 100,
            "peak_cuda_allocated_bytes": None,
            "peak_cuda_reserved_bytes": None,
            "samples": 8,
            "batches": 2,
            "samples_per_second": 4.0,
            "batches_per_second": 1.0,
        }
    ]

    _save_resource_metrics(metrics, tmp_path)

    with (tmp_path / "resource_metrics.csv").open() as source:
        assert list(csv.DictReader(source))[0]["role"] == "client"
    summary = read_yaml(tmp_path / "resource_summary.yaml")
    assert summary["event_count"] == 1
    assert summary["experiment_wall_time_seconds"] is None
    assert summary["max_peak_rss_bytes"] == 100
    assert summary["roles"] == ["client"]
