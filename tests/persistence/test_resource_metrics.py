import csv
from types import SimpleNamespace

from src.application.artifacts import (
    _save_resource_metrics,
    _save_splitfed_quality,
)
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


def test_save_resource_metrics_writes_client_averages(tmp_path):
    metrics = [
        {
            "role": "client",
            "client_id": 0,
            "round": round_number,
            "phase": "train",
            "wall_time_seconds": float(round_number),
            "cpu_user_seconds": 1.0,
            "cpu_system_seconds": 0.5,
            "peak_rss_bytes": 100 + round_number,
            "peak_cuda_allocated_bytes": None,
            "peak_cuda_reserved_bytes": None,
            "samples": 8,
            "batches": 2,
            "samples_per_second": 8 / round_number,
            "batches_per_second": 2 / round_number,
        }
        for round_number in (1, 2)
    ]

    _save_resource_metrics(metrics, tmp_path)

    with (tmp_path / "resource_by_client.csv").open() as source:
        rows = list(csv.DictReader(source))
    assert rows[0]["role"] == "client"
    assert rows[0]["client_id"] == "0"
    assert float(rows[0]["avg_wall_time_seconds"]) == 1.5
    assert rows[0]["max_peak_rss_bytes"] == "102"


def test_save_splitfed_quality_writes_final_client_summary(tmp_path):
    (tmp_path / "Client0_round_10_eval.csv").write_text(
        "probs,preds,labels\n0.9,1,1\n0.1,0,0\n",
        encoding="utf-8",
    )
    (tmp_path / "Client0_round_20_eval.csv").write_text(
        "probs,preds,labels\n0.9,1,1\n0.9,1,0\n",
        encoding="utf-8",
    )
    config = SimpleNamespace(
        training=SimpleNamespace(
            mode=SimpleNamespace(value="splitfed"), num_rounds=20
        ),
        clients=[
            SimpleNamespace(
                client_id=0,
                dataset=SimpleNamespace(name=SimpleNamespace(value="SAVEE")),
            )
        ],
    )

    _save_splitfed_quality(tmp_path, config)

    summary = read_yaml(tmp_path / "splitfed_summary.yaml")
    assert summary["complete"] is True
    assert summary["missing_client_ids"] == []
    assert summary["rows"][0]["round"] == 20
    assert summary["rows"][0]["accuracy"] == 0.5
    assert summary["rows"][0]["anger_f1"] == 2 / 3
    assert summary["sample_weighted_average"]["accuracy"] == 0.5


def test_save_splitfed_quality_marks_missing_and_stale_clients(tmp_path):
    stale_path = tmp_path / "Client0_round_20_eval.csv"
    stale_path.write_text(
        "probs,preds,labels\n0.9,1,1\n",
        encoding="utf-8",
    )
    config = SimpleNamespace(
        training=SimpleNamespace(
            mode=SimpleNamespace(value="splitfed"), num_rounds=20
        ),
        clients=[
            SimpleNamespace(
                client_id=0,
                dataset=SimpleNamespace(name=SimpleNamespace(value="SAVEE")),
            ),
            SimpleNamespace(
                client_id=1,
                dataset=SimpleNamespace(name=SimpleNamespace(value="RAVDESS")),
            ),
        ],
    )

    _save_splitfed_quality(
        tmp_path,
        config,
        created_after=stale_path.stat().st_mtime + 1,
    )

    summary = read_yaml(tmp_path / "splitfed_summary.yaml")
    assert summary["complete"] is False
    assert summary["missing_client_ids"] == ["0", "1"]
    assert summary["rows"] == []
    with (tmp_path / "splitfed_metrics.csv").open() as source:
        assert list(csv.DictReader(source)) == []
