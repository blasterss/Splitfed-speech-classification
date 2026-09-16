import csv
from types import SimpleNamespace

import pytest

from src.application.artifacts import (
    _save_resource_metrics,
    _save_splitfed_quality,
)
from src.utils.config import read_yaml
from src.utils.runtime.resource_metrics import (
    RESOURCE_MEASUREMENT_POLICY,
    RESOURCE_METRICS_SCHEMA_VERSION,
)


def test_save_resource_metrics_writes_rows_and_summary(tmp_path):
    metrics = [
        {
            "schema_version": RESOURCE_METRICS_SCHEMA_VERSION,
            "measurement_policy": RESOURCE_MEASUREMENT_POLICY,
            "role": "client",
            "client_id": 0,
            "process_id": 123,
            "round": 1,
            "phase": "train",
            "wall_time_seconds": 2.0,
            "cpu_user_seconds": 1.0,
            "cpu_system_seconds": 0.5,
            "process_peak_rss_bytes": 100,
            "peak_cuda_allocated_bytes": None,
            "peak_cuda_reserved_bytes": None,
            "samples": 8,
            "batches": 2,
            "samples_per_second": 4.0,
            "batches_per_second": 1.0,
            "model_parameter_bytes": 10,
            "model_buffer_bytes": 2,
            "optimizer_state_bytes": 8,
            "owned_dataset_bytes": 20,
            "owned_static_bytes": 40,
        }
    ]

    _save_resource_metrics(metrics, tmp_path)

    with (tmp_path / "resource_metrics.csv").open() as source:
        assert list(csv.DictReader(source))[0]["role"] == "client"
    summary = read_yaml(tmp_path / "resource_summary.yaml")
    assert summary["schema_version"] == RESOURCE_METRICS_SCHEMA_VERSION
    assert summary["event_count"] == 1
    assert summary["experiment_wall_time_seconds"] is None
    assert summary["training"]["max_process_peak_rss_bytes"] == 100
    assert summary["owned_footprint"]["max_participant_bytes"] == 40
    assert summary["roles"] == ["client"]


def test_save_resource_metrics_writes_client_averages(tmp_path):
    metrics = [
        {
            "schema_version": RESOURCE_METRICS_SCHEMA_VERSION,
            "measurement_policy": RESOURCE_MEASUREMENT_POLICY,
            "role": "client",
            "client_id": 0,
            "process_id": 123,
            "round": round_number,
            "phase": "train",
            "wall_time_seconds": float(round_number),
            "cpu_user_seconds": 1.0,
            "cpu_system_seconds": 0.5,
            "process_peak_rss_bytes": 100 + round_number,
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

    with (tmp_path / "resource_by_participant.csv").open() as source:
        rows = list(csv.DictReader(source))
    assert rows[0]["role"] == "client"
    assert rows[0]["client_id"] == "0"
    assert float(rows[0]["avg_wall_time_seconds"]) == 1.5
    assert rows[0]["max_process_peak_rss_bytes"] == "102"


def test_save_resource_metrics_writes_transport_breakdown(tmp_path):
    metrics = [
        {
            "schema_version": RESOURCE_METRICS_SCHEMA_VERSION,
            "measurement_policy": RESOURCE_MEASUREMENT_POLICY,
            "role": "transport",
            "client_id": 2,
            "process_id": None,
            "round": None,
            "phase": "split_uplink",
            "byte_accounting": "logical_payload_estimate_v1",
            "wall_time_seconds": 0.0,
            "cpu_user_seconds": 0.0,
            "cpu_system_seconds": 0.0,
            "messages_sent": 3,
            "bytes_sent": 1024,
            "by_message_type": {"train_step": {"messages": 3, "bytes": 1024}},
        }
    ]

    _save_resource_metrics(metrics, tmp_path)

    with (tmp_path / "resource_transport.csv").open() as source:
        rows = list(csv.DictReader(source))
    assert rows == [
        {
            "client_id": "2",
            "channel": "split_uplink",
            "message_type": "train_step",
            "byte_accounting": "logical_payload_estimate_v1",
            "messages": "3",
            "bytes": "1024",
        }
    ]


def test_save_resource_metrics_rejects_legacy_or_mixed_contracts(tmp_path):
    with pytest.raises(ValueError, match="resource_metrics_v2"):
        _save_resource_metrics(
            [
                {
                    "schema_version": 1,
                    "role": "client",
                    "phase": "train",
                }
            ],
            tmp_path,
        )


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
