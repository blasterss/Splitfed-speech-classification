# Resource Metrics v2

This describes the implemented measurement contract, not an implementation plan.
Measurements use schema `2` and policy `resource_metrics_v2`. The owning code is
[resource_metrics.py](../../src/utils/runtime/resource_metrics.py), persistence
is in [artifacts.py](../../src/application/artifacts.py), and transport accounting
belongs to [src/transport/](../../src/transport/__init__.py).

## Distinct quantities

| Quantity | Interpretation | Aggregation |
| --- | --- | --- |
| Owned static bytes | Parameters, model buffers, optimizer state and dataset tensor storage | Max participant and sum across participants |
| Process peak RSS | Lifetime high-water mark, including interpreter/framework/cache | Per-process max; never sum as host peak |
| Phase sampled RSS | Samples of process resident memory during one interval | Per-process/phase max |
| CUDA allocated/reserved peak | PyTorch allocator peaks for the interval | Per-process peaks; not total device usage |
| Wall/CPU time | Interval elapsed and user/system CPU time | Separate phase/role summaries |
| Transport bytes/messages | Logical payload and serialized envelope accounting | Direction/channel/message-type totals |

All persisted memory values are bytes. Use MiB = bytes / 1024^2 and
GiB = bytes / 1024^3 for presentation. Tensor views sharing storage are
counted once inside each ownership component. The static sum excludes transient
activation/gradient buffers, framework overhead and non-tensor Python objects.
Optimizer storage depends on whether optimizer state has been initialized.

## Measurement behavior

`ResourceTracker` records role, optional client ID, PID, round, phase, wall/CPU
time and memory. Linux phase RSS sampling uses `/proc/self/status` at a default
0.05-second interval. Sampling may miss short peaks. When current RSS is
unavailable, phase fields are null and the backend is `process_peak_only_v1`.
Lifetime RSS comes from `getrusage`, with platform-specific conversion to bytes.
CUDA intervals reset PyTorch peak counters; unavailable CUDA fields are null.
Sampler threads are stopped when the interval finishes.

Measurements distinguish dataset load, model setup, training, aggregation,
evaluation and checkpoint/lifetime work where emitted by the owning worker.
The persisted `training` summary includes non-evaluation rows except controller
rows; `evaluation` includes phases beginning with `evaluation`. Consequently,
inspect the phase rows if a comparison requires excluding checkpoint or setup
cost. Sums of worker wall intervals are work totals, not concurrent experiment
elapsed time. `experiment_wall_time_seconds` comes from the controller interval
when available.

## Ownership by topology

| Role | Persistent tensor ownership |
| --- | --- |
| Local training | Complete model, optimizer and source corpus |
| Local evaluation | Fresh evaluation process, checkpoint model and target views |
| Centralized | Complete model, optimizer and combined corpus views |
| Federated client | Complete local model, optimizer and local corpus |
| Split client | Client partition, optimizer and local corpus |
| SplitServer | Server partition and optimizer; personalized workers stay separate |
| FedServer | Aggregation/global state, with no training dataset ownership |

Local starts separate `LocalTrain` and `LocalEval` spawn processes. Evaluation
loads other corpora only after the training process ends. Training resource
peaks therefore do not include local cross-corpus evaluation allocations.

The owned-footprint summary uses each role/client/PID participant's maximum
reported static bytes in `train` or `aggregation`, then aggregates by role.
Its logical total does not measure simultaneous physical host memory or account
for sharing between processes. Concurrent host usage needs another sampler.

## Transport

Transport rows retain channel, direction, participant and message-type breakdown.
Queue accounting describes logical tensor/envelope sizes; gRPC includes protobuf
serialization size. Neither is a measurement of physical network traffic,
network protocol overhead or link bandwidth. Activations, gradients, labels,
model state and envelope fields are attributed by the transport serializer.
Aggregate sent and received totals carefully to avoid counting the same hop twice.

## Artifacts

Training runs backed by `models_save_path` write under
`<models_save_path>/<experiment.name>/metrics/`:

- `resource_metrics.csv`: interval rows, including schema/policy and process IDs.
- `resource_by_participant.csv`: participant/phase summaries.
- `resource_transport.csv`: channel/direction/message-type breakdown.
- `resource_summary.yaml`: training/evaluation summaries, owned footprint,
  participant records, roles and controller elapsed time when available.

Local cross-corpus output places resource files under its `local_cross_corpus/`
directory. Missing observations are not zero-cost evidence. Schema-v1 rows are
rejected by current persistence and must not be mixed silently with schema v2.
Older saved runs remain historical artifacts with their original schema.

## Reporting and validation

Report training and evaluation separately, include maximum client and server
role values, and distinguish owned footprint from runtime memory. Do not sum
RSS/CUDA peaks as host usage. Publish the measurement policy, corpus/actor
coverage, workload policy and repeated-sample semantics alongside quality.

Tests cover storage accounting, RSS/timer lifecycle, phase summaries, transport
breakdown and process isolation. Focused checks include:

```bash
uv run pytest tests/runtime/test_resource_metrics.py \
  tests/persistence/test_resource_metrics.py \
  tests/transport/test_resource_accounting.py
```

A full protocol benchmark, automated paper-table generation and simultaneous
host sampler are separate research/development work. See the
[experiment plan](../EXPERIMENT_PLAN.md) and [roadmap](../development/ROADMAP.md).
