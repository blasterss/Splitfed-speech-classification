# Training configuration

Copy the example without committing the local file:

```bash
cp configs/config.example.yaml configs/config.yaml
```

Then update at least:

- every `clients[].dataset.root`;
- `models_save_path` (note the plural form) is the artifact root;
- client and split-server devices;
- batch size and local steps for available GPU memory;
- channel timeouts;
- the noise configuration, if perturbation is required.

Relative paths are resolved from the repository root when the command is run
there. The example expects the downloaded datasets in `../datasets`.

## Modes and validation

- unknown fields are rejected at every configuration level;
- `training.mode` is typed as `local`, `centralized`, `federated`, `split` or
  `splitfed`; mode-specific server/channel topology is validated. Execution is
  implemented for `local`, `centralized`, `federated`, `split/shared`,
  `split/personalized` and `splitfed`;
- `split_server.model_scope` is `shared` or `personalized`. Personalized
  SplitFed keeps one server model and optimizer per client while aggregating
  both model partitions at the configured cadence. Canonical SFLv1 equivalence
  still requires numerical protocol verification. Personalized split keeps one
  server model, optimizer, metrics stream and checkpoint per client;
- `training.fed_every` currently controls federated synchronization;
- `split_server.training_strategy` supports `concat_v1`, `sequential_v1`,
  `mergesfl_v1` and `mergesfl_algorithm1_v1`. The static `mergesfl_v1` path
  follows the public reference implementation:
  it merges matched client activations into one batch and rescales each
  dispatched gradient by the merged batch size divided by that client's batch
  size. It requires `fixed_steps_v1` and equal `local_steps`; per-client
  `batch_size` remains explicit configuration rather than an automatic
  resource optimizer;
- `mergesfl_algorithm1_v1` enables the reconstructed Algorithm 1 control
  plane. `TrainingController` deterministically selects a cohort, regulates
  batch sizes from timestamped compute/transfer telemetry, enforces bandwidth
  and KL constraints, and sends one correlated `RoundPlan` to clients and both
  servers. It requires shared SplitFed, `drop_last: true`, `fed_every: 1`,
  final aggregation, SGD on both model partitions and
  `mergesfl_batch_weighted_v1`. Planned batches are capped by each train split;
  client learning rates scale with their planned batches, and every candidate
  sends a round-scoped measurement or heartbeat. The GA and integer
  refinement remain documented reconstructions because neither the paper nor
  the public repository publishes enough solver detail to reproduce them
  exactly;
- personalized SplitFed uses the same synchronization cadence and aggregation
  strategy for both partitions; a coordinator without a model aggregates
  server states, while each client-owned server model, optimizer and RNG run
  in a separate process. A correlated server ACK forms a round barrier;
- `clients[].runtime.workload_policy` is `max_steps_v1` by default;
  `full_epoch_v1` consumes the complete local loader and makes `local_steps`
  an unused compatibility value for that client; `fixed_steps_v1` cycles a
  non-empty loader as needed and completes exactly `local_steps`, so smaller
  clients may reuse samples within one round;
- `training.barrier_timeout_sec` bounds client ready/evaluation barriers;
- `split_server.model.gradient_accumulation_steps` is fixed at `1`; each
  strategy batch causes one server optimizer update;
- `split_server.model.batch_timeout_sec` bounds incomplete split batches;
- `training.eval_every` schedules synchronized evaluation snapshots; the final
  round is always evaluated;
- `fed_server.min_clients` and `quorum_timeout_sec` control partial aggregation;
  accepted participants receive the result immediately, while a validated
  client arriving later in the same completed round receives the current
  global state correlated to its own request;
- `fed_server.strategy: fedavg` assigns equal weight to every accepted client;
  `weighted_fedavg` weights floating tensors by dataset size. Non-floating
  buffers come from the largest accepted dataset. This is recorded as buffer
  policy `weighted_floating_state_largest_nonfloating_v1`; floating BatchNorm
  running statistics are therefore averaged, while integer counters are not.
  `aggregation_freq` must equal
  `training.fed_every`, which is the single synchronization cadence;
- server channel references must match the four canonical logical roles used by
  the controller;
- client IDs must be unique; referenced server channels and feasible client
  quorum are validated before controller setup.
- device values must be `cpu`, `cuda`, or `cuda:N`; requested CUDA devices are
  checked for availability before any child process is spawned.
- optimizer and noise distribution names are typed registries. Channel
  compression and gRPC TLS selections are rejected before setup because those
  paths remain unimplemented;
- `experiment.analysis_only: true` makes a dataset-analysis configuration
  valid for notebooks but rejects accidental dispatch to the training runtime;
- `experiment.cross_corpus_evaluation: true` is available only for centralized
  and federated complete-model runs backed by `models_save_path`. Federated
  evaluation additionally requires an aggregation on the final round.

## Transport configuration

For gRPC, set `experiment.transport: grpc` and configure a unique receiver
address for every client on each logical channel. Client IDs are integer keys:

```yaml
channels:
  split_uplink:
    transport: grpc
    name: split_uplink
    addresses:
      0: 127.0.0.1:51000
      1: 127.0.0.1:51001
    use_tls: false
    timeout_sec: 30
    max_message_bytes: 67108864
```

Apply the same structure with non-overlapping addresses to every channel owned
by the selected mode. The current gRPC implementation is intended for local
process research runs. TLS/mTLS, authentication, health RPCs and container
deployment are not implemented.

## Profiles, seeds and lifecycle

The [development roadmap](../development/ROADMAP.md) proposes additional
launch profiles, simulation, isolated client containers and throughput-aware
client scheduling. The built-in profile registry validates unique names and
exposes typed, versioned definitions. The `smoke` and `unit` profiles currently
provide bounded training defaults without inventing dataset roots, clients or
topology; the other planned profiles and capabilities are not part of the
current runtime.
The controller creates only mode-owned roles and channels. Centralized mode
trains one complete `SpeechRecognitionModel` over the combined client dataset
views without transport channels or servers. Federated mode trains and
aggregates complete client models; split/shared uses the client partition and
one shared SplitServer; SplitFed adds client-partition FedAvg.
Federated and SplitFed clients initialize their aggregatable model partition
from the common `training.seed`; per-client loader shuffling remains controlled
by `clients[].runtime.seed`.

`TrainingController` explicitly owns a multiprocessing `spawn` context for its
manager, queues and child processes; callers do not need to set a global start
method before using the controller API. The CLI always tears down the manager
after stopping workers and persisting available artifacts, including setup,
training and artifact failures. Forced process shutdown performs a final
bounded join after kill and reports a process that still cannot be reaped.
Spawned training workers ignore terminal SIGINT so the controller alone turns
it into the shared stop event and barrier abort. Blocking channel receives poll
that event and exit without waiting for the full channel timeout.

Centralized dataset views must use identical model, batch size, device, noise,
feature ordering and target sample-rate settings. The first client entry owns
that single runtime configuration; `local_steps` is not used because every
combined training batch is consumed once per centralized round.

Each dataset view uses `dataset.split_seed` for its actor-disjoint train/test
partition. The default remains `42` for compatibility.

## Owning code

Configuration is defined in [src/schema/](../../src/schema/__init__.py),
resolved by [application/configuration.py](../../src/application/configuration.py),
and dispatched by [application/dispatch.py](../../src/application/dispatch.py).
The E0 dataset-only format is documented separately in
[E0 domain shift](../experiments/E0_DOMAIN_SHIFT.md) and is not a training
`ConfigSchema` input.

## Launch and inspect

```bash
uv run secureasr --config-file configs/config.yaml
uv run python -m src.main --config-file configs/config.yaml
```

Both commands use the training CLI. Overrides are repeatable and parsed as YAML:

```bash
uv run secureasr --config-file configs/config.yaml \
  --set training.num_rounds=1 \
  --set clients.0.dataset.reduced=true
```

`--profile smoke` or `--profile unit` provides versioned defaults. YAML overrides
profile defaults; `--set` overrides YAML. A profile does not invent corpus roots,
change devices, select reduced data or remove unused topology automatically.
Review the fully resolved settings before running.

The mode matrix exercises implemented ownership topologies with a validated base:

```bash
uv run python -m src.experiments.mode_matrix \
  --config-file configs/config.example.yaml --rounds 1 \
  --artifact-root artifacts/mode_matrix
```

Use repeatable `--mode` options to select a subset. Configure reduced data and
CPU before using it as a smoke check. The summary records elapsed wall time and
pass/fail; this is a diagnostic runner, not a matched-budget benchmark.
