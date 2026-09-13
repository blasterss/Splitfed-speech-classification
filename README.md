# SecureASR

SecureASR is an early research prototype for binary classification of emotional
speech with Split Federated Learning (SplitFed). It combines a client-side
feature extractor, a split-learning server and federated averaging of client
models.

Despite the repository name, the current task is **not automatic speech
recognition in the conventional speech-to-text sense**. The implemented target
is binary classification where anger is treated as the positive class. The
project should therefore be understood as an experimental privacy-aware speech
classifier, not as a production ASR or conflict-detection system.

> **Project status: pre-alpha research prototype.** The implementation is useful
> for experiments and further development, but it is not fault-tolerant,
> cryptographically secure or ready for deployment. Read
> [Known limitations](#known-limitations) before running long experiments.

## Current scope

The repository currently provides:

- loaders for CREMA-D, RAVDESS and SAVEE;
- MFCC, RMS, ZCR, Mel and spectral-contrast feature extraction;
- actor-disjoint train/test splitting;
- a client-side residual 1D CNN;
- a server-side CNN with either global pooling or a bidirectional RNN;
- local multiprocessing channels based on `multiprocessing.Queue`;
- protobuf-serialized insecure gRPC channels for local multi-process runs;
- explicit `local`, `centralized`, `federated`, `split/shared`,
  `split/personalized` and `splitfed` execution topologies;
- synchronous split-learning forward/backward steps;
- explicit per-client round completion for unequal local loader lengths;
- validated split message identity/correlation and duplicate-step rejection;
- configured uniform or dataset-weighted FedAvg for the client-side model;
- bounded partial-quorum FedAvg windows with correlated late-client catch-up;
- strict configuration, a typed registry with a versioned smoke profile and
  typed CLI overrides;
- experiment-scoped resolved config, provenance, dataset manifests, metrics and
  atomic ownership-aware model checkpoints;
- per-process resource metrics for wall/CPU time, throughput, peak RSS and
  PyTorch CUDA memory, plus transport-message byte counts;
- optional Gaussian or Laplace perturbation of intermediate activations;
- local evaluation with accuracy, F1, precision and recall.

The following are declared or partially scaffolded but are **not implemented as
working features**:

- secure aggregation;
- formal differential privacy accounting;
- streaming microphone inference;
- fault-tolerant or asynchronous clients;
- a production inference service.

## Architecture

```text
Configured experiment
  local               -> one isolated complete model per corpus
  TrainingController
  centralized         -> one complete model, no channels or servers
  federated           -> complete client model x N <-> FedServer
  split/shared        -> client partition x N <-> one SplitServer model
  split/personalized  -> client partition x N <-> isolated server model x N
  splitfed            -> split server + client-partition FedAvg; the server
                           scope may be shared or personalized

SplitServer
  validated activations + labels -> server model -> correlated gradients

FedServer
  validated client states + dataset weights -> configured FedAvg ->
  correlated global client state
```

The controller creates only the processes and logical channels owned by the
selected mode. It supervises child/server exit codes and coordinates bounded
cancellation and shutdown.

All components currently run on one Unix host. Queue transport simulates
distributed participants but does not provide network isolation.

## Repository layout

```text
configs/
  config.example.yaml     example experiment configuration
  logger.yaml             logging configuration
notebooks/
  exploration.ipynb       exploratory data analysis
  debug_runtime.ipynb     model/runtime experiments
src/
  dataset/
    audio/                WAV loading and standalone segmentation
    features/             configured acoustic feature extraction
    processors/           corpus discovery and filename parsing
    analytics/            exploratory feature aggregation
  model/                  client-side and server-side neural networks
  splitfed/               clients, servers and training controller
  transport/              message and local queue abstractions
  main.py                 command-line entry point
  schema.py               Pydantic configuration models
```

## Environment

The development target is Unix with Python 3.10-3.12. Queue transport runs
locally; CUDA is optional and should be enabled only when an NVIDIA runtime is
available.

### Python dependencies

Use Python 3.10-3.12. PyTorch is deliberately separated from the ordinary
dependency set because its wheel is large and must match the selected CUDA
runtime.

Create the environment and install the default dependency set with `uv`:

```bash
uv venv --python 3.12
uv sync --extra train
```

Install development tools when needed:

```bash
uv sync --extra train --extra dev
```

On Linux and Windows, the `train` extra resolves the pinned official PyTorch
CUDA 12.8 wheel; this supports Blackwell (`sm_120`) GPUs. macOS falls back to
the PyPI wheel. `uv` keeps downloaded packages in its shared cache and normally
hardlinks them into project environments, so CUDA libraries are not downloaded
again for every project. A compatible system NVIDIA driver is still required;
installing a system CUDA Toolkit does not replace the runtime bundled with the
PyTorch wheel. Verify the runtime before loading datasets:

```bash
uv run python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_arch_list())'
```

## Configuration

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

Important configuration caveats:

- unknown fields are rejected at every configuration level;
- `training.mode` is typed as `local`, `centralized`, `federated`, `split` or
  `splitfed`; mode-specific server/channel topology is validated. Execution is
  implemented for `local`, `centralized`, `federated`, `split/shared`,
  `split/personalized` and `splitfed`;
- `split_server.model_scope` is `shared` or `personalized`. Personalized
  SplitFed keeps one server model and optimizer per client while aggregating
  client-side models; this is not the canonical SFLv1 server-model aggregation
  protocol. Personalized split keeps one server model, optimizer, metrics
  stream and checkpoint per client;
- `training.fed_every` currently controls federated synchronization;
- personalized SplitFed uses the same synchronization cadence and aggregation
  strategy for both partitions; a correlated server ACK forms a round barrier;
- `clients[].runtime.workload_policy` is `max_steps_v1` by default;
  `full_epoch_v1` consumes the complete local loader and makes `local_steps`
  an unused compatibility value for that client; `fixed_steps_v1` cycles a
  non-empty loader as needed and completes exactly `local_steps`, so smaller
  clients may reuse samples within one round;
- `training.barrier_timeout_sec` bounds client ready/evaluation barriers;
- `split_server.model.gradient_accumulation_steps` controls how many server
  batches are averaged per optimizer update; each round flushes its remainder;
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

The repository also contains a forward-looking research plan for additional
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
it into the shared stop event and barrier abort. Blocking queue receives poll
that event and exit without waiting for the full channel timeout.

Centralized dataset views must use identical model, batch size, device, noise,
feature ordering and target sample-rate settings. The first client entry owns
that single runtime configuration; `local_steps` is not used because every
combined training batch is consumed once per centralized round.

Each dataset view uses `dataset.split_seed` for its actor-disjoint train/test
partition. The default remains `42` for compatibility.

## Running

From the repository root:

```bash
uv run secureasr --config-file configs/config.yaml
```

The equivalent module invocation is:

```bash
uv run python -m src.main --config-file configs/config.yaml
```

Existing configuration values can be overridden with repeatable, typed
`--set PATH=VALUE` arguments. Values use YAML scalar/list syntax and unknown
paths are rejected before controller setup, for example:

```bash
uv run secureasr --config-file configs/config.yaml \
  --set training.num_rounds=1 \
  --set clients.0.dataset.reduced=true
```

`--profile smoke` supplies versioned defaults for one round, evaluation every
round and a 30-second lifecycle barrier. YAML values override those defaults,
and `--set` overrides YAML. The profile does not rewrite dataset paths, client
counts, reduced-data selection or devices, so those must still be configured
for the intended CPU smoke topology.

Start with reduced datasets and a small number of rounds. A full CUDA run should
only be attempted after all clients load successfully and a one-round smoke test
has completed.

Run every implemented ownership topology from one validated base configuration
with:

```bash
uv run python -m src.experiments.mode_matrix \
  --config-file configs/config.real.yaml --rounds 1 \
  --artifact-root artifacts/mode_matrix
```

Use repeatable `--mode` values to select a subset. The runner writes elapsed
wall time and pass/fail status to `mode_matrix_summary.yaml`; it is a smoke and
diagnostic runner, not a benchmark harness. In SplitFed, evaluation at an
aggregation round occurs before FedAvg so that the client encoder and shared
server model are a trained, compatible pair. The global client encoder is then
installed for the next round.

Run the E1 local-only cross-corpus baseline with:

```bash
uv run secureasr \
  --config-file configs/experiments/config.e1.1.yaml
```

It trains one independent complete model per corpus and evaluates every model
on the actor-disjoint test view of every corpus. Evaluation always reuses the
training corpus normalization statistics. The harness writes a tidy CSV,
metric matrices in YAML, and one checkpoint per training corpus under
`artifacts/e1_local_cross_corpus/local_cross_corpus/`.

Run the complete-model E1 baselines with:

```bash
uv run secureasr --config-file configs/experiments/config.e1.2_centr.yaml
uv run secureasr --config-file configs/experiments/config.e1.2_fl.yaml
```

Both configs evaluate the validated final checkpoint separately on each
actor-disjoint test corpus and write rate metrics plus macro/worst-corpus
summaries under `<experiment>/metrics/cross_corpus/`. The federated config uses
`full_epoch_v1`, sample-weighted aggregation and a final global aggregation.

Expected generated artifacts include:

- rotating logs under `logs/` when logging configuration is loaded;
- evaluation CSV files under
  `<models_save_path>/<experiment.name>/metrics/`;
- centralized, server and global client checkpoints under
  `<models_save_path>/<experiment.name>/checkpoints/`;
- validated `resolved_config.yaml` and `run_metadata.yaml` under
  `<models_save_path>/<experiment.name>/metadata/`, including environment
  provenance, Git dirty state, the configured seed tree, selected profile
  name/version/source, typed CLI override records, canonical resolved config
  SHA-256, Git revision, `uv.lock` SHA-256, local multiprocessing identity and
  implemented transport/aggregation policy versions. Git/lock values are null
  when unavailable; container image digest and scheduler policy remain null for
  the current local runtime.
- `dataset_manifest.yaml` in the same metadata directory with per-client
  extraction counts/failure reasons, split seed, feature ordering,
  actor-disjoint IDs and train/test sample/actor/class coverage. Missing client
  reports are explicit and mark the manifest incomplete.
- `diagnostics/first_failure.yaml` after a captured worker/controller failure,
  with a bounded versioned record containing component, optional
  client/round/step context, exception type/message and traceback.

Queue and gRPC messages carry and validate protocol identity
`secureasr.transport` version 4 plus a bounded non-empty request ID. Split
responses and accepted federated updates must echo the originating request ID
in addition to matching sender, type, round and step. Channel send stamps an
unset hop deadline from the positive channel timeout; expired messages are
rejected at send and receive boundaries and again before client/server payload
use. This is not yet a single end-to-end RPC deadline across split batching or
federated quorum waiting.
Split and federated workers reject request IDs repeated within a bounded FIFO
window of 10,000 accepted messages. This in-process cache is reset on worker
restart and is not durable broker-level replay protection.

Model files use checkpoint schema version 1 and record training mode, server
model scope and personalized client identity where applicable. Checkpoint writes
are atomic, and the loader rejects incompatible ownership plus mismatched tensor
keys, shapes and dtypes before returning a state dict.

If custom logging configuration does not create its parent directory, prepare
the log directory before running:

```bash
mkdir -p logs
```

## Verification status

The repository has focused tests for schema/configuration, dataset parsers,
padding, models/FedAvg, queue transport and controller lifecycle. Run them with:

```bash
uv run pytest
```

The suite includes a synthetic CPU `spawn` smoke cycle with two unequal clients,
real split/federated workers, training, FedAvg and evaluation. It does not
replace a reduced real-data or CUDA smoke test. Use a reduced dataset and one
round before starting a long experiment.

## Dataset and evaluation assumptions

The current loaders map only the `ANG` emotion to the positive class. All other
supported emotions map to zero. Consequently:

- results measure anger classification, not conflict understanding;
- acted emotional datasets may not generalise to real conversations;
- dataset identity, recording conditions and speaker characteristics may become
  shortcuts for the model;
- SAVEE has very few actors, making actor-disjoint evaluation unstable;
- a single train/test split is insufficient for strong scientific conclusions.

Any reported experiment should include class counts, actor counts, per-dataset
metrics, multiple random seeds and an external or leave-one-dataset-out test.
Macro-F1 and per-class recall should be reported alongside accuracy because the
positive class is imbalanced.

## Privacy and security statement

Split learning prevents clients from sending raw waveform data directly to the
server. This alone does not guarantee confidentiality.

The current server receives intermediate activations and labels. The federated
server receives complete client model parameters. Local queues and the current
insecure gRPC mode provide no cryptographic isolation. TLS/mTLS and
authentication are not implemented. Intermediate activations may still reveal
speaker identity, recording domain or speech content.

The optional `PrivacyLayer` adds random noise, but it is **not a complete
differential privacy implementation**:

- clipping is not exposed through configuration and is disabled by default;
- no epsilon/delta accounting is performed;
- the example noise level is too small to represent meaningful protection;
- configured training noise is disabled by `model.eval()`;
- no reconstruction, membership-inference or attribute-inference evaluation is
  included.

Do not describe the current implementation as cryptographically secure or
differentially private. A defensible privacy claim requires an explicit threat
model, calibrated clipping and noise, privacy accounting and empirical leakage
tests.

## Known limitations

### Training and process lifecycle

- Clients send `round_end` after their final local step, allowing longer client
  loaders to continue without waiting for an already-finished peer. A missing
  completion message leaves a partial batch only until its configured timeout;
  the server then sends correlated errors to clients already waiting for it.
- The controller has a polling supervision loop and propagates non-zero child
  exit codes. Client initialization and barrier failures now set the shared
  stop event, abort peer barriers and publish a bounded first-failure record,
  but queue timeouts and server failures are not yet one complete cancellation
  protocol. SplitServer and FedServer publish their original failure context
  before cancellation; recovery is not yet implemented.
- Queue timeouts can still leave peers waiting in some failure paths.

### Numerical correctness

- Server gradient accumulation is configurable and remainder gradients are
  flushed at the end of each completed round. Client activation gradients use
  the full batch loss; only accumulated server parameter gradients are averaged.
- FedAvg averages floating tensors and copies non-floating buffers from the
  accepted client with the largest dataset. This prevents accidental averaging
  of integer counters, but it is not a researched BatchNorm policy for non-IID
  clients; FedBN and server-local alternatives remain unimplemented.
- Optimiser and RNG state are not restored with model checkpoints; schema v1
  currently provides model ownership and tensor validation, not resume
  equivalence.

### Messages and aggregation

- Client responses validate protocol, deadline, request ID, sender, type, round
  and step before payload use.
- Federated client/global updates validate sender, type, round, step, keys,
  tensor shapes and dtypes before aggregation or `load_state_dict`.
- Partial quorum uses an arrival window, not a fairness-aware cohort scheduler.
  Aggregation responses go only to accepted participants. A validated client
  arriving later in the same completed round receives a correlated catch-up
  model without changing the completed aggregate. A client lagging by more
  than one completed round still has no catch-up/resume protocol and fails
  through bounded lifecycle handling.
- `aggregation_freq` and `training.fed_every` are required to match; clients use
  that cadence to trigger server aggregation.
- Metrics are logged locally and are not collected by the federated worker.

### Data pipeline

- Feature extraction loads each client dataset fully into memory.
- Stacked features record valid frame counts before padding. Train-only
  normalization ignores padded frames and restores padded positions to zero
  after normalization.
- Actor-disjoint splits record per-split actor/sample/class coverage and reject
  training partitions that do not contain both binary classes.
- Per-file feature extraction failures are skipped but recorded as discovered,
  loaded and failed counts grouped by exception type; raw paths are not
  persisted in the report.
- Spectral-contrast failures are counted as extraction loss instead of being
  hidden behind synthetic zero features.
- Multi-channel utilities are incomplete and inconsistent with the main path.

### Evaluation and artifacts

- Evaluation runs every `eval_every` rounds and always on the final round.
  Per-round client CSV names prevent later snapshots from overwriting earlier
  results.
- Evaluation metrics include sample, positive-label and positive-prediction
  denominators and remain finite for one-class test partitions.
- Checkpoint envelopes contain mode/scope ownership but not optimiser state,
  RNG state, resolved configuration or dataset manifest. Resolved configuration
  and the bounded dataset manifest are stored as separate metadata artifacts.
- The experiment cannot currently be resumed reliably.

### Packaging and unfinished modules

- Imports mix package-relative, `src.*` and top-level forms.
- The microphone capture module contains undefined constants and empty methods.
- The audio segmenter calls the audio loader with an incompatible signature.
- gRPC is limited to insecure local process endpoints; TLS/mTLS,
  authentication, health RPCs and container orchestration are absent.
- There is no CI workflow, resume-equivalence suite or CUDA test matrix yet.
  Synthetic CPU spawn and checkpoint contract round-trips are automated.
  Reduced real-data and RTX 5060/CUDA 12.8 runs have been executed manually,
  but are not reproducible CI gates.

## Remaining improvement plan

Completed startup validation, protocol correlation/deadlines/replay checks,
partial quorum, configurable accumulation, data-split controls, artifact
metadata and mode baselines are intentionally omitted here. The remaining
work, in delivery order, is:

1. Add the typed policy registry, deterministic heterogeneous simulator and
   remaining named research profiles.
2. Unify queue, quorum, barrier and shutdown deadlines under one typed
   cancellation protocol and persist explicit failed run status.
3. Implement `ClientLoadController` telemetry, leases, fairness debt,
   quarantine and replayable cohort policies.
4. Bound dataset memory use, freeze input manifests and decide a typed
   class-balance sampling policy.
5. Extend checkpoints with optimizer/RNG/round/config/manifest state and prove
   resume equivalence.
6. Persist process resource/transport telemetry and aggregate per-client,
   per-dataset and worst-client metrics.
7. Add multi-seed and leave-one-dataset-out experiments plus CI/static gates.
8. Only after P0 stability, implement and test privacy accounting/leakage
   baselines and the TLS/authenticated gRPC deployment path.
9. Add the non-root Compose reference topology after the research runtime,
   policies and artifact contracts are stable.

## Experimental roadmap

The implementation audit, baseline definitions, required changes, and the
refactored experiment sequence are maintained in
[docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md). The original
`exp_plan.docx` remains source material rather than an executable protocol.

The intended order is: quantify corpus heterogeneity; complete the common
evaluation contract; establish Local/Centralized/FedAvg references; implement
faithful SFLv1, SFLv2 and MergeSFL baselines; then evaluate server optimization,
failure robustness, unequal workloads and leave-one-corpus-out generalization.
The existing mode matrix remains a smoke/diagnostic runner, not a benchmark
harness.

## License

This project is distributed under the MIT License. See [LICENSE](LICENSE).
