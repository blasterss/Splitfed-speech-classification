# SecureASR Repository Guide

This document is the working instruction for changing and validating the
repository. It describes the current implementation, not an idealized future
architecture.

## Project identity

SecureASR is a pre-alpha research prototype for binary emotional-speech
classification with Split Federated Learning (SplitFed). The positive class is
anger (`ANG`); every other supported emotion is currently mapped to zero. It is
not a speech-to-text ASR system and should not be described as a production
privacy or security solution.

The local queue transport simulates distributed participants inside one Unix
host. It does not provide network isolation, cryptographic protection, secure
aggregation, or formal differential privacy.

## Runtime contract

- Operating system target: Unix/Linux.
- Python: `>=3.10,<3.13`.
- Dependency manager: `uv`.
- Package name: `secureasr`.
- Console entry point: `secureasr`.
- Module entry point: `python -m src.main`.
- Default transport: local multiprocessing queues.
- CUDA is optional; the example configuration uses CPU devices.

Run commands from the repository root. Relative paths in configuration are
validated against the current working directory, so launching from another
directory can make valid repository-relative paths fail.

## Repository map

- `src/main.py`: CLI, YAML loading, schema validation, controller lifecycle.
- `src/schema.py`: Pydantic configuration models and enums.
- `src/dataset/`: WAV discovery, audio loading, feature extraction, dataset
  wrappers, and dataset-specific filename parsers.
- `src/dataset/audio/`: native-rate/optional-resampling WAV loading and the
  standalone audio segment iterator; live capture is not implemented.
- `src/dataset/features/`: ordered MFCC, RMS, ZCR, Mel, and spectral-contrast
  extraction for stacked and analytics-only multi-channel representations.
- `src/dataset/processors/`: CREMA-D, RAVDESS, and SAVEE loaders registered in
  `DatasetLoaderFactory`.
- `src/dataset/analytics/`: experimental multi-channel feature aggregation;
  it is not part of the supported training path.
- Live audio capture is not implemented; the dataset package operates on WAV
  files already present under validated dataset roots.
- `src/model/`: client-side and server-side PyTorch models.
- `src/splitfed/`: clients, split server, federated server, and
  `TrainingController`; component packages own their protocol and model
  operations, `split_server/worker/` separates shared and personalized child
  processes, and `common/` owns shared process lifecycle helpers.
- `src/splitfed/client/`: client data/model runtime, response validation, and
  child-process lifecycle separated into `client.py`, `evaluation.py`,
  `protocol.py`, and `worker.py` while preserving the
  `src.splitfed.client` import surface.
- `src/splitfed/controller/`: `TrainingController` supervision plus isolated
  runtime-topology construction and diagnostic queue/report handling; the
  controller remains the owner of process startup, cancellation, and cleanup.
- `src/splitfed/centralized/`: centralized process facade, child-process
  training loop, and variable-length dataset collation/shape validation.
- `src/transport/`: `Message`, queue channel, channel factory, and the
  unimplemented gRPC channel.
- `src/utils/`: shared utilities; `persistence/` owns artifact paths,
  checkpoint envelopes, and state-dict byte serialization; `runtime/` owns
  process signal policy and structured failure publication; `training/` owns
  deterministic seed setup and per-round loss statistics; `config/` owns YAML
  configuration serialization.
- `configs/config.example.yaml`: portable CPU example configuration.
- `configs/config.real.yaml`: repository-local real-data research configuration
  for 480 files per corpus, 30 SplitFed rounds, aggregation every 10 rounds,
  CPU clients and CUDA split/federated servers.
- `configs/logger.yaml`: logging configuration.
- `tests/`: tests grouped by owning component (`config/`, `dataset/`,
  `experiments/`, `model/`, `persistence/`, `runtime/`, `splitfed/`, and
  `transport/`); pytest discovers all groups from the repository-level path.
- `notebooks/`: exploratory work; Ruff excludes notebooks.
- `uv.lock`: resolved dependency set; update it with `uv lock` when project
  dependencies change.

## Data layout

The loaders recursively discover lowercase `.wav` files below each configured
root. The current external dataset layout is:

```text
../datasets/
  CREMA-D/raw/AudioWAV/*.wav
  RAVDESS/raw/Actor_01/*.wav
  RAVDESS/raw/Actor_02/*.wav
  ...
  SAVEE/raw/ALL/*.wav
```

The archives and audio data must remain outside the Git repository. Expected
corpora are:

- CREMA-D: 7,442 WAV files, 91 actors.
- RAVDESS speech: 1,440 WAV files, 24 actors.
- SAVEE: 480 WAV files, 4 male actors.

Dataset-specific parsing is filename-based:

- CREMA-D: `actor_sentence_emotion_intensity.wav`.
- RAVDESS: seven numeric fields separated by `-`; actor is the final field.
- SAVEE: `actor_emotion_number.wav`.

Before changing a parser, test it against representative real filenames from
its corpus and preserve actor IDs because the split is actor-disjoint.

## Environment and commands

Create or refresh the environment:

```bash
uv venv --python 3.12
uv sync --extra train
```

Install development tools:

```bash
uv sync --extra train --extra dev
```

Inspect the runtime:

```bash
uv run python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_arch_list())'
```

The locked `train` extra uses the explicit official PyTorch CUDA 12.8 index on
Linux/Windows and the normal PyPI source on macOS. CUDA 12.8 is required by the
supported wheel to execute on Blackwell `sm_120`; the host still owns the
NVIDIA driver. Reuse the default shared `uv` cache rather than installing an
untracked wheel into each environment.

Run the application:

```bash
uv run secureasr --config-file configs/config.yaml
# equivalent:
uv run python -m src.main --config-file configs/config.yaml
```

Use repeatable `--set PATH=VALUE` arguments for typed YAML-value overrides of
existing fields (including numeric list indices such as
`clients.0.runtime.batch_size`). Unknown paths are rejected before process
creation, and applied overrides are recorded in `run_metadata.yaml`.

The typed built-in profile registry currently exposes `--profile smoke` and
`--profile unit`. Both provide versioned, bounded training defaults; neither
invents dataset roots, clients or topology. Duplicate registered names are
rejected.
Resolution precedence is `profile < YAML < --set`; it intentionally does not
invent dataset roots or force a topology. Configure reduced datasets, CPU
devices and the desired clients in the YAML used for a smoke run.

Run artifacts include a canonical SHA-256 of the validated resolved config,
profile/override sources, the current Git revision and dirty state, and a
SHA-256 of `uv.lock` in `run_metadata.yaml`. Runtime identity and implemented
transport/aggregation policy versions are also explicit. Unavailable checkout
or lock information is recorded as null rather than preventing checkpoint
persistence; scheduler policy and container image digest are null for the
current local runtime.
Captured client, SplitServer, FedServer and controller failures additionally
produce the bounded versioned artifact `diagnostics/first_failure.yaml`.
Workers publish original exception context before setting cancellation; if no
worker record is available, the controller writes its own fallback context.

Every local queue `Message` validates `secureasr.queue` protocol version 3 and
a bounded non-empty request ID, which are also represented in run metadata.
Split responses and federated responses to accepted updates echo request IDs.
Queue send stamps a hop deadline using the positive channel timeout; send,
receive and payload validators reject expired envelopes. It is not yet one
end-to-end deadline spanning split batching or federated quorum waiting. With
partial federated quorum, accepted clients receive the aggregate immediately;
a validated client arriving later in that same completed round receives a
correlated catch-up response. Older-round catch-up is not implemented.

Each split and federated worker has a FIFO replay guard covering the most recent
10,000 accepted request IDs. Replays in that window fail the worker before
payload processing and propagate into controller cancellation. The cache is
in-process and resets on restart; durable cross-restart replay protection is
not implemented.

Create local configuration and output directories:

```bash
cp configs/config.example.yaml configs/config.yaml
mkdir -p logs artifacts
```

Use a reduced dataset, few rounds, and CPU for the first smoke test. Do not
start a full training run until every client has loaded successfully and a
one-round run completes.

Run the validated ownership matrix from one base configuration with:

```bash
uv run python -m src.experiments.mode_matrix \
  --config-file configs/config.real.yaml --rounds 1 \
  --artifact-root artifacts/mode_matrix
```

Repeat `--mode` to select any subset of `centralized`, `federated`,
`split-shared`, `split-personalized` and `splitfed`. The summary records wall
time and outcome; it does not yet sample per-process CPU/RSS/GPU utilization.

Run the channel-free E1 local-only cross-corpus matrix with:

```bash
uv run python -m src.experiments.local_cross_corpus \
  --config-file configs/experiments/config.e1.local.yaml
```

The harness treats each configured client as one corpus view, resets the same
model seed before each isolated training run, and evaluates the resulting model
against every configured test corpus using only the source training corpus
normalization statistics. It does not invoke `TrainingController` and creates
no multiprocessing workers, channels, split servers, or federated servers.

## Change workflow

1. Read the owning module and its nearest caller or test before editing.
2. State one local hypothesis and one cheap check that can falsify it.
3. Keep the first edit narrow and validate it immediately.
4. Preserve relative imports under the `src` package; do not reintroduce
   `sys.path` manipulation or script-only imports.
5. Prefer existing Pydantic, pathlib, queue, and PyTorch patterns over new
   abstractions.
6. Keep datasets, logs, checkpoints, `.venv`, and `config.yaml` out of Git.
7. Run focused checks first, then broader checks only when the change warrants
   them.
8. Never silently alter label semantics, actor splitting, or device defaults.
9. Do not claim privacy guarantees that the implementation does not provide.
10. Do not rewrite unrelated user changes in a dirty working tree.

For Python changes, the normal focused checks are:

```bash
uv run ruff check path/to/changed.py
uv run black --check path/to/changed.py
uv run python -m py_compile path/to/changed.py
```

For configuration or loader changes, also validate the example and a real WAV:

```bash
uv run python -c "from src.schema import ConfigSchema; from src.utils.config import read_yaml; ConfigSchema(**read_yaml('configs/config.example.yaml')); print('schema-ok')"
```

The repository has focused tests for schemas/configuration, dataset parsers,
padding, models/FedAvg, queue transport and controller lifecycle. It also has a
synthetic CPU `spawn` cycle covering unequal client steps, split training,
FedAvg, evaluation, clean process exit and state handoff. Run them with
`uv run pytest`. Reduced real-data and RTX 5060/CUDA 12.8 mode runs have been
completed manually; an automated real-data/CUDA matrix and CI workflow are
still missing.

## Configuration invariants

- Unknown fields are rejected in root and nested configuration models.
- `training.mode` selects mode-specific server and channel requirements.
  Controller setup creates only those roles. Execution is available for
  `centralized`, `federated`, `split/shared`, `split/personalized` and
  `splitfed`.
- `split_server.model_scope` supports `shared` and `personalized`; SplitFed
  requires `shared`. Personalized models, optimizers, metrics and checkpoint
  files remain isolated by client ID.
- The root field is `models_save_path` (plural), not `model_save_path`; it owns
  `<root>/<experiment.name>/{metadata,checkpoints,metrics}`.
- Runs with `models_save_path` persist the validated JSON-compatible
  `resolved_config.yaml` in the experiment `metadata/` directory.
- The adjacent `run_metadata.yaml` records Python/PyTorch/platform/CUDA and Git
  details, local runtime identity, config-layer provenance, implemented policy
  versions, and experiment, training, dataset, client and server seeds.
- `dataset_manifest.yaml` records each reporting client's dataset name,
  extraction loss/reasons, split seed, feature ordering, train/test actor IDs
  and sample/actor/class coverage. Missing client reports are listed and set
  `complete: false`; raw WAV paths and tensors are not persisted.
- Model checkpoint schema v1 is written atomically and records mode, server
  scope and personalized client identity. Loading validates ownership, keys,
  shapes and dtypes; optimizer/RNG resume is not implemented yet.
- Client IDs must be unique. Server channel references and `min_clients` versus
  configured client count are validated before controller setup.
- `dataset.split_seed` controls the actor-disjoint split and defaults to `42`.
- Channel definitions use four canonical logical names: `split_uplink`,
  `split_downlink`, `federated_uplink`, and `federated_downlink`. A selected
  mode must define its owned pair(s), and the controller does not instantiate
  unused pairs.
- Dataset roots must exist before `ConfigSchema` validation.
- Client, split-server, and federated-server device choices must match the
  installed runtime; syntax, CUDA availability and indexed device bounds are
  validated before controller setup.
- `training.fed_every` controls aggregation cadence. `training.eval_every`
  schedules synchronized snapshots and the final round is always evaluated.
- `training.barrier_timeout_sec` bounds client ready/evaluation barrier waits.
- `split_server.training_strategy: concat_v1` is the current OUR path: matched
  client activations are concatenated into one server batch and cause one
  optimizer update. `sflv2_sequential_v1` instead serves clients in configured
  order through `round_end`, updating the shared server after every client
  batch. `split_server.model.gradient_accumulation_steps` is fixed at `1`;
  server gradients are never averaged across batches.
- `split_server.model.batch_timeout_sec` bounds incomplete split batches;
  waiting contributors receive a correlated error and fail into cancellation.
- `fed_server.min_clients` and `quorum_timeout_sec` define a bounded partial
  aggregation window. The completed global state is sent to accepted
  participants. A validated late update for that same completed round receives
  a correlated catch-up response without changing the completed aggregate;
  older rounds remain unsupported and are discarded.
- Partial quorum is arrival-window based and does not yet provide fairness or
  leases. `fedavg` uses uniform accepted-client weights; `weighted_fedavg` uses
  dataset-size weights for floating tensors. Non-floating buffers come from the
  largest accepted dataset. `aggregation_freq` must match
  `training.fed_every`, the single client/server synchronization cadence.
- Server channel references are validated against the controller's four
  canonical logical roles before setup.
- Only queue transport without compression is operational. gRPC/compression
  selections and unknown optimizer/noise names fail schema validation.

## Architecture and process boundaries

`src.main` creates `ConfigSchema`, seeds the process, constructs
`TrainingController`, calls `setup()`, and starts training.

`TrainingController.setup()` creates a multiprocessing manager and only the
channels and server roles owned by the selected mode. Centralized mode creates
one complete-model trainer without transport; federated creates only
`FedServer`; split creates only `SplitServer`; SplitFed creates both.

`TrainingController.start_training()` starts the selected roles, creates ready
and evaluation barriers for distributed modes, then starts one client process
per client configuration. Split clients send intermediate activations and
labels to the split server, which returns activation gradients. Federated
clients send validated model state and sample counts to the federated server,
which returns the configured aggregate.
Clients also send a typed `round_end` control message after their last local
batch so that peers with longer loaders are not blocked on an inactive client.
The split server validates channel sender identity, round/step correlation,
activation/label batch compatibility and duplicate steps before model use.

The process lifecycle has basic supervision but remains incomplete at this
stage. Client construction, training, evaluation and bounded barrier waits
share one worker failure boundary; failures set the shared stop event and abort
peer barriers. Remaining limitations include:

- first failures are structured, but typed cancellation delivery and recovery
  remain incomplete;
- joins use a polling loop and bounded terminate/kill fallback, but there is no
  recovery protocol;
- local queue receive waits participate in the shared cancellation event, but
  cancellation is not yet represented as a typed transport message;
- gRPC and message serialization are stubs.

Changes to process coordination require a multiprocessing smoke test, not only
an import test. `TrainingController` explicitly constructs its manager, queues
and processes from a `spawn` context; `src/main.py` also sets that method for
other multiprocessing code. The CLI attempts to stop every configured server
and always tears down the manager, including setup, training, stop and artifact
failures. Controller and server shutdown paths perform a final bounded join
after kill and raise if a child still remains alive.

## Known implementation hazards

- Feature extraction currently loads complete client datasets into memory.
- Stacked feature metadata records valid frame counts; train-only mask-aware
  normalization excludes padding and keeps normalized padded frames at zero.
- Actor-disjoint splits expose actor/sample/class counts and reject a training
  split missing either binary class. One-class test splits remain supported.
- Client training currently shuffles without class-balanced sampling.
- Dataset instances expose a bounded extraction report with discovered/loaded/
  failed counts and failure reason types; failed paths are not persisted.
- Checkpoint metadata does not include full configuration, seed, or optimizer
  state.
- `GrpcChannel` and message byte serialization are unimplemented.
- Capture and segmentation modules are incomplete and are not part of the
  supported training path.

Most planned research infrastructure in `docs/dev_plan` is not implemented
yet: there are no per-client Docker runtimes, ClientLoadController, scheduler
policy registry or heterogeneous-client simulator in the current runtime. A
typed profile registry exposes versioned `smoke` and `unit` default profiles;
the remaining named profiles are still planned. The mode matrix is executable
through `python -m src.experiments.mode_matrix`: centralized uses one complete
model and combined dataset view without channels, federated-only uses complete
client classifiers, split/shared uses one shared server model,
split/personalized owns one server model and optimizer per client, and SplitFed
combines split training with client-partition FedAvg. The matrix derives only
the roles and channels owned by each topology, validates every derived
configuration and writes a pass/fail and wall-time summary. SplitFed evaluates
the compatible pre-FedAvg encoder/server pair on aggregation rounds; the
aggregated encoder takes effect for the following round.

Treat each of these as a separate issue or commit. Avoid bundling lifecycle,
model correctness, data semantics, and packaging changes into one patch.

## Commit policy

Commits are allowed for this repository, but they must be atomic and follow
Conventional Commits. Use a single purpose and a narrow scope per commit, for
example:

```text
docs: add repository working guide
build: configure uv project metadata
fix: normalize dataset import paths
fix: make queue lifecycle cancellable
test: add dataset parser coverage
```

Before committing:

```bash
git diff --check
git status --short
git diff -- path/to/changed/files
```

Stage only files belonging to the atomic change. Do not include downloaded
corpora, generated logs, checkpoints, `.venv`, or unrelated user modifications.
Do not amend or squash another change unless explicitly requested.
