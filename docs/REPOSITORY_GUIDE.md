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
- `src/dataset/processors/`: CREMA-D, RAVDESS, and SAVEE loaders registered in
  `DatasetLoaderFactory`.
- `src/model/`: client-side and server-side PyTorch models.
- `src/splitfed/`: clients, split server, federated server, and
  `TrainingController`.
- `src/transport/`: `Message`, queue channel, channel factory, and the
  unimplemented gRPC channel.
- `src/utils/`: YAML helpers, training utilities, statistics, and feature
  aggregation utilities.
- `configs/config.example.yaml`: portable CPU example configuration.
- `configs/logger.yaml`: logging configuration.
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
uv run python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)'
```

Run the application:

```bash
uv run secureasr --config-file configs/config.yaml
# equivalent:
uv run python -m src.main --config-file configs/config.yaml
```

Create local configuration and output directories:

```bash
cp configs/config.example.yaml configs/config.yaml
mkdir -p logs experiments/results checkpoints
```

Use a reduced dataset, few rounds, and CPU for the first smoke test. Do not
start a full training run until every client has loaded successfully and a
one-round run completes.

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
uv run python -c "from src.schema import ConfigSchema; from src.utils.common import read_yaml; ConfigSchema(**read_yaml('configs/config.example.yaml')); print('schema-ok')"
```

The repository has focused tests for schemas/configuration, dataset parsers,
padding, models/FedAvg, queue transport and controller lifecycle. Run them with
`uv run pytest`. A real-data multi-process smoke test, CUDA matrix and CI
workflow are still missing.

## Configuration invariants

- Unknown fields are rejected in root and nested configuration models.
- The root field is `models_save_path` (plural), not `model_save_path`.
- Client IDs must be unique. Server channel references and `min_clients` versus
  configured client count are validated before controller setup.
- Every client needs all four fixed channel names:
  `split_uplink`, `split_downlink`, `federated_uplink`, and
  `federated_downlink`.
- Dataset roots must exist before `ConfigSchema` validation.
- Client, split-server, and federated-server device choices must match the
  installed runtime.
- `training.fed_every` is used by training; `training.eval_every` is currently
  not used by the training loop.
- `training.barrier_timeout_sec` bounds client ready/evaluation barrier waits.
- `split_server.model.gradient_accumulation_steps` controls averaged server
  optimizer updates. Client activation gradients are not scaled by this value,
  and incomplete accumulation windows flush at the completed round boundary.
- `fed_server.strategy`, `aggregation_freq`, and `min_clients` are currently
  configured but not fully honoured by the worker.
- The controller uses fixed channel constants rather than the channel names
  stored in server configuration.

## Architecture and process boundaries

`src.main` creates `ConfigSchema`, seeds the process, constructs
`TrainingController`, calls `setup()`, and starts training.

`TrainingController.setup()` creates a multiprocessing manager, one set of
queue channels per client, a `SplitServer`, and a `FedServer`.

`TrainingController.start_training()` starts both servers, creates ready and
evaluation barriers, then starts one client process per client configuration.
Clients send intermediate activations and labels to the split server. The split
server returns activation gradients. Clients send client model state and sample
counts to the federated server, which returns the aggregated state.
Clients also send a typed `round_end` control message after their last local
batch so that peers with longer loaders are not blocked on an inactive client.

The process lifecycle has basic supervision but remains incomplete at this
stage. Client construction, training, evaluation and bounded barrier waits
share one worker failure boundary; failures set the shared stop event and abort
peer barriers. Remaining limitations include:

- server exit codes are checked, but recovery and structured error propagation
  are incomplete;
- joins use a polling loop but do not yet implement a complete cancellation
  protocol;
- queue timeouts and server shutdown are not one unified cancellation protocol;
- gRPC and message serialization are stubs.

Changes to process coordination require a multiprocessing smoke test, not only
an import test. Keep `spawn` compatibility in mind because `src/main.py` sets
that start method explicitly.

## Known implementation hazards

- Feature extraction currently loads complete client datasets into memory.
- Padding zeros can affect normalization statistics.
- Actor split uses a hard-coded random state rather than the configured seed.
- Class coverage is not validated before training.
- A weighted sampler is created but not used by the training loader.
- Noise is currently active during evaluation.
- Precision and recall edge cases are not handled explicitly.
- Checkpoint metadata does not include full configuration, seed, or optimizer
  state.
- `GrpcChannel` and message byte serialization are unimplemented.
- Capture and segmentation modules are incomplete and are not part of the
  supported training path.

The planned research infrastructure in `docs/dev_plan` is not implemented yet:
there are no per-client Docker runtimes, ClientLoadController, named policy
registry, launch profiles or heterogeneous-client simulator in the current
runtime. The same applies to the planned training-mode matrix: federated-only,
split-only shared-server and split-only personalized-server execution paths
are design targets, not current entry points.

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
