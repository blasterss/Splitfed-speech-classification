# Repository guide

Maintainer workflow for the current pre-alpha implementation. Start with the
[documentation index](README.md) and [project README](../README.md).

## Project identity

SplitFed Speech Emotion Classification is a pre-alpha research framework for
binary anger classification over acted emotional-speech corpora. The positive
class is anger (`ANG`); every other supported emotion is currently mapped to
zero. It is not a speech-to-text ASR system and should not be described as a
production privacy or security solution.

The distribution name, console command and protocol namespace remain
`secureasr` for compatibility. Treat them as technical identifiers rather than
the project or task name.

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

- `src/main.py`: CLI parsing and dispatch through `src/application/`.
- `src/application/`: configuration resolution, experiment dispatch, lifecycle
  and artifact persistence.
- `src/schema/`: Pydantic configuration models and enums.
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
- `src/transport/`: `Message`, strict protobuf serialization, queue and gRPC
  channels, generated protobuf contracts, and the channel factory.
- `src/utils/`: shared utilities; `persistence/` owns artifact paths,
  checkpoint envelopes, and state-dict byte serialization; `runtime/` owns
  process signal policy and structured failure publication; `training/` owns
  deterministic seed setup and per-round loss statistics; `config/` owns YAML
  configuration serialization.
- `configs/config.example.yaml`: portable CPU example configuration.
- `configs/config.real.yaml`: hardware-specific real-data research configuration;
  inspect its devices and budgets before use.
- `configs/logger.yaml`: logging configuration.
- `tests/`: tests grouped by owning component (`config/`, `dataset/`,
  `experiments/`, `model/`, `persistence/`, `runtime/`, `splitfed/`, and
  `transport/`); pytest discovers all groups from the repository-level path.
- `src/experiments/`: domain shift, local cross-corpus, checkpoint evaluation
  and topology smoke runners.
- `notebooks/`: exploratory work and the E0 analysis front end; Ruff excludes
  notebooks.
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

## Where to find detailed contracts

- [Configuration](runtime/CONFIGURATION.md): schema, overrides and topology validation.
- [Runtime](runtime/CONTRACTS.md): process ownership, lifecycle and artifacts.
- [Experiments](EXPERIMENT_PLAN.md): implemented E0-E4 profiles.
- [E0](experiments/E0_DOMAIN_SHIFT.md): analysis without training splits.
- [Resources](runtime/RESOURCE_METRICS.md): measurement and aggregation rules.
- [MergeSFL](runtime/MERGESFL_ALGORITHM1.md): implemented planner reconstruction.
- [Roadmap](development/ROADMAP.md): future architecture and acceptance criteria.

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
padding, models/FedAvg, queue and gRPC transports, and controller lifecycle. It
also has synthetic CPU `spawn` cycles covering unequal client steps, split
training, FedAvg, evaluation, clean process exit and state handoff. Run them
with `uv run pytest`. An automated external-data/CUDA matrix and CI workflow
are not present; report which checks actually ran for each change.

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
corpora, generated logs, checkpoints or unrelated user modifications. Do not
amend or squash another change without explicit authorization.
