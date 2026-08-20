# SecureASR Agent Instructions

## Mission

Treat this repository as a pre-alpha research prototype for binary emotional
speech classification with Split Federated Learning. The positive label is
anger (`ANG`); all other supported emotions currently map to zero.

Do not describe the project as conventional speech-to-text ASR, conflict
understanding, production infrastructure, cryptographically secure, or
differentially private unless the relevant mechanism and tests have actually
been implemented.

Read [README.md](README.md) and
[docs/REPOSITORY_GUIDE.md](docs/REPOSITORY_GUIDE.md)
for repository context. Use [docs/dev_plan](docs/dev_plan) for the intended
implementation sequence and target architecture.

## Working Rules

1. Inspect the owning module, its nearest caller, and the closest test before
   editing. State one falsifiable hypothesis and one cheap check that could
   disconfirm it.
2. Make the smallest focused change that tests the hypothesis. Preserve public
   APIs, label semantics, actor-disjoint splitting, relative package imports,
   and configured device defaults unless the task explicitly changes them.
3. After the first edit, run the narrowest relevant executable check before
   reading or changing unrelated code. Repair failures in the same slice and
   rerun that check.
4. Do not revert, overwrite, or reformat unrelated user changes in a dirty
   worktree. Never use destructive Git commands such as `git reset --hard` or
   `git checkout --`.
5. Do not commit or create branches unless explicitly requested. Keep commits
   atomic when the user asks for commits.
6. Prefer existing Pydantic, pathlib, multiprocessing, queue, and PyTorch
   patterns over new abstractions. Add an abstraction only when it removes
   meaningful duplication or defines a real protocol boundary.
7. Use ASCII for new source and documentation unless non-ASCII is required by
   the existing document or user-facing language.

## Product and Architecture Constraints

- Supported runtime: Unix/Linux, Python `>=3.10,<3.13`, managed with `uv`.
- Package entry points are `uv run secureasr` and `uv run python -m src.main`.
- The default queue transport simulates participants on one host. It provides
  no network isolation, authentication, TLS, secure aggregation, or formal DP.
- `src/main.py` owns CLI/config startup; `src/schema.py` owns configuration;
  `src/dataset/` owns discovery, parsing, features, and actor splits;
  `src/model/` owns neural models; `src/splitfed/` owns clients and processes;
  `src/transport/` owns message/channel contracts.
- `TrainingController` owns process creation, channels, barriers, shared stop
  state, supervision, and cleanup. Client and server workers must propagate
  failures instead of silently logging them.
- The target research topology has one isolated client runtime/container per
  client plus dedicated controller, load-controller, split-server, federated
  server, and transport services. Docker Compose is the reference development
  target; it is not by itself a production security guarantee.
- `ClientLoadController` owns admission, telemetry, leases, cohort selection,
  per-client budgets, deadlines, fairness debt, and quarantine. It must remain
  separate from model aggregation and expose replayable policy decisions.
- Keep multiprocessing `spawn` compatibility. Process coordination changes
  require a multiprocessing smoke test, not only an import test.
- The current gRPC channel and message byte serialization are stubs. Do not
  imply that selecting `grpc` makes the system operational.

## Non-Negotiable Invariants

- The configuration field is `models_save_path`, not `model_save_path`.
- Client IDs must be unique. Every client must have the four logical channels:
  `split_uplink`, `split_downlink`, `federated_uplink`, and
  `federated_downlink`.
- Validate channel references, client quorum, device availability, dataset
  roots, and cross-field constraints before spawning child processes.
- Never silently change `ANG` label mapping, actor IDs, actor-disjoint split,
  seed behavior, feature ordering, or CPU defaults.
- Never allow a client/server to wait indefinitely for a peer. Timeouts,
  cancellation, barrier abort, and process shutdown must form one protocol.
- Validate message sender, type, protocol identity, round, step, request ID,
  payload keys, tensor shapes, dtypes, and state-dict schemas before use.
- Do not load an unvalidated federated payload directly into `load_state_dict`.
- Make gradient accumulation, remainder flushing, client scaling, quorum, and
  aggregation frequency explicit. Do not restore hard-coded accumulation rules.
- Decide and document a policy for non-floating buffers and BatchNorm statistics;
  do not average them accidentally across non-IID clients.
- Evaluation must use `model.eval()`, `no_grad()`, and disabled training noise.
  Handle one-class and no-positive-prediction metrics explicitly.
- Do not claim privacy guarantees without a threat model, clipping/noise
  calibration, accounting, and empirical leakage tests.
- Keep `dataset_weight`, `compute_budget`, and `participation_credit` separate;
  large datasets and fast clients must not silently monopolize training.
- Scheduler policies must have stable names/versions and typed parameters.
  Never instantiate an arbitrary class from YAML.
- Debug, synthetic, and fault-injection modes must preserve validation,
  deadlines, authentication, and cancellation. Raw audio/tensor dumps require
  explicit opt-in, bounded size, redaction, and retention limits.

## Data and Experiment Safety

- Dataset archives, WAV files, logs, checkpoints, `.venv`, and local
  `configs/config.yaml` stay outside commits.
- Treat skipped or failed feature extraction as measurable data loss: collect
  reasons and counts rather than hiding systematic failures.
- Preserve train-only normalization. Padding must not distort statistics; use a
  mask-aware or fixed-duration policy when changing this behavior.
- Record resolved configuration, seed, environment, dataset manifest, actor and
  class counts, model shapes, metrics, and checkpoint metadata for experiments.
- Use reduced datasets, few rounds, and CPU for smoke tests. Do not launch a
  full dataset/CUDA run as a validation shortcut.
- Prefer named research profiles such as `unit`, `smoke`, `debug`,
  `fault_injection`, `heterogeneous`, `benchmark`, `repro`, `privacy`, and
  `distributed`. Save the resolved profile, CLI overrides, policy versions,
  seed tree, simulator parameters, and environment/image digest.
- For scheduler work, compare at least all-client, fixed/random, throughput,
  throughput-plus-fairness, data-balanced, and availability-aware policies
  under matched seeds and budgets. Report worst-client quality, fairness,
  participation frequency, dataset coverage, latency, throughput, bytes, and
  convergence.
- Keep lifecycle, protocol, numerical correctness, data semantics, privacy,
  transport, and packaging changes in separate focused slices when practical.

## Validation Commands

Run commands from the repository root. Install the development environment with:

```bash
uv sync --extra train --extra dev
```

For changed Python files, run the focused checks first:

```bash
uv run ruff check path/to/changed.py
uv run black --check path/to/changed.py
uv run python -m py_compile path/to/changed.py
```

Then run the relevant tests, followed by the full suite when the change is
shared or cross-process:

```bash
uv run pytest tests/test_relevant_area.py
uv run pytest
```

For schema or configuration changes, also validate the example configuration:

```bash
uv run python -c "from src.schema import ConfigSchema; from src.utils.common import read_yaml; ConfigSchema(**read_yaml('configs/config.example.yaml')); print('schema-ok')"
```

For controller, channel, barrier, or worker changes, require a synthetic CPU
smoke test with `spawn`, failure injection where relevant, and confirmation that
no child processes remain alive. For model or FedAvg changes, add a small
deterministic reference test for shapes, gradients, buffers, or aggregation.
For container or scheduler changes, also run the isolated Compose smoke path or
the narrowest available simulator/policy test, including health probes,
deadlines, quorum, lease expiry, slow/dropout clients, and graceful shutdown.

Before handing work back, run:

```bash
git diff --check
git status --short
git diff -- path/to/changed/files
```

Report commands that could not run, missing datasets, CUDA limitations, and
remaining unrelated failures explicitly.

## Documentation and Change Handoff

Update documentation when behavior, configuration, supported transport,
privacy claims, artifact format, or operational commands change. Keep
`README.md` and `docs/REPOSITORY_GUIDE.md` factual; do not document planned
features as implemented.

When documentation changes, cross-check all four project contracts:
`README.md` for user-facing current behavior, `docs/REPOSITORY_GUIDE.md` for
maintainer workflow, `docs/dev_plan` for future architecture, and this file for
agent behavior. Keep planned Docker, scheduler, simulation, and policy work
clearly separated from the current runtime in the first two documents.

The final handoff should briefly state:

- what changed and why;
- files touched;
- focused and broader checks run, with outcomes;
- known limitations, skipped tests, or required external datasets;
- whether generated artifacts or user changes were intentionally left alone.