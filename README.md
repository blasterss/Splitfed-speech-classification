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
- synchronous split-learning forward/backward steps;
- explicit per-client round completion for unequal local loader lengths;
- weighted FedAvg for the client-side model;
- optional Gaussian or Laplace perturbation of intermediate activations;
- local evaluation with accuracy, F1, precision and recall.

The following are declared or partially scaffolded but are **not implemented as
working features**:

- gRPC transport and message serialization;
- secure aggregation;
- formal differential privacy accounting;
- streaming microphone inference;
- fault-tolerant or asynchronous clients;
- a production inference service.

## Architecture

```text
Client process
  audio features
      -> ClientSideModel
      -> intermediate activations + labels
      -> SplitServer

SplitServer process
  activations from all clients for the same (round, step)
      -> ServerSideModel
      -> BCEWithLogitsLoss
      -> activation gradients returned to each client

Federated server process
  client-side state_dict + local dataset size
      -> weighted FedAvg
      -> global client-side state_dict returned to every client

TrainingController
  creates queues, barriers and processes
  starts clients and servers
  supervises child exit codes and coordinates shutdown
```

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
  dataset/                audio loading and feature extraction
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

For an NVIDIA machine, install the CUDA wheel selected by the official PyTorch
installation selector using `uv pip` after creating the environment. Verify the
runtime before loading the datasets:

```bash
uv run python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)'
```

## Configuration

Copy the example without committing the local file:

```bash
cp configs/config.example.yaml configs/config.yaml
```

Then update at least:

- every `clients[].dataset.root`;
- `models_save_path` (note the plural form expected by the current schema);
- client and split-server devices;
- batch size and local steps for available GPU memory;
- channel timeouts;
- the noise configuration, if perturbation is required.

Relative paths are resolved from the repository root when the command is run
there. The example expects the downloaded datasets in `../datasets`.

Important configuration caveats:

- unknown fields are rejected at every configuration level;
- `training.fed_every` currently controls federated synchronization;
- `training.barrier_timeout_sec` bounds client ready/evaluation barriers;
- `split_server.model.gradient_accumulation_steps` controls how many server
  batches are averaged per optimizer update; each round flushes its remainder;
- `training.eval_every` is defined but not used by the training loop;
- `fed_server.strategy`, `aggregation_freq` and `min_clients` are currently not
  honoured by the worker;
- channel names stored inside server configuration are not used by the
  controller, which relies on fixed names;
- client IDs must be unique; referenced server channels and feasible client
  quorum are validated before controller setup.

The repository also contains a forward-looking research plan for named launch
profiles, simulation, isolated client containers and throughput-aware client
scheduling. Those capabilities are planned, not part of the current runtime.
The planned training-mode matrix also includes federated-only training and
split-only training with either a shared or personalized server model.

## Running

From the repository root:

```bash
uv run secureasr --config-file configs/config.yaml
```

The equivalent module invocation is:

```bash
uv run python -m src.main --config-file configs/config.yaml
```

Start with reduced datasets and a small number of rounds. A full CUDA run should
only be attempted after all clients load successfully and a one-round smoke test
has completed.

Expected generated artifacts include:

- rotating logs under `logs/` when logging configuration is loaded;
- evaluation CSV files under `experiments/results/`;
- server and global client checkpoints under the configured model directory.

Some output directories are not created consistently by the current code. Create
them before running if necessary:

```bash
mkdir -p logs experiments/results checkpoints
```

## Verification status

The repository has focused tests for schema/configuration, dataset parsers,
padding, models/FedAvg, queue transport and controller lifecycle. Run them with:

```bash
uv run pytest
```

There is no CI workflow yet, and the suite does not replace a real-data,
multi-process or CUDA smoke test. Use a reduced dataset and one round before
starting a long experiment.

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
server receives complete client model parameters. Local queues provide no
cryptographic isolation, and the gRPC/TLS path is not implemented. Intermediate
activations may still reveal speaker identity, recording domain or speech
content.

The optional `PrivacyLayer` adds random noise, but it is **not a complete
differential privacy implementation**:

- clipping is not exposed through configuration and is disabled by default;
- no epsilon/delta accounting is performed;
- the example noise level is too small to represent meaningful protection;
- noise is currently also applied during evaluation;
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
  or malformed completion message can still leave a partial batch until its
  timeout, which is discarded without a correlated error response.
- The controller has a polling supervision loop and propagates non-zero child
  exit codes. Client initialization and barrier failures now set the shared
  stop event and abort peer barriers, but queue timeouts and server failures are
  not yet one complete cancellation protocol.
- Split and federated server exit codes are monitored, but failure reporting and
  recovery are not yet fault-tolerant or restartable.
- Queue timeouts can still leave peers waiting in some failure paths.

### Numerical correctness

- Server gradient accumulation is configurable and remainder gradients are
  flushed at the end of each completed round. Client activation gradients use
  the full batch loss; only accumulated server parameter gradients are averaged.
- FedAvg does not treat floating parameters and integer BatchNorm counters
  separately.
- BatchNorm statistics from non-IID clients are averaged without an explicit
  policy.
- Optimiser state is not federated or restored with model checkpoints.

### Messages and aggregation

- Client responses are not validated against sender, type, round and step.
- Federated client/global updates validate sender, type, round, step, keys,
  tensor shapes and dtypes before aggregation or `load_state_dict`.
- `min_clients`, aggregation strategy and aggregation frequency are configured
  but not fully honoured by the worker.
- Metrics are logged locally and are not collected by the federated worker.

### Data pipeline

- Feature extraction loads each client dataset fully into memory.
- Padding zeros participate in normalisation statistics.
- The actor split seed is hard-coded rather than taken from configuration.
- Split class coverage is not validated.
- A `WeightedRandomSampler` is created but not used by the training loader.
- Per-file feature extraction errors are skipped and can hide systematic data
  loss.
- Spectral-contrast failures fall back to zero features.
- Multi-channel utilities are incomplete and inconsistent with the main path.

### Evaluation and artifacts

- `eval_every` is unused; evaluation occurs only after training.
- Noise remains active in the client model during evaluation.
- Undefined precision/recall cases are not handled explicitly.
- Evaluation and checkpoint directories are not always created before writing.
- Checkpoints do not contain configuration, optimiser state, seed or dataset
  manifest.
- The experiment cannot currently be resumed reliably.

### Packaging and unfinished modules

- Imports mix package-relative, `src.*` and top-level forms.
- The microphone capture module contains undefined constants and empty methods.
- The audio segmenter calls the audio loader with an incompatible signature.
- Message byte serialization and `GrpcChannel` are stubs.
- There is no CI workflow, multi-process spawn smoke suite, checkpoint round-trip
  suite or CUDA test matrix yet.

## Recommended improvement plan

### 1. Make startup deterministic

- normalise package imports and add a stable module/console entry point;
- validate CUDA availability;
- create artifact directories centrally;
- save the resolved configuration and environment metadata per run.

### 2. Introduce process supervision

- use one shared cancellation event and a structured error channel;
- monitor all client and server exit codes concurrently;
- unify bounded joins, queue waits and server cancellation;
- propagate child failures to the command exit code.

### 3. Formalise the training protocol

- define typed payloads and request IDs;
- validate sender, message type, round, step, tensor shape and dtype;
- send explicit error responses for rejected or timed-out batches;
- define how clients with different dataset sizes participate in a round;
- separate training, evaluation and control states.

### 4. Correct optimisation and FedAvg

- implement the configured aggregation strategy and minimum-client policy;
- handle non-floating buffers explicitly;
- decide between local and global BatchNorm statistics;
- aggregate evaluation metrics in the controller or a dedicated evaluator.

### 5. Strengthen data and privacy experiments

- use masked normalisation or fixed-duration segmentation;
- validate class and actor coverage;
- use configuration seeds and run multiple seeds;
- either use the weighted sampler or remove it;
- disable perturbation during evaluation;
- add clipping, calibrated noise and privacy accounting if DP is claimed;
- measure reconstruction, membership and attribute leakage;
- compare centralised, federated, split and SplitFed baselines.

### 6. Add verification

- unit tests for schemas, parsers, padding, models and FedAvg;
- message contract and timeout tests;
- multiprocessing tests using the Windows `spawn` method;
- a synthetic CPU smoke test and a separate CUDA smoke test;
- checkpoint round-trip and deterministic-gradient tests;
- Ruff, Black, mypy and pytest in CI.

## Experimental roadmap

A meaningful study should compare:

1. centralised training;
2. federated-only training;
3. split-only training;
4. SplitFed training;
5. SplitFed under several clipping/noise settings.

Measure classification quality, per-client quality, communication volume, round
latency, peak GPU memory and privacy leakage. Vary the split point and evaluate
leave-one-dataset-out generalisation. This would turn the prototype into a
reproducible research platform rather than only a multiprocessing demonstration.

## License

This project is distributed under the MIT License. See [LICENSE](LICENSE).
