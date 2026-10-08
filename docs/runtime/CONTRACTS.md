# Runtime contracts

Mode validation, channel references, seed ownership, workload policies and
aggregation rules are documented in [training configuration](CONFIGURATION.md).
The owning modules are [application](../../src/application/dispatch.py),
[TrainingController](../../src/splitfed/controller/__init__.py),
[transport](../../src/transport/__init__.py) and
[checkpoint persistence](../../src/utils/persistence/checkpoint.py).

## Architecture and process boundaries

`src.main` resolves configuration, validates `ConfigSchema`, seeds the process
and dispatches through `src.application`. Local uses its experiment runner;
other training modes use `TrainingController`.

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
- channel receive waits participate in the shared cancellation event, but
  cancellation is not yet represented as a typed transport message;
- gRPC retry is bounded by the message deadline, but health RPCs, TLS/mTLS,
  authentication and distributed controller cancellation are not implemented.

Changes to process coordination require a multiprocessing smoke test, not only
an import test. `TrainingController` explicitly constructs its manager, queues
and processes from a `spawn` context; `src/main.py` also sets that method for
other multiprocessing code. The CLI attempts to stop every configured server
and always tears down the manager, including setup, training, stop and artifact
failures. Controller and server shutdown paths perform a final bounded join
after kill and raise if a child still remains alive.

## Local execution

The local cross-corpus runner avoids split/federated servers and channels.
It launches one fresh `LocalTrain` spawn process per source corpus, followed
by a separate `LocalEval` process. Training loads only the source corpus;
evaluation loads the target corpus views and reuses source training statistics.
Process isolation separates training resource peaks from cross-corpus evaluation.

## Artifact and checkpoint contracts

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

## Message correlation

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

## Checkpoint schema

Model files use checkpoint schema version 1 and record training mode, server
model scope and personalized client identity where applicable. Checkpoint writes
are atomic, and the loader rejects incompatible ownership plus mismatched tensor
keys, shapes and dtypes before returning a state dict.

If custom logging configuration does not create its parent directory, prepare
the log directory before running:

```bash
mkdir -p logs
```

## Ownership and known limits

Each mode creates only its required roles and channels. Queue and insecure
gRPC are supported for local processes; neither provides authenticated network
isolation or encryption. Checkpoints validate model ownership and tensor
schemas, but omit optimizer/RNG state and do not provide equivalent resumption.
Configuration and dataset provenance are separate metadata artifacts.

Feature extraction materializes each corpus in memory. Training splits are
actor-disjoint, and normalization excludes padded frames and uses training
statistics only. E0 intentionally uses all records and separate analysis
normalization. Extraction failures are counted by reason, not hidden.

The standalone file segmenter works with native-rate audio; streaming microphone
capture and an inference service are not implemented. There is no general
ClientLoadController, container topology, secure aggregation, formal privacy
accounting, fault simulator, or CI CUDA matrix. See the
[roadmap](../development/ROADMAP.md) for proposed work.

Resource interpretation is defined in [Resource Metrics v2](RESOURCE_METRICS.md).
