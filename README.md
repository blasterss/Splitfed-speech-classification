# SplitFed Speech Emotion Classification

Pre-alpha research framework for binary anger classification with local,
centralized, federated, split-learning and SplitFed experiments over CREMA-D,
RAVDESS and SAVEE. Anger (`ANG`) maps to `1`; other supported emotions map to `0`.
The installed package, CLI and transport namespace retain the legacy identifier
`secureasr`. The research task is speech emotion classification.

## Quick start

Use Unix/Linux and Python 3.10-3.12. Run commands from the repository root:

```bash
uv venv --python 3.12
uv sync --extra train --extra dev
cp configs/config.example.yaml configs/config.yaml
mkdir -p logs artifacts
```

Set dataset roots in the local configuration. External WAV files are not bundled;
the example expects datasets under `../datasets`. Use CPU, reduced datasets and
one round for the first smoke run:

```bash
uv run secureasr --config-file configs/config.yaml \
  --set training.num_rounds=1 \
  --set training.eval_every=1 \
  --set training.fed_every=1 \
  --set fed_server.aggregation_freq=1
```

These overrides change cadence only. Configure every participant device as CPU
and set each `clients[].dataset.reduced` before launching the smoke run.
The equivalent CLI is `uv run python -m src.main`. Typed overrides must name
existing fields; omit server overrides for configurations without those roles.

PyTorch belongs to the `train` extra. Linux/Windows resolve the pinned CUDA 12.8
wheel; CPU execution remains supported. macOS uses the PyPI wheel. CUDA runs
require a compatible NVIDIA driver; verify availability before selecting CUDA.
See [configuration](docs/runtime/CONFIGURATION.md) for runtime and topology rules.

## Implemented modes

| Mode | Model ownership |
| --- | --- |
| Local | One independent complete model per corpus; separate training/evaluation processes |
| Centralized | One complete model over combined training corpus views |
| Federated | Complete client models plus FedServer aggregation |
| Split/shared | Client partitions plus one shared server model |
| Split/personalized | One client/server pair per client, isolated server optimizers |
| SplitFed | Split training plus client aggregation; personalized scope also aggregates server models |

Shared server strategies are `concat_v1`, `sequential_v1`, static `mergesfl_v1`,
and reconstructed `mergesfl_algorithm1_v1`. These are different update protocols,
not interchangeable names for the same method. Personalized SplitFed is an
implemented experimental variant; canonical SFLv1 equivalence is not established.

## Experiments and documentation

- [Documentation index](docs/README.md): navigation by purpose.
- [Repository guide](docs/REPOSITORY_GUIDE.md): code ownership and validation.
- [Training configuration](docs/runtime/CONFIGURATION.md): modes, devices,
  channels, workload policies and overrides.
- [Runtime contracts](docs/runtime/CONTRACTS.md): processes, messages,
  artifacts and checkpoints.
- [Experiment plan](docs/EXPERIMENT_PLAN.md): implemented E0-E4 experiments.
- [E0 domain shift](docs/experiments/E0_DOMAIN_SHIFT.md): MMD raw/local normalized
  at n=480, normalized W1 between/within at n=240, 50 repetitions and ratio R.
- [Resource Metrics v2](docs/runtime/RESOURCE_METRICS.md): ownership, phase memory
  peaks, timing and transport interpretation.
- [MergeSFL Algorithm 1](docs/runtime/MERGESFL_ALGORITHM1.md): reconstruction
  boundaries, plan enforcement and remaining verification.
- [Development roadmap](docs/development/ROADMAP.md): future work, separate
  from implemented behavior.

## Data and evaluation

Training uses actor-disjoint splits and mask-aware train-only normalization.
Complete-model cross-corpus evaluation reuses the source training statistics.
E0 analyzes all extracted records without a split and fits its analysis transform
independently per corpus. Extraction losses, actor/class coverage and seeds must
accompany results. SAVEE has only four actors; record-level E0 inference ignores
actor dependence. Acted-corpus results do not establish performance on real
conversations. Multi-seed and held-out-corpus validation remain necessary for
strong generalization claims.

Generated artifacts, WAV files, logs, checkpoints, `.venv` and local
`configs/config.yaml` are excluded from commits. Model checkpoints are atomic,
versioned and ownership-validated; optimizer/RNG resumption is not implemented.

## Validation

```bash
uv run pytest
```

The suite includes configuration, data, numerical, artifact and protocol tests,
plus synthetic CPU `spawn` tests for local processes and queue/gRPC transport.
CUDA and external dataset checks require the corresponding environment; synthetic
smoke tests do not replace them. Use the focused commands in the repository guide
before broad validation.

## Known limitations

The supported training topology runs on one Unix host. Queue and insecure gRPC
provide no TLS, authentication, secure aggregation or participant isolation.
Activation noise provides no formal differential privacy guarantee. Server
activations and labels can disclose information; privacy claims require a threat
model, calibrated mechanisms, accounting and leakage evaluation.

Cancellation and timeout handling are bounded but do not provide fault recovery
or transactional rollback of every interrupted step. Partial quorum is an
arrival-window policy, not a fairness scheduler. Floating BatchNorm statistics
are averaged under the configured aggregation policy; FedBN is not implemented.
Datasets are materialized in memory. There is no working streaming inference
service, distributed container deployment or automated CUDA CI matrix.

## License

MIT; see [LICENSE](LICENSE).
