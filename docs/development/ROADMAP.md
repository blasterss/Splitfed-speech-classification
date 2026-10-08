# Development roadmap

This document describes proposed work. It does not announce implemented
capabilities. Current behavior is documented in the
[runtime contracts](../runtime/CONTRACTS.md); research profiles are described in
the [experiment plan](../EXPERIMENT_PLAN.md).

## Evaluation and reproducibility

1. Extend the shared complete-model binary evaluator to composed split checkpoints.
2. Verify `concat_v1`, `sequential_v1` and personalized aggregation against small
   deterministic numerical references, including optimizer-state semantics.
3. Automate matched actor folds, seeds, effective/repeated samples and multi-seed
   summaries. Keep protocol-faithful comparisons distinct from controlled ablations.
4. Add actor-level E0 sensitivity and leave-one-corpus-out evaluation without
   held-out normalization or model selection leakage.
5. Generate paper tables from versioned artifacts; do not mix resource schema v1
   with v2 or historical E0 protocols with the current one.

## Runtime reliability and memory

1. Unify channel, quorum, barrier and shutdown deadlines under a typed
   cancellation contract; define failed/completed/aborted run state.
2. Add seeded delay/dropout/straggler scenarios and specify what an interrupted
   optimizer step may have already changed. Bounded shutdown is not rollback.
3. Extend checkpoints with optimizer/RNG/round/config/manifest state and prove
   uninterrupted-versus-resumed numerical equivalence.
4. Bound corpus memory use and decide an explicit class-balanced sampling policy.
5. Add a simultaneous host memory sampler; per-process peaks cannot establish
   concurrent host usage.

## Scheduling and isolated deployment

The current controller embeds the MergeSFL Algorithm 1 planner. A proposed
standalone `ClientLoadController` would own admission, telemetry windows,
leases, per-client budgets, fairness debt and quarantine. Aggregation stays with
model servers. Keep dataset weight, compute budget and participation credit
separate, and persist replayable decisions with versioned typed policies.

A heterogeneous simulator should compare all-client, fixed/random, throughput,
throughput-plus-fairness, data-balanced and availability-aware policies under
matched seeds/budgets. Report worst-client quality, coverage, participation,
latency, bytes and convergence alongside average throughput.

Proposed deployment has isolated client runtimes plus controller, load-controller,
split-server, federated-server and transport services. A non-root Docker Compose
reference needs health checks, deadlines, cancellation and graceful shutdown.
Compose alone does not establish security. Neither this topology nor a general
scheduler registry is currently implemented.

## Privacy and validation

TLS/authentication and privacy mechanisms require explicit threat models,
validation and leakage evaluation. Noise without calibration/accounting is not
formal DP. Add reproducible CPU CI and optional dataset/CUDA gates before treating
manual hardware runs as acceptance evidence. Future debug/fault profiles must
preserve validation and bounded lifecycle handling.
