# MergeSFL Algorithm 1: implementation design

## Status and scope

The `mergesfl_algorithm1_v1` runtime implements the control and execution
contracts described below. The separate `mergesfl_v1` runtime remains the
static feature-merging and gradient-dispatch baseline from the public MergeSFL
code. The Algorithm 1 GA and integer refinement are explicitly versioned
reconstructions because their implementation details are absent upstream.

The policy belongs to `ClientLoadController`, not to `SplitServer` or
`FedServer`. The controller produces a replayable `RoundPlan`; model workers
only execute a validated plan.

## Typed inputs

For every round `h`, the policy consumes:

- `WorkerProfile(client_id, label_distribution, participation_count)`;
- `WorkerTelemetry(client_id, compute_seconds_per_sample,
  transfer_seconds_per_sample, observed_at, sample_count)`;
- `MergeSFLPolicyConfig(max_batch_size, local_steps, ema_alpha,
  ingress_budget, feature_bytes_per_sample, kl_threshold, min_clients,
  max_clients, ga_seed, population_size, generations)`;
- the previous estimates `mu[i]` and `beta[i]`.

Label distributions contain non-negative finite probabilities, sum to one and
use the global class order. Telemetry has a bounded age. Missing, stale or
non-positive measurements make a worker ineligible; values are never silently
imputed from another worker.

## Policy pipeline

### 1. State estimation (Equations 5 and 6)

For eligible worker `i`:

```text
mu_h[i]   = alpha * mu_(h-1)[i]   + (1 - alpha) * observed_compute[i]
beta_h[i] = alpha * beta_(h-1)[i] + (1 - alpha) * observed_transfer[i]
cost[i]   = mu_h[i] + beta_h[i]
```

The first valid observation initializes the estimate directly. Estimates and
their source observations are written to the decision artifact.

### 2. Initial batch regulation (Equations 7--9)

Let `l` be the fastest worker, with the smallest positive `cost`. Assign
`batch[l] = D`, where `D` is `max_batch_size`. For every worker:

```text
batch[i] = max(1, floor(D * cost[l] / cost[i]))
duration[i] = local_steps * batch[i] * cost[i]
```

Ties are resolved by ascending client ID. A configured per-client memory limit
may only reduce the result and must be recorded as a constraint.

### 3. Bandwidth feasibility (Equation 10)

A cohort is feasible iff:

```text
sum(batch[i] * feature_bytes_per_sample for i in cohort) <= ingress_budget
```

The unit is bytes per training iteration. Configuration parsing must reject an
ingress budget that cannot admit `min_clients` with batch size one.

### 4. Selection priority (Equation 13)

```text
priority[i] = sum(K[j] + 1 for j in eligible) / (K[i] + 1)
```

Priority affects initial GA sampling, not the objective. Participation counts
increase only after a completed round, never when a lease is merely issued.

### 5. Cohort objective (Equations 11 and 12)

For cohort `S`, its merged label distribution is:

```text
phi_S = sum(batch[i] * V[i] for i in S) / sum(batch[i] for i in S)
```

The reference distribution `phi_0` is the unweighted mean of all eligible
worker distributions, matching the paper. The primary objective is
`KL(phi_S || phi_0)`. Zero probabilities use a configured epsilon only inside
the logarithm; the epsilon and normalization procedure are persisted.

### 6. Genetic search

The paper does not specify mutation, crossover or stopping details. The first
implementation therefore exposes a versioned `binary_ga_v1` policy:

1. encode one bit per eligible worker;
2. seed the initial population using weighted sampling by priority;
3. repair candidates to `min_clients..max_clients` and bandwidth feasibility;
4. rank by `(KL, unused_bandwidth, cohort_client_ids)`;
5. use seeded tournament selection, single-point crossover and bit mutation;
6. retain the best candidate (elitism) for a fixed number of generations.

Every random operation uses `SeedSequence(experiment_seed, round, ga_seed)`.
The artifact records the initial population, best score per generation and
final candidate. This is a documented reconstruction, not code copied from the
public repository.

### 7. Batch refinement (Equation 14)

The paper describes a Lagrange-dual refinement but does not publish the
solver. Implement a separate versioned `integer_refinement_v1`:

- start from Equation 9 batch sizes;
- search integer increments/decrements within `[1, D]`;
- preserve bandwidth feasibility;
- require `KL(phi_S || phi_0) <= kl_threshold` when feasible;
- minimize mean added duration, then KL, then unused bandwidth;
- proportionally scale the final vector to consume remaining bandwidth and
  repair deterministically.

The result must report whether the KL constraint was feasible. An infeasible
round fails planning or follows an explicitly configured fallback; it must not
silently relax the threshold.

## RoundPlan contract

```text
RoundPlan(
  policy_name="mergesfl_algorithm1_v1",
  round,
  seed,
  cohort,
  batch_size_by_client,
  local_steps,
  required_quorum,
  deadline_at,
  model_version,
  estimates,
  bandwidth_used,
  reference_distribution,
  merged_distribution,
  kl_divergence,
  decision_trace,
)
```

Clients rebuild their training DataLoader from the plan with `drop_last=True`.
Only cohort members contribute train steps and model updates. Non-participants
send typed round-completion and model-sync requests so synchronized evaluation
uses the same global client-side model. The split server merges exactly the
planned clients for every step.
The federated server weights bottom-model states by planned batch size as in
Equation 17. A dropout aborts the synchronous merged step; it must not produce
a partial server update.

## Implemented contracts and remaining experimental acceptance

1. Implemented: policy math, deterministic GA/refinement, strict KL failure,
   typed plans, dynamic loaders, server-side plan enforcement, Equation 17,
   telemetry/decision artifacts and a multi-process selected/skipped smoke.
2. Remaining experimental validation: calibrated bootstrap timings, explicit
   slow/dropout fault injection, repeated-sample and waiting-time accounting,
   and a matched multi-seed benchmark with confidence intervals.
3. Matched experiment requires identical seed, actor folds, model, loss,
   evaluation
   cadence and total effective-sample budget for `sequential_v1`, `concat_v1`,
   repo-faithful `mergesfl_v1` and `mergesfl_algorithm1_v1`.

## Safety boundaries

- Algorithm 1 does not add privacy, authentication or secure aggregation.
- Label distributions are sensitive metadata and must not be logged at
  per-sample granularity.
- Debug and synthetic profiles keep validation, deadlines and cancellation.
- The repo-faithful static baseline remains separate from the reconstructed
  Algorithm 1 policy so experimental results cannot conflate them.
