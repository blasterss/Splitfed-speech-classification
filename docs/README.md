# Documentation

Start with the project [README](../README.md) for installation and a first run.
The [repository guide](REPOSITORY_GUIDE.md) describes maintainer workflow.

The documentation separates implemented behavior from proposed work. Experiment
configurations may require external datasets and CUDA; their presence does not
establish scientific equivalence to a published method.

## Run and maintain

| Document | Purpose |
| --- | --- |
| [Project README](../README.md) | Installation, quick start and limitations |
| [Repository guide](REPOSITORY_GUIDE.md) | Module ownership, data layout, focused checks and commits |
| [Training configuration](runtime/CONFIGURATION.md) | Schema, devices, topology, workload policies and overrides |

## Implemented runtime

| Document | Purpose |
| --- | --- |
| [Runtime contracts](runtime/CONTRACTS.md) | Processes, messages, cancellation, artifacts and checkpoints |
| [Resource Metrics v2](runtime/RESOURCE_METRICS.md) | Owned footprint, phase memory, time and transport |
| [MergeSFL Algorithm 1](runtime/MERGESFL_ALGORITHM1.md) | Planner contract and explicitly reconstructed solver details |

## Research and future work

| Document | Purpose |
| --- | --- |
| [Experiment plan](EXPERIMENT_PLAN.md) | Implemented E0-E4 profiles and comparison rules |
| [E0 domain shift](experiments/E0_DOMAIN_SHIFT.md) | All-record MMD and relative normalized W1 protocol |
| [Development roadmap](development/ROADMAP.md) | Proposed evaluation, reliability, scheduling and deployment |
| [Agent instructions](../AGENTS.md) | Repository editing and validation rules |

Scientific manuscript sources live under `paper/`; generated experiment reports
under ignored `artifacts/` describe individual runs, not the current API contract.
The previous implementation plans for resources and MergeSFL have been replaced
by the runtime documents above. The deleted `docs/dev_plan` is superseded by the
short development roadmap; future capabilities are not presented as implemented.
