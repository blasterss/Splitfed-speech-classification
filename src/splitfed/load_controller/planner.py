"""Deterministic MergeSFL Algorithm 1 round planner."""

import random
import time
from dataclasses import dataclass

from ...schema import MergeSFLPolicyConfig
from .mergesfl import (
    ClientId,
    WorkerProfile,
    WorkerState,
    WorkerTelemetry,
    bandwidth_usage,
    estimate_worker_state,
    initial_batch_sizes,
    kl_divergence,
    merged_label_distribution,
    reference_label_distribution,
    selection_priorities,
)


@dataclass(frozen=True)
class CohortSelection:
    cohort: tuple[ClientId, ...]
    batch_sizes: dict[ClientId, int]
    merged_distribution: tuple[float, ...]
    reference_distribution: tuple[float, ...]
    kl: float
    bandwidth_used: int
    trace: dict


class MergeSFLPlanner:
    """Stateful Algorithm 1 planner owned by the training controller."""

    def __init__(self, config: MergeSFLPolicyConfig, experiment_seed: int):
        self.config = config
        self.experiment_seed = experiment_seed
        self.estimates: dict[ClientId, WorkerState] = {}

    def plan(
        self,
        profiles: list[WorkerProfile],
        telemetry: list[WorkerTelemetry],
        *,
        round_idx: int,
        model_version: str,
        now: float | None = None,
    ):
        """Create a validated, replayable plan from current observations."""
        from .mergesfl import RoundPlan

        current_time = time.time() if now is None else now
        telemetry_by_id = {item.client_id: item for item in telemetry}
        eligible_profiles = []
        next_estimates = {}
        rejected = {}
        for profile in sorted(profiles, key=lambda item: str(item.client_id)):
            observation = telemetry_by_id.get(profile.client_id)
            if observation is None:
                rejected[str(profile.client_id)] = "missing_telemetry"
                continue
            age = current_time - observation.observed_at
            if age < 0 or age > self.config.telemetry_max_age_sec:
                rejected[str(profile.client_id)] = "stale_telemetry"
                continue
            estimate = estimate_worker_state(
                observation.state,
                self.estimates.get(profile.client_id),
                alpha=self.config.ema_alpha,
            )
            eligible_profiles.append(profile)
            next_estimates[profile.client_id] = estimate
        if len(eligible_profiles) < self.config.min_clients:
            raise ValueError("not enough eligible clients for MergeSFL round")

        selection = select_cohort_binary_ga(
            eligible_profiles,
            next_estimates,
            self.config,
            experiment_seed=self.experiment_seed,
            round_idx=round_idx,
        )
        refined = refine_batch_sizes(
            [
                profile
                for profile in eligible_profiles
                if profile.client_id in selection.cohort
            ],
            selection.batch_sizes,
            next_estimates,
            selection.reference_distribution,
            self.config,
        )
        selected_profiles = [
            profile
            for profile in eligible_profiles
            if profile.client_id in selection.cohort
        ]
        merged = merged_label_distribution(selected_profiles, refined)
        used = bandwidth_usage(
            selection.cohort,
            refined,
            self.config.feature_bytes_per_sample,
        )
        self.estimates.update(next_estimates)
        return RoundPlan(
            round=round_idx,
            seed=self.experiment_seed,
            cohort=selection.cohort,
            batch_size_by_client=refined,
            local_steps=self.config.local_steps,
            required_quorum=len(selection.cohort),
            deadline_at=current_time + self.config.round_timeout_sec,
            model_version=model_version,
            estimates={
                item: next_estimates[item] for item in selection.cohort
            },
            bandwidth_used=used,
            reference_distribution=selection.reference_distribution,
            merged_distribution=merged,
            kl_divergence=kl_divergence(
                merged, selection.reference_distribution
            ),
            decision_trace={
                **selection.trace,
                "rejected": rejected,
                "initial_batch_sizes": selection.batch_sizes,
                "refined_batch_sizes": refined,
            },
        )


def refine_batch_sizes(
    profiles,
    initial_batches,
    states,
    reference,
    config,
) -> dict[ClientId, int]:
    """Deterministically refine integer batches under KL and bandwidth."""
    batches = dict(initial_batches)

    def score(candidate):
        merged = merged_label_distribution(profiles, candidate)
        divergence = kl_divergence(merged, reference)
        added_time = sum(
            abs(candidate[item] - initial_batches[item]) * states[item].cost
            for item in candidate
        ) / len(candidate)
        return (
            divergence > config.kl_threshold,
            max(0.0, divergence - config.kl_threshold),
            added_time,
            divergence,
        )

    for _ in range(config.max_batch_size * len(batches)):
        current_score = score(batches)
        candidates = []
        for client_id in sorted(batches, key=str):
            for delta in (-1, 1):
                value = batches[client_id] + delta
                if not 1 <= value <= config.max_batch_size:
                    continue
                candidate = {**batches, client_id: value}
                if bandwidth_usage(
                    tuple(candidate),
                    candidate,
                    config.feature_bytes_per_sample,
                ) <= config.ingress_budget_bytes:
                    candidates.append(candidate)
        if not candidates:
            break
        best = min(
            candidates,
            key=lambda item: (score(item), tuple(item.values())),
        )
        if score(best) >= current_score:
            break
        batches = best
    return batches


def select_cohort_binary_ga(
    profiles: list[WorkerProfile],
    states: dict[ClientId, WorkerState],
    config: MergeSFLPolicyConfig,
    *,
    experiment_seed: int,
    round_idx: int,
) -> CohortSelection:
    """Select a feasible cohort using reconstructed ``binary_ga_v1``."""
    profile_by_id = {profile.client_id: profile for profile in profiles}
    if set(profile_by_id) != set(states):
        raise ValueError("profiles and states must contain the same clients")
    if not config.min_clients <= len(profiles):
        raise ValueError("not enough eligible clients for min_clients")

    client_ids = tuple(sorted(profile_by_id, key=str))
    maximum_clients = min(config.max_clients, len(client_ids))
    batches = initial_batch_sizes(states, config.max_batch_size)
    reference = reference_label_distribution(profiles)
    priorities = selection_priorities(profiles)
    rng = random.Random(
        f"{experiment_seed}:{round_idx}:{config.ga_seed}:binary_ga_v1"
    )

    def feasible(candidate: tuple[bool, ...]) -> bool:
        cohort = _decode(candidate, client_ids)
        return (
            config.min_clients <= len(cohort) <= maximum_clients
            and bandwidth_usage(
                cohort, batches, config.feature_bytes_per_sample
            )
            <= config.ingress_budget_bytes
        )

    def repair(candidate: tuple[bool, ...]) -> tuple[bool, ...] | None:
        selected = {
            client_id
            for client_id, included in zip(client_ids, candidate, strict=True)
            if included
        }
        while len(selected) > maximum_clients or _usage(
            selected, batches, config
        ) > config.ingress_budget_bytes:
            if len(selected) <= config.min_clients:
                return None
            selected.remove(
                min(
                    selected,
                    key=lambda item: (
                        priorities[item],
                        -batches[item],
                        str(item),
                    ),
                )
            )
        for client_id in sorted(
            set(client_ids) - selected,
            key=lambda item: (-priorities[item], str(item)),
        ):
            if len(selected) >= config.min_clients:
                break
            proposed = selected | {client_id}
            if (
                _usage(proposed, batches, config)
                <= config.ingress_budget_bytes
            ):
                selected = proposed
        if len(selected) < config.min_clients:
            return None
        return tuple(client_id in selected for client_id in client_ids)

    def fitness(candidate: tuple[bool, ...]) -> tuple:
        cohort = _decode(candidate, client_ids)
        cohort_profiles = [profile_by_id[item] for item in cohort]
        merged = merged_label_distribution(cohort_profiles, batches)
        used = bandwidth_usage(
            cohort, batches, config.feature_bytes_per_sample
        )
        return (
            kl_divergence(merged, reference),
            config.ingress_budget_bytes - used,
            tuple(map(str, cohort)),
        )

    population = []
    attempts = 0
    while len(population) < config.population_size and attempts < 1000:
        attempts += 1
        target_size = rng.randint(config.min_clients, maximum_clients)
        chosen = _weighted_sample_without_replacement(
            rng, client_ids, priorities, target_size
        )
        candidate = repair(tuple(item in chosen for item in client_ids))
        if candidate is not None:
            population.append(candidate)
    if not population:
        raise ValueError("no bandwidth-feasible MergeSFL cohort")

    initial_population = [_decode(item, client_ids) for item in population]
    best_scores = []
    for _ in range(config.generations):
        population.sort(key=fitness)
        best_scores.append(fitness(population[0])[:2])
        next_population = [population[0]]
        while len(next_population) < config.population_size:
            first = _tournament(rng, population, fitness)
            second = _tournament(rng, population, fitness)
            child = _crossover(rng, first, second)
            child = tuple(
                not bit
                if rng.random() < config.mutation_probability
                else bit
                for bit in child
            )
            repaired = repair(child)
            next_population.append(repaired or population[0])
        population = next_population

    population.sort(key=fitness)
    best = population[0]
    cohort = _decode(best, client_ids)
    cohort_profiles = [profile_by_id[item] for item in cohort]
    merged = merged_label_distribution(cohort_profiles, batches)
    used = bandwidth_usage(cohort, batches, config.feature_bytes_per_sample)
    return CohortSelection(
        cohort=cohort,
        batch_sizes={item: batches[item] for item in cohort},
        merged_distribution=merged,
        reference_distribution=reference,
        kl=kl_divergence(merged, reference),
        bandwidth_used=used,
        trace={
            "policy": "binary_ga_v1",
            "client_order": list(client_ids),
            "initial_population": [list(item) for item in initial_population],
            "best_scores": best_scores,
            "attempts": attempts,
        },
    )


def _usage(selected, batches, config) -> int:
    if not selected:
        return 0
    return bandwidth_usage(
        tuple(selected), batches, config.feature_bytes_per_sample
    )


def _decode(bits, client_ids) -> tuple[ClientId, ...]:
    return tuple(
        client_id
        for client_id, included in zip(client_ids, bits, strict=True)
        if included
    )


def _weighted_sample_without_replacement(
    rng, client_ids, priorities, count
) -> set[ClientId]:
    remaining = list(client_ids)
    chosen = set()
    while remaining and len(chosen) < count:
        weights = [priorities[item] for item in remaining]
        item = rng.choices(remaining, weights=weights, k=1)[0]
        chosen.add(item)
        remaining.remove(item)
    return chosen


def _tournament(rng, population, fitness):
    contenders = rng.sample(population, k=min(3, len(population)))
    return min(contenders, key=fitness)


def _crossover(rng, first, second):
    if len(first) < 2:
        return first
    point = rng.randint(1, len(first) - 1)
    return first[:point] + second[point:]
