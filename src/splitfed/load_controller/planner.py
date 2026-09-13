"""Deterministic MergeSFL Algorithm 1 round planner."""

import random
from dataclasses import dataclass

from ...schema import MergeSFLPolicyConfig
from .mergesfl import (
    ClientId,
    WorkerProfile,
    WorkerState,
    bandwidth_usage,
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
