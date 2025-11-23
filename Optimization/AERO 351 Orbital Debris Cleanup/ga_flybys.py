#!/usr/bin/env python3
"""
GA for selecting [GEO_ID, MEO_ID, LEO1_ID, LEO2_ID] using PyGAD.
- Each gene is drawn from its respective list (GEO, MEO, LEO, LEO).
- Enforces LEO1 != LEO2 via a small repair step after crossover/mutation and in the initial population.
- Assumes you provide `get_rendezvous(geo_id: int, meo_id: int, leo1_id: int, leo2_id: int) -> float`
  which returns a COST to minimize. We convert it to a fitness to maximize.

Usage:
  1) Fill GEO_IDS, MEO_IDS, LEO_IDS below.
  2) Replace the import for `get_rendezvous` with your actual module.
  3) `pip install pygad numpy`
  4) Run: `python ga_flybys.py`

Notes:
  - Fitness = 1 / (1 + cost). If your costs are huge/ill-scaled, consider a different transform.
  - The hard constraint LEO1 != LEO2 is enforced by construction and repaired after genetic ops.
"""

from __future__ import annotations
import random
from typing import List, Sequence, Tuple
import numpy as np
import pygad
from rendezvous_cost import get_rendezvous_cost


# Debris Body IDs
GEO_IDS: List[int] = []
MEO_IDS: List[int] = []
LEO_IDS: List[int] = []


# GA hyperparameters (tune as needed)
SEED = 42
NUM_GENES = 4
SOL_PER_POP = 64
NUM_PARENTS = 24
NUM_GENERATIONS = 250
MUTATION_PCT_GENES = 25  # percentage of genes per child to mutate
KEEP_PARENTS = 2


def _validate_id_lists():
    if not GEO_IDS or not MEO_IDS or not LEO_IDS:
        raise ValueError("GEO_IDS, MEO_IDS, and LEO_IDS must be non-empty lists of ints.")
    if len(LEO_IDS) < 2:
        raise ValueError("LEO_IDS must contain at least 2 distinct IDs to enforce LEO1 != LEO2.")
    for name, lst in [("GEO_IDS", GEO_IDS), ("MEO_IDS", MEO_IDS), ("LEO_IDS", LEO_IDS)]:
        if not all(isinstance(x, int) for x in lst):
            raise TypeError(f"All entries in {name} must be integers.")


def _sample_initial_solution() -> np.ndarray:
    """Sample one valid solution [GEO, MEO, LEO1, LEO2] with LEO1 != LEO2."""
    geo = random.choice(GEO_IDS)
    meo = random.choice(MEO_IDS)
    leo1, leo2 = random.sample(LEO_IDS, 2)  # guarantees distinct
    return np.array([geo, meo, leo1, leo2], dtype=int)


def _make_initial_population(n: int) -> np.ndarray:
    pop = np.empty((n, NUM_GENES), dtype=int)
    for i in range(n):
        pop[i, :] = _sample_initial_solution()
    return pop


def _repair_solution(sol: np.ndarray) -> None:
    """In-place repair: ensure gene2 != gene3 (LEO1 != LEO2)."""
    # Index mapping: 0=GEO, 1=MEO, 2=LEO1, 3=LEO2
    leo1 = int(sol[2])
    leo2 = int(sol[3])
    if leo1 == leo2:
        # Choose a different LEO2 while staying in LEO_IDS.
        choices = [x for x in LEO_IDS if x != leo1]
        sol[3] = random.choice(choices)


def _repair_offspring(_: "pygad.GA", offspring: np.ndarray) -> None:
    """Callback to repair all offspring after crossover/mutation. Mutates in place."""
    for k in range(offspring.shape[0]):
        _repair_solution(offspring[k, :])


def _fitness_func(ga: "pygad.GA", solution: Sequence[float], solution_idx: int) -> float:
    # Coerce to ints
    geo, meo, leo1, leo2 = map(int, solution)

    # Evaluate cost
    try:
        cost = float(get_rendezvous_cost(geo, meo, leo1, leo2))
    except Exception as e:
        # In case the function throws
        cost = float("inf")

    if not np.isfinite(cost):
        cost = float("inf")

    # Convert minimization to maximization
    # If cost is inf, fitness -> 0
    return 0.0 if not np.isfinite(cost) else 1.0 / (1.0 + cost)


def _on_generation(ga: "pygad.GA") -> None:
    best = ga.best_solution()
    best_fit = best[1]
    gen = ga.generations_completed
    print(f"Gen {gen:4d} | best fitness = {best_fit:.6f}")


def build_ga() -> "pygad.GA":
    _validate_id_lists()

    random.seed(SEED)
    np.random.seed(SEED)

    # Each gene has its own discrete space.
    gene_space = [GEO_IDS, MEO_IDS, LEO_IDS, LEO_IDS]

    initial_population = _make_initial_population(SOL_PER_POP)

    ga = pygad.GA(
        num_generations=NUM_GENERATIONS,
        sol_per_pop=SOL_PER_POP,
        num_parents_mating=NUM_PARENTS,
        num_genes=NUM_GENES,
        gene_type=int,
        gene_space=gene_space,
        fitness_func=_fitness_func,
        parent_selection_type="tournament",     # robust default
        keep_parents=KEEP_PARENTS,
        crossover_type="two_points",            # good exploration
        mutation_type="random",
        mutation_by_replacement=True,           # pick from gene_space on mutation
        mutation_percent_genes=MUTATION_PCT_GENES,
        on_crossover=_repair_offspring,
        on_mutation=_repair_offspring,
        on_generation=_on_generation,
        stop_criteria=["saturate_50"],          # stop if no improvement for 50 gens
        random_seed=SEED,
        initial_population=initial_population,
    )
    return ga


def main() -> None:
    ga = build_ga()
    ga.run()

    solution, fitness, idx = ga.best_solution()
    geo, meo, leo1, leo2 = map(int, solution)
    try:
        best_cost, details = float(get_rendezvous_cost(geo, meo, leo1, leo2))
    except Exception:
        best_cost = float("nan")

    print("\nBest decision vector [GEO, MEO, LEO1, LEO2]:", [geo, meo, leo1, leo2])
    print("Objective cost:", best_cost)
    print("Fitness:", fitness)
    print("Transfer Details: ")

    # save GA and plot
    ga.save("ga_flybys.pkl")
    ga.plot_fitness()

if __name__ == "__main__":
    main()
