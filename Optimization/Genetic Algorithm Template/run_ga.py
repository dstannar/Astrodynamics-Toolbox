"""
GA template
"""

from __future__ import annotations
import random
from typing import List, Sequence, Tuple
import numpy as np
import pygad
from ga_cost import get_cost



# GA hyperparameters (tune as needed)
SEED = 42
NUM_GENES = 4
SOL_PER_POP = 64
NUM_PARENTS = 24
NUM_GENERATIONS = 250
MUTATION_PCT_GENES = 25  # percentage of genes per child to mutate
KEEP_PARENTS = 2
decision_vector = []


def _sample_initial_solution() -> np.ndarray:
    """Sample one valid solution"""


def _make_initial_population(n: int) -> np.ndarray:
    pop = np.empty((n, NUM_GENES), dtype=int)
    for i in range(n):
        pop[i, :] = _sample_initial_solution()
    return pop


def _fitness_func(ga: "pygad.GA", solution: Sequence[float], solution_idx: int) -> float:
    # Coerce to ints

    # Evaluate cost
    try:
        cost = float(get_cost(decision_vector))
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

    random.seed(SEED)
    np.random.seed(SEED)

    # Each gene has its own discrete space.
    gene_space = []

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
        best_cost, details = float(get_cost(decision_vector))
    except Exception:
        best_cost = float("nan")

    print("\nBest decision vector:", decision_vector)
    print("Objective cost:", best_cost)
    print("Fitness:", fitness)
    print("Transfer Details: ")

    # save GA and plot
    ga.save("ga_flybys.pkl")
    ga.plot_fitness()

if __name__ == "__main__":
    main()
