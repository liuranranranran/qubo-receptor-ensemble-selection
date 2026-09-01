"""Solver adapters for the active-docking batch QUBO."""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass

import numpy as np

from .qubo import BatchQUBO
from .state import Task


class BatchSolverError(ValueError):
    """Raised when a batch solver cannot solve the supplied QUBO."""


@dataclass(frozen=True)
class BatchSolverResult:
    backend: str
    tasks: tuple[Task, ...]
    energy: float
    metadata: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "tasks": [list(task) for task in self.tasks],
            "energy": self.energy,
            "metadata": self.metadata,
        }


def _result(qubo: BatchQUBO, backend: str, tasks: tuple[Task, ...], started: float, **metadata: object) -> BatchSolverResult:
    diagnostic = qubo.validate_tasks(tasks)
    if not diagnostic.is_feasible:
        raise BatchSolverError(f"solver returned infeasible tasks: {diagnostic.violations}")
    return BatchSolverResult(
        backend=backend,
        tasks=tuple(sorted(tasks)),
        energy=qubo.energy_for_tasks(tasks),
        metadata={
            "solver_time_seconds": time.perf_counter() - started,
            "feasibility": diagnostic.as_dict(),
            **metadata,
        },
    )


def _exact(qubo: BatchQUBO, backend: str, started: float) -> BatchSolverResult:
    best: tuple[float, tuple[Task, ...]] | None = None
    states = 0
    for size in range(len(qubo.tasks) + 1):
        for combination in itertools.combinations(qubo.tasks, size):
            states += 1
            if not qubo.validate_tasks(combination).is_feasible:
                continue
            energy = qubo.energy_for_tasks(combination)
            candidate = (energy, tuple(combination))
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise BatchSolverError("no feasible task set exists")
    return _result(qubo, backend, best[1], started, states_evaluated=states, selection_rule="minimum QUBO energy, then lexicographic tasks")


def _greedy_tasks(qubo: BatchQUBO) -> tuple[Task, ...]:
    selected: list[Task] = []
    while True:
        candidates: list[tuple[float, Task]] = []
        for task in qubo.tasks:
            if task in selected:
                continue
            proposed = tuple(sorted((*selected, task)))
            if not qubo.validate_tasks(proposed).is_feasible:
                continue
            selected_receptors = {selected_task[1] for selected_task in selected}
            cost = qubo.constraints.task_costs[task]
            if task[1] not in selected_receptors:
                cost += qubo.constraints.activation_cost(task[1])
            value = qubo.values[task]
            candidates.append((float(value) / float(cost), task))
        if not candidates:
            break
        ratio, task = max(candidates, key=lambda item: (item[0], tuple(reversed(item[1]))))
        if ratio <= 0:
            break
        selected.append(task)
    return tuple(sorted(selected))


def _greedy(qubo: BatchQUBO, started: float) -> BatchSolverResult:
    tasks = _greedy_tasks(qubo)
    return _result(qubo, "value_greedy", tasks, started, selection_rule="maximum visible value per incremental task cost")


def _one_swap(qubo: BatchQUBO, started: float) -> BatchSolverResult:
    selected = _greedy_tasks(qubo)
    improved = True
    swaps = 0
    while improved:
        improved = False
        current_energy = qubo.energy_for_tasks(selected)
        for outgoing in selected:
            for incoming in qubo.tasks:
                if incoming in selected:
                    continue
                candidate = tuple(sorted((set(selected) - {outgoing}) | {incoming}))
                if not qubo.validate_tasks(candidate).is_feasible:
                    continue
                candidate_energy = qubo.energy_for_tasks(candidate)
                if (candidate_energy, candidate) < (current_energy, selected):
                    selected = candidate
                    swaps += 1
                    improved = True
                    break
            if improved:
                break
    return _result(qubo, "greedy_one_swap", selected, started, swaps=swaps, selection_rule="value-greedy followed by improving one-for-one swaps")


def _anneal(qubo: BatchQUBO, backend: str, random_seed: int, started: float, iterations: int = 2000) -> BatchSolverResult:
    rng = np.random.default_rng(random_seed)
    current = _greedy_tasks(qubo)
    best = current
    current_energy = qubo.energy_for_tasks(current)
    best_energy = current_energy
    for iteration in range(iterations):
        if not qubo.tasks:
            break
        task = qubo.tasks[int(rng.integers(0, len(qubo.tasks)))]
        proposal_set = set(current)
        if task in proposal_set:
            proposal_set.remove(task)
        else:
            proposal_set.add(task)
        proposal = tuple(sorted(proposal_set))
        if not qubo.validate_tasks(proposal).is_feasible:
            continue
        proposal_energy = qubo.energy_for_tasks(proposal)
        temperature = max(1e-6, 1.0 - iteration / max(iterations, 1))
        accept = proposal_energy <= current_energy or rng.random() < math.exp(min(0.0, (current_energy - proposal_energy) / temperature))
        if accept:
            current, current_energy = proposal, proposal_energy
            if (current_energy, current) < (best_energy, best):
                best, best_energy = current, current_energy
    return _result(qubo, backend, best, started, iterations=iterations, random_seed=random_seed, selection_rule="feasible simulated annealing over task variables")


def solve_batch_qubo(
    qubo: BatchQUBO,
    *,
    backend: str,
    random_seed: int = 0,
    time_budget_seconds: float | None = None,
) -> BatchSolverResult:
    """Solve one frozen QUBO through a named, auditable backend."""
    started = time.perf_counter()
    if time_budget_seconds is not None and time_budget_seconds <= 0:
        raise BatchSolverError("time_budget_seconds must be positive")
    if backend == "exact":
        result = _exact(qubo, backend, started)
    elif backend in {"value_greedy", "greedy"}:
        result = _greedy(qubo, started)
    elif backend in {"greedy_one_swap", "greedy+one_swap"}:
        result = _one_swap(qubo, started)
    elif backend == "simulated_annealing":
        result = _anneal(qubo, backend, random_seed, started)
    elif backend in {"quantum_compatible_simulator", "quantum_compatible"}:
        result = _anneal(qubo, "quantum_compatible_simulator", random_seed, started)
        result = BatchSolverResult(
            backend=result.backend,
            tasks=result.tasks,
            energy=result.energy,
            metadata={
                **result.metadata,
                "backend_type": "quantum_compatible_simulation",
                "quantum_hardware_used": False,
                "quantum_execution_result": "not_run",
            },
        )
    elif backend in {"cp_sat", "milp"}:
        raise BatchSolverError(f"optional backend is unavailable in this environment: {backend}")
    else:
        raise BatchSolverError(f"unknown batch solver backend: {backend}")
    if time_budget_seconds is not None:
        result.metadata["time_budget_seconds"] = time_budget_seconds
    return result
