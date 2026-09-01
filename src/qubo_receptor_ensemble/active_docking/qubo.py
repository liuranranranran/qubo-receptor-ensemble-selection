"""Budget-constrained batch QUBO for ligand-receptor task selection."""

from __future__ import annotations

import math
import hashlib
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .state import Task


@dataclass(frozen=True)
class BatchConstraints:
    budget: float
    task_costs: Mapping[Task, float]
    max_per_ligand: int | None = None
    max_per_receptor: int | None = None
    max_per_scaffold: int | None = None
    ligand_scaffolds: Mapping[str, str] = field(default_factory=dict)
    receptor_activation_cost: Mapping[str, float] | float | None = None
    penalty: float = 10.0
    cost_unit: float | None = None
    equal_cost: bool = False
    coefficient_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.budget < 0 or not math.isfinite(self.budget):
            raise ValueError("budget must be a non-negative finite value")
        if self.penalty <= 0 or not math.isfinite(self.penalty):
            raise ValueError("penalty must be a positive finite value")
        if self.coefficient_scale <= 0 or not math.isfinite(self.coefficient_scale):
            raise ValueError("coefficient_scale must be positive and finite")
        if self.cost_unit is not None and (self.cost_unit <= 0 or not math.isfinite(self.cost_unit)):
            raise ValueError("cost_unit must be positive and finite")
        for limit in (self.max_per_ligand, self.max_per_receptor, self.max_per_scaffold):
            if limit is not None and limit < 0:
                raise ValueError("constraint limits must be non-negative")
        if isinstance(self.receptor_activation_cost, Mapping):
            costs = self.receptor_activation_cost.values()
        elif self.receptor_activation_cost is None:
            costs = ()
        else:
            costs = (self.receptor_activation_cost,)
        if any(float(cost) < 0 or not math.isfinite(float(cost)) for cost in costs):
            raise ValueError("receptor activation costs must be non-negative and finite")

    def activation_cost(self, receptor_id: str) -> float:
        if self.receptor_activation_cost is None:
            return 0.0
        if isinstance(self.receptor_activation_cost, Mapping):
            return float(self.receptor_activation_cost.get(receptor_id, 0.0))
        return float(self.receptor_activation_cost)


@dataclass(frozen=True)
class FeasibilityResult:
    is_feasible: bool
    violations: tuple[str, ...]
    total_cost: float
    integerization_error: float

    def as_dict(self) -> dict[str, object]:
        return {
            "is_feasible": self.is_feasible,
            "violations": list(self.violations),
            "total_cost": self.total_cost,
            "integerization_error": self.integerization_error,
        }


def _slack_weights(capacity: int) -> tuple[int, ...]:
    if capacity <= 0:
        return ()
    weights: list[int] = []
    covered = 0
    power = 1
    while covered + power < capacity:
        weights.append(power)
        covered += power
        power *= 2
    weights.append(capacity - covered)
    return tuple(weights)


def _encode_slack(value: int, weights: Sequence[int]) -> list[int]:
    if value < 0:
        raise ValueError("slack value must be non-negative")
    remaining = value
    bits: list[int] = []
    for weight in reversed(tuple(weights)):
        bit = min(1, remaining // weight) if weight else 0
        bits.append(bit)
        remaining -= bit * weight
    if remaining:
        raise ValueError(f"slack weights cannot encode value {value}")
    return list(reversed(bits))


@dataclass
class BatchQUBO:
    tasks: tuple[Task, ...]
    values: dict[Task, float]
    interactions: dict[tuple[Task, Task], float]
    constraints: BatchConstraints
    matrix: np.ndarray
    variables: tuple[str, ...]
    task_indices: dict[Task, int]
    linear_terms: dict[str, float]
    quadratic_terms: dict[tuple[str, str], float]
    constant: float
    constraint_specs: tuple[tuple[str, tuple[tuple[str, int], ...], int], ...]
    cost_scale: int
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def variable_count(self) -> int:
        return len(self.variables)

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update("|".join(self.variables).encode("utf-8"))
        digest.update(self.matrix.tobytes())
        digest.update(str(self.constant).encode("ascii"))
        return digest.hexdigest()

    def _assignment_array(self, assignment: Mapping[str, int] | Sequence[int]) -> np.ndarray:
        if isinstance(assignment, Mapping):
            values = np.asarray([int(assignment.get(name, 0)) for name in self.variables], dtype=float)
        else:
            values = np.asarray(list(assignment), dtype=float)
            if len(values) != self.variable_count:
                raise ValueError("assignment length does not match QUBO variable count")
        if np.any((values != 0) & (values != 1)):
            raise ValueError("QUBO assignments must be binary")
        return values

    def energy(self, assignment: Mapping[str, int] | Sequence[int]) -> float:
        vector = self._assignment_array(assignment)
        return float(vector @ self.matrix @ vector + self.constant)

    def task_set_from_assignment(self, assignment: Mapping[str, int] | Sequence[int]) -> tuple[Task, ...]:
        vector = self._assignment_array(assignment)
        return tuple(
            task for task, index in sorted(self.task_indices.items(), key=lambda item: item[1])
            if vector[index] == 1
        )

    def assignment_for_tasks(self, tasks: Sequence[Task]) -> dict[str, int]:
        selected = tuple(sorted(tasks))
        feasibility = self.validate_tasks(selected)
        if not feasibility.is_feasible:
            raise ValueError(f"cannot create a feasible QUBO assignment: {feasibility.violations}")
        assignment = {name: 0 for name in self.variables}
        for task in selected:
            assignment[f"z|{task[0]}|{task[1]}"] = 1
        selected_receptors = {task[1] for task in selected}
        for receptor_id in selected_receptors:
            variable = f"x|{receptor_id}"
            if variable in assignment:
                assignment[variable] = 1
        for name, terms, target in self.constraint_specs:
            used = sum(weight * assignment[variable] for variable, weight in terms)
            remainder = target - used
            slack_names = [variable for variable, _ in terms if variable.startswith(f"slack|{name}|")]
            weights = [weight for variable, weight in terms if variable in slack_names]
            for variable, bit in zip(slack_names, _encode_slack(remainder, weights)):
                assignment[variable] = bit
        return assignment

    def energy_for_tasks(self, tasks: Sequence[Task]) -> float:
        selected = tuple(sorted(tasks))
        if self.validate_tasks(selected).is_feasible:
            return self.energy(self.assignment_for_tasks(selected))
        return float("inf")

    def validate_tasks(self, tasks: Sequence[Task]) -> FeasibilityResult:
        selected = tuple(tasks)
        violations: list[str] = []
        if len(set(selected)) != len(selected):
            violations.append("duplicate_task")
        unknown = sorted(set(selected) - set(self.tasks))
        if unknown:
            violations.append(f"unknown_tasks:{unknown}")
        selected = tuple(task for task in selected if task in self.tasks)
        total_cost = sum(float(self.constraints.task_costs[task]) for task in selected)
        active_receptors = {task[1] for task in selected}
        total_cost += sum(self.constraints.activation_cost(receptor) for receptor in active_receptors)
        if total_cost > self.constraints.budget + 1e-9:
            violations.append("budget")
        if self.constraints.max_per_ligand is not None:
            counts: dict[str, int] = {}
            for ligand_id, _ in selected:
                counts[ligand_id] = counts.get(ligand_id, 0) + 1
            if any(count > self.constraints.max_per_ligand for count in counts.values()):
                violations.append("max_per_ligand")
        if self.constraints.max_per_receptor is not None:
            counts = {}
            for _, receptor_id in selected:
                counts[receptor_id] = counts.get(receptor_id, 0) + 1
            if any(count > self.constraints.max_per_receptor for count in counts.values()):
                violations.append("max_per_receptor")
        if self.constraints.max_per_scaffold is not None:
            counts = {}
            for ligand_id, _ in selected:
                scaffold = self.constraints.ligand_scaffolds.get(ligand_id, "__unknown__")
                counts[scaffold] = counts.get(scaffold, 0) + 1
            if any(count > self.constraints.max_per_scaffold for count in counts.values()):
                violations.append("max_per_scaffold")
        return FeasibilityResult(
            is_feasible=not violations,
            violations=tuple(violations),
            total_cost=total_cost,
            integerization_error=abs(total_cost - self._represented_cost(selected)),
        )

    def _represented_cost(self, tasks: Sequence[Task]) -> float:
        unit = self.metadata.get("cost_unit", 1.0 / self.cost_scale)
        integer = sum(round(float(self.constraints.task_costs[task]) / float(unit)) for task in tasks)
        integer += sum(round(self.constraints.activation_cost(receptor) / float(unit)) for receptor in {task[1] for task in tasks})
        return integer * float(unit)


def build_batch_qubo(
    tasks: Sequence[Task],
    values: Mapping[Task, float],
    interactions: Mapping[tuple[Task, Task], float],
    constraints: BatchConstraints,
    batch_interaction_weight: float = 1.0,
) -> BatchQUBO:
    ordered_tasks = tuple(sorted(tuple(task) for task in tasks))
    if len(set(ordered_tasks)) != len(ordered_tasks):
        raise ValueError("tasks must be unique")
    if any(task not in constraints.task_costs for task in ordered_tasks):
        raise ValueError("task costs must be supplied for every task")
    if any(not math.isfinite(float(values[task])) for task in ordered_tasks):
        raise ValueError("task values must be finite")
    cost_unit = constraints.cost_unit
    if cost_unit is None:
        cost_unit = 1.0 / 1000.0
    cost_scale = round(1.0 / cost_unit)
    if cost_scale <= 0:
        raise ValueError("cost_unit must produce a positive integer cost scale")
    budget_units = round(constraints.budget / cost_unit)
    task_units = {task: round(float(constraints.task_costs[task]) / cost_unit) for task in ordered_tasks}
    if any(value <= 0 for value in task_units.values()):
        raise ValueError("task costs are too small for the configured cost_unit")

    base_variables = [f"z|{ligand_id}|{receptor_id}" for ligand_id, receptor_id in ordered_tasks]
    receptor_ids = sorted({task[1] for task in ordered_tasks})
    activation_variables = [f"x|{receptor_id}" for receptor_id in receptor_ids if constraints.activation_cost(receptor_id) > 0]
    variables = list(base_variables) + activation_variables
    task_indices = {task: index for index, task in enumerate(ordered_tasks)}
    specs: list[tuple[str, tuple[tuple[str, int], ...], int]] = []
    specs.append(("budget", tuple(
        [(f"z|{task[0]}|{task[1]}", task_units[task]) for task in ordered_tasks]
        + [(f"x|{receptor_id}", round(constraints.activation_cost(receptor_id) / cost_unit)) for receptor_id in receptor_ids if constraints.activation_cost(receptor_id) > 0]
        + [(f"slack|budget|{index}", weight) for index, weight in enumerate(_slack_weights(budget_units))]
    ), budget_units))
    for name, limit, grouping in (
        ("ligand", constraints.max_per_ligand, lambda task: task[0]),
        ("receptor", constraints.max_per_receptor, lambda task: task[1]),
        ("scaffold", constraints.max_per_scaffold, lambda task: constraints.ligand_scaffolds.get(task[0], "__unknown__")),
    ):
        if limit is None:
            continue
        for group in sorted({grouping(task) for task in ordered_tasks}):
            group_tasks = [task for task in ordered_tasks if grouping(task) == group]
            slack = _slack_weights(limit)
            specs.append((
                f"{name}|{group}",
                tuple([(f"z|{task[0]}|{task[1]}", 1) for task in group_tasks]
                      + [(f"slack|{name}|{group}|{index}", weight) for index, weight in enumerate(slack)]),
                limit,
            ))
    if constraints.equal_cost and ordered_tasks:
        costs = [float(constraints.task_costs[task]) for task in ordered_tasks]
        if max(costs) - min(costs) <= 1e-12:
            cardinality_limit = math.floor(constraints.budget / costs[0] + 1e-12)
            slack = _slack_weights(cardinality_limit)
            specs.append((
                "cardinality",
                tuple([(f"z|{task[0]}|{task[1]}", 1) for task in ordered_tasks]
                      + [(f"slack|cardinality|{index}", weight) for index, weight in enumerate(slack)]),
                cardinality_limit,
            ))
    for _, terms, _ in specs:
        variables.extend(variable for variable, _ in terms if variable not in variables)

    matrix = np.zeros((len(variables), len(variables)), dtype=float)
    linear_terms: dict[str, float] = {name: 0.0 for name in variables}
    quadratic_terms: dict[tuple[str, str], float] = {}
    constant = 0.0
    indices = {name: index for index, name in enumerate(variables)}

    def add_linear(variable: str, coefficient: float) -> None:
        linear_terms[variable] += coefficient
        matrix[indices[variable], indices[variable]] += coefficient

    def add_pair(first: str, second: str, coefficient: float) -> None:
        key = tuple(sorted((first, second)))
        quadratic_terms[key] = quadratic_terms.get(key, 0.0) + coefficient
        matrix[indices[first], indices[second]] += coefficient / 2.0
        matrix[indices[second], indices[first]] += coefficient / 2.0

    for task in ordered_tasks:
        add_linear(f"z|{task[0]}|{task[1]}", -float(values[task]))
    for (first, second), value in interactions.items():
        first, second = tuple(first), tuple(second)
        if first in task_indices and second in task_indices and first != second:
            add_pair(f"z|{first[0]}|{first[1]}", f"z|{second[0]}|{second[1]}", -batch_interaction_weight * float(value))
    penalty = constraints.penalty
    for receptor_id in receptor_ids:
        activation = f"x|{receptor_id}"
        if activation not in indices:
            continue
        add_linear(activation, 0.0)
        for task in ordered_tasks:
            if task[1] == receptor_id:
                z_variable = f"z|{task[0]}|{task[1]}"
                add_linear(z_variable, penalty)
                add_pair(z_variable, activation, -penalty)

    for name, terms, target in specs:
        target = int(target)
        constant += penalty * target * target
        for variable, weight in terms:
            add_linear(variable, penalty * weight * weight - 2.0 * penalty * target * weight)
        for index, (first, first_weight) in enumerate(terms):
            for second, second_weight in terms[index + 1 :]:
                add_pair(first, second, 2.0 * penalty * first_weight * second_weight)

    coefficient_scale = float(constraints.coefficient_scale)
    if coefficient_scale != 1.0:
        matrix *= coefficient_scale
        linear_terms = {name: value * coefficient_scale for name, value in linear_terms.items()}
        quadratic_terms = {key: value * coefficient_scale for key, value in quadratic_terms.items()}
        constant *= coefficient_scale

    qubo = BatchQUBO(
        tasks=ordered_tasks,
        values={task: float(values[task]) for task in ordered_tasks},
        interactions={(tuple(first), tuple(second)): float(value) for (first, second), value in interactions.items()},
        constraints=constraints,
        matrix=matrix,
        variables=tuple(variables),
        task_indices={task: indices[f"z|{task[0]}|{task[1]}"] for task in ordered_tasks},
        linear_terms=linear_terms,
        quadratic_terms=quadratic_terms,
        constant=constant,
        constraint_specs=tuple(specs),
        cost_scale=cost_scale,
        metadata={
            "objective_mode": "posterior_expected_utility",
            "cost_unit": cost_unit,
            "cost_scale": cost_scale,
            "budget_units": budget_units,
            "penalty": penalty,
            "batch_interaction_weight": batch_interaction_weight,
            "coefficient_scale": coefficient_scale,
            "constraint_diagnostics": "available via validate_tasks",
        },
    )
    return qubo
