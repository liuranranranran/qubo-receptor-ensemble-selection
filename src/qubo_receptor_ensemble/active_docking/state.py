"""Explicit state for a partially observed ligand-receptor score matrix."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

Task = tuple[str, str]
_FORBIDDEN_METADATA_KEYS = {"label", "active", "decoy", "hidden_score", "hidden_label"}


class StateError(ValueError):
    """Raised when an invalid state transition is requested."""


def _task_key(task: Task) -> str:
    return f"{task[0]}||{task[1]}"


def _as_task(value: object) -> Task:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise StateError(f"task must be a ligand/receptor pair: {value!r}")
    ligand_id, receptor_id = (str(value[0]), str(value[1]))
    if not ligand_id or not receptor_id:
        raise StateError("task identifiers must be non-empty")
    return ligand_id, receptor_id


def _contains_forbidden_key(value: object) -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in _FORBIDDEN_METADATA_KEYS:
                return str(key)
            found = _contains_forbidden_key(item)
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for item in value:
            found = _contains_forbidden_key(item)
            if found is not None:
                return found
    return None


@dataclass
class PartialObservationState:
    """Mutable visible state; hidden oracle data lives outside this object."""

    ligand_manifest: list[dict[str, object]]
    receptor_manifest: list[dict[str, object]]
    candidate_tasks: set[Task] | None = None
    observed_scores: dict[Task, float] = field(default_factory=dict)
    current_round: int = 0
    warm_start_state: dict[str, object] = field(default_factory=dict)
    docking_cost: float = 0.0
    task_costs: dict[Task, float] = field(default_factory=dict)
    scaffold_metadata: dict[str, object] = field(default_factory=dict)
    receptor_cluster_metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.ligand_manifest = [dict(row) for row in self.ligand_manifest]
        self.receptor_manifest = [dict(row) for row in self.receptor_manifest]
        ligand_ids = [str(row.get("ligand_id", "")) for row in self.ligand_manifest]
        receptor_ids = [str(row.get("receptor_id", "")) for row in self.receptor_manifest]
        if not ligand_ids or any(not value for value in ligand_ids) or len(set(ligand_ids)) != len(ligand_ids):
            raise StateError("ligand_manifest must contain unique non-empty ligand_id values")
        if not receptor_ids or any(not value for value in receptor_ids) or len(set(receptor_ids)) != len(receptor_ids):
            raise StateError("receptor_manifest must contain unique non-empty receptor_id values")
        forbidden = _contains_forbidden_key(self.ligand_manifest)
        if forbidden is not None:
            raise StateError(f"hidden evaluation metadata is not allowed in state: {forbidden}")
        forbidden = _contains_forbidden_key(self.receptor_manifest)
        if forbidden is not None:
            raise StateError(f"hidden evaluation metadata is not allowed in state: {forbidden}")
        all_tasks = {(ligand_id, receptor_id) for ligand_id in ligand_ids for receptor_id in receptor_ids}
        if self.candidate_tasks is None:
            self.candidate_tasks = set(all_tasks)
        else:
            self.candidate_tasks = {_as_task(task) for task in self.candidate_tasks}
            if not self.candidate_tasks.issubset(all_tasks):
                raise StateError("candidate_tasks contains an unknown ligand or receptor")
        self.observed_scores = {_as_task(task): float(score) for task, score in self.observed_scores.items()}
        if not set(self.observed_scores).issubset(self.candidate_tasks):
            raise StateError("observed_scores contains a task outside candidate_tasks")
        if any(not math.isfinite(score) for score in self.observed_scores.values()):
            raise StateError("observed scores must be finite")
        self.task_costs = {_as_task(task): float(cost) for task, cost in self.task_costs.items()}
        if not set(self.task_costs).issubset(self.candidate_tasks):
            raise StateError("task_costs contains a task outside candidate_tasks")
        if any(not math.isfinite(cost) or cost <= 0.0 for cost in self.task_costs.values()):
            raise StateError("task costs must be positive finite values")
        self.docking_cost = float(self.docking_cost)
        if not math.isfinite(self.docking_cost) or self.docking_cost < 0:
            raise StateError("docking_cost must be a non-negative finite value")
        if self.current_round < 0:
            raise StateError("current_round must be non-negative")
        self.assert_no_hidden_information()

    def completed_tasks(self) -> tuple[Task, ...]:
        return tuple(sorted(self.observed_scores))

    def unfinished_tasks(self) -> tuple[Task, ...]:
        return tuple(sorted(self.candidate_tasks - set(self.observed_scores)))

    def score_for(self, task: Task) -> float:
        task = _as_task(task)
        try:
            return self.observed_scores[task]
        except KeyError as exc:
            raise StateError(f"score has not been revealed for task: {task}") from exc

    def cost_for(self, task: Task) -> float:
        task = _as_task(task)
        return float(self.task_costs.get(task, 1.0))

    def reveal(self, scores: Mapping[Task, float]) -> None:
        """Atomically reveal selected oracle scores and charge their costs."""
        normalized = {_as_task(task): float(score) for task, score in scores.items()}
        if not normalized:
            raise StateError("at least one task must be revealed")
        invalid = sorted(set(normalized) - self.candidate_tasks)
        if invalid:
            raise StateError(f"cannot reveal tasks outside candidate set: {invalid}")
        duplicate = sorted(set(normalized) & set(self.observed_scores))
        if duplicate:
            raise StateError(f"cannot reveal completed tasks again: {duplicate}")
        if any(not math.isfinite(score) for score in normalized.values()):
            raise StateError("revealed scores must be finite")
        cost = sum(self.cost_for(task) for task in normalized)
        self.observed_scores.update(normalized)
        self.docking_cost += cost
        self.current_round += 1

    def copy(self) -> "PartialObservationState":
        return copy.deepcopy(self)

    def assert_no_hidden_information(self) -> None:
        for value in (self.ligand_manifest, self.receptor_manifest, self.warm_start_state,
                      self.scaffold_metadata, self.receptor_cluster_metadata):
            forbidden = _contains_forbidden_key(value)
            if forbidden is not None:
                raise StateError(f"hidden information is present in state: {forbidden}")

    def to_dict(self) -> dict[str, object]:
        self.assert_no_hidden_information()
        return {
            "ligand_manifest": copy.deepcopy(self.ligand_manifest),
            "receptor_manifest": copy.deepcopy(self.receptor_manifest),
            "candidate_tasks": [list(task) for task in sorted(self.candidate_tasks)],
            "observed_scores": [
                {"ligand_id": task[0], "receptor_id": task[1], "score": score}
                for task, score in sorted(self.observed_scores.items())
            ],
            "current_round": self.current_round,
            "warm_start_state": copy.deepcopy(self.warm_start_state),
            "docking_cost": self.docking_cost,
            "task_costs": [
                {"ligand_id": task[0], "receptor_id": task[1], "cost": cost}
                for task, cost in sorted(self.task_costs.items())
            ],
            "scaffold_metadata": copy.deepcopy(self.scaffold_metadata),
            "receptor_cluster_metadata": copy.deepcopy(self.receptor_cluster_metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "PartialObservationState":
        def records(name: str, score_key: str) -> dict[Task, float]:
            raw = value.get(name, [])
            if not isinstance(raw, list):
                raise StateError(f"{name} must be a list")
            output: dict[Task, float] = {}
            for item in raw:
                if not isinstance(item, Mapping):
                    raise StateError(f"{name} records must be objects")
                task = _as_task((item.get("ligand_id", ""), item.get("receptor_id", "")))
                output[task] = float(item[score_key])
            return output

        return cls(
            ligand_manifest=[dict(row) for row in value.get("ligand_manifest", [])],
            receptor_manifest=[dict(row) for row in value.get("receptor_manifest", [])],
            candidate_tasks={_as_task(task) for task in value.get("candidate_tasks", [])},
            observed_scores=records("observed_scores", "score"),
            current_round=int(value.get("current_round", 0)),
            warm_start_state=dict(value.get("warm_start_state", {})),
            docking_cost=float(value.get("docking_cost", 0.0)),
            task_costs=records("task_costs", "cost"),
            scaffold_metadata=dict(value.get("scaffold_metadata", {})),
            receptor_cluster_metadata=dict(value.get("receptor_cluster_metadata", {})),
        )
