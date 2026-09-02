"""Posterior-sampling acquisition and batch interaction evaluation."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

from .predictor import ScorePrediction, ScorePredictor
from .state import PartialObservationState, Task


@dataclass(frozen=True)
class AcquisitionConfig:
    top_q: int = 10
    monte_carlo_samples: int = 64
    risk_lambda: float = 0.0
    utility_mode: str = "ranking_score"
    random_seed: int = 0

    def __post_init__(self) -> None:
        if self.top_q <= 0 or self.monte_carlo_samples <= 0:
            raise ValueError("top_q and monte_carlo_samples must be positive")
        if self.risk_lambda < 0 or not math.isfinite(self.risk_lambda):
            raise ValueError("risk_lambda must be a non-negative finite value")
        if self.utility_mode not in {"ranking_score", "activity_prior"}:
            raise ValueError("utility_mode must be ranking_score or activity_prior")


def _stable_int(seed: int, task: Task) -> int:
    digest = hashlib.sha256(f"{seed}|{task[0]}|{task[1]}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


class PosteriorAcquisitionEvaluator:
    """Evaluate expected ranking utility using only visible state and posterior draws."""

    def __init__(
        self,
        state: PartialObservationState,
        predictor: ScorePredictor,
        config: AcquisitionConfig,
        activity_prior: Mapping[str, float] | None = None,
    ) -> None:
        state.assert_no_hidden_information()
        self.state = state
        self.predictor = predictor
        self.config = config
        self.activity_prior = {str(key): float(value) for key, value in (activity_prior or {}).items()}
        self.predictions: dict[Task, ScorePrediction] = predictor.predict(state.unfinished_tasks())
        self._draw_cache: dict[Task, np.ndarray] = {}
        self._value_cache: dict[tuple[Task, ...], float] = {}

    def _draws_for(self, task: Task) -> np.ndarray:
        task = tuple(task)
        if task not in self._draw_cache:
            draws = self.predictor.sample(
                [task], random_state=_stable_int(self.config.random_seed, task)
            )
            self._draw_cache[task] = np.asarray(
                [sample[task] for sample in draws[: self.config.monte_carlo_samples]],
                dtype=float,
            )
            if len(self._draw_cache[task]) < self.config.monte_carlo_samples:
                self._draw_cache[task] = np.resize(
                    self._draw_cache[task], self.config.monte_carlo_samples
                )
        return self._draw_cache[task]

    def _visible_scores(self) -> dict[str, list[float]]:
        scores: dict[str, list[float]] = {}
        for (ligand_id, _), score in self.state.observed_scores.items():
            scores.setdefault(ligand_id, []).append(score)
        return scores

    def _raw_utility(self, tasks: tuple[Task, ...]) -> float:
        tasks = tuple(sorted(tasks))
        if tasks in self._value_cache:
            return self._value_cache[tasks]
        visible = self._visible_scores()
        ligand_ids = sorted(str(row["ligand_id"]) for row in self.state.ligand_manifest)
        task_draws = {task: self._draws_for(task) for task in tasks}
        sample_count = self.config.monte_carlo_samples
        utilities: list[float] = []
        fused_by_ligand: dict[str, np.ndarray] = {}
        for ligand_id in ligand_ids:
            values = np.asarray(visible.get(ligand_id, []), dtype=float)
            fused = np.tile(np.mean(values), sample_count) if len(values) else np.full(sample_count, np.inf)
            selected = [task for task in tasks if task[0] == ligand_id]
            if selected:
                all_values = [fused]
                for task in selected:
                    all_values.append(task_draws[task][:sample_count])
                fused = np.mean(np.vstack(all_values), axis=0)
            fused_by_ligand[ligand_id] = fused
        for sample_index in range(sample_count):
            ranking = sorted(
                ligand_ids,
                key=lambda ligand_id: (fused_by_ligand[ligand_id][sample_index], ligand_id),
            )
            top = ranking[: min(self.config.top_q, len(ranking))]
            if self.config.utility_mode == "activity_prior" and self.activity_prior:
                utility = sum(
                    self.activity_prior.get(ligand_id, 0.0) / (rank + 1)
                    for rank, ligand_id in enumerate(ranking[: len(top)])
                )
            else:
                utility = -float(np.mean([fused_by_ligand[ligand_id][sample_index] for ligand_id in top])) if top else 0.0
            if self.config.risk_lambda:
                risk = float(np.mean([np.std(fused_by_ligand[ligand_id]) for ligand_id in top])) if top else 0.0
                utility -= self.config.risk_lambda * risk
            utilities.append(utility)
        result = float(np.mean(utilities)) if utilities else 0.0
        self._value_cache[tasks] = result
        return result

    def fusion_utility(self, tasks: Sequence[Task]) -> float:
        """Return F_t(S), the expected utility before subtracting F_t(empty)."""
        return self._raw_utility(tuple(tasks))

    def set_value(self, tasks: Sequence[Task]) -> float:
        tasks_tuple = tuple(sorted(tuple(task) for task in tasks))
        return self._raw_utility(tasks_tuple) - self._raw_utility(())

    def marginal_value(self, tasks: Sequence[Task]) -> float:
        return self.set_value(tasks)

    def task_value(self, task: Task) -> float:
        return self.set_value((task,))

    def unit_value(self, task: Task) -> float:
        return self.task_value(task) / self.state.cost_for(task)

    def pairwise_interaction(self, first: Task, second: Task) -> float:
        if tuple(first) == tuple(second):
            raise ValueError("pairwise interaction requires distinct tasks")
        return self.set_value((first, second)) - self.set_value((first,)) - self.set_value((second,))

    def interaction_matrix(
        self,
        tasks: Sequence[Task],
        *,
        progress: Callable[[int, int], None] | None = None,
    ) -> dict[tuple[Task, Task], float]:
        ordered = tuple(sorted(tuple(task) for task in tasks))
        total = len(ordered) * (len(ordered) - 1) // 2
        report_interval = max(1, math.ceil(total / 20)) if total else 1
        interactions: dict[tuple[Task, Task], float] = {}
        completed = 0
        for index, first in enumerate(ordered):
            for second in ordered[index + 1 :]:
                interactions[(first, second)] = self.pairwise_interaction(first, second)
                completed += 1
                if progress is not None and (
                    completed == 1 or completed == total or completed % report_interval == 0
                ):
                    progress(completed, total)
        return interactions

    def all_task_values(self, tasks: Sequence[Task] | None = None) -> dict[Task, float]:
        candidates = self.state.unfinished_tasks() if tasks is None else tuple(tasks)
        return {task: self.task_value(task) for task in candidates}
