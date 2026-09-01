"""Selected-task docking executor with deterministic seed fusion."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Mapping, Protocol, Sequence

from ..io import safe_filename, write_json
from ..docking_adapters import DockingAdapter
from .state import Task


class TaskExecutionError(RuntimeError):
    """Raised when selected docking tasks cannot be fused safely."""


@dataclass(frozen=True)
class TaskExecutionResult:
    """Fused score and auditable cost for one selected ligand-receptor task."""

    task: Task
    seed_scores: dict[int, float]
    fused_score: float
    cost: float


class _BatchAdapter(Protocol):
    name: str

    def run_batch(self, **kwargs: object) -> list[dict[str, object]]:
        ...


_HIDDEN_KEYS = {"label", "active", "decoy", "hidden_score", "hidden_label"}


def _rooted(path: object, root: Path) -> Path:
    value = Path(str(path))
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _task(value: object) -> Task:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise TaskExecutionError(f"invalid selected task: {value!r}")
    ligand_id, receptor_id = str(value[0]), str(value[1])
    if not ligand_id or not receptor_id:
        raise TaskExecutionError("selected task identifiers must be non-empty")
    return ligand_id, receptor_id


def _reject_hidden(row: Mapping[str, object], where: str) -> None:
    for key in row:
        if str(key).lower() in _HIDDEN_KEYS:
            raise TaskExecutionError(f"hidden evaluation field {key!r} is not allowed in {where}")


class SelectedTaskExecutor:
    """Execute exactly the selected task set for every configured seed."""

    def __init__(
        self,
        *,
        adapter: _BatchAdapter | DockingAdapter,
        data_root: str | Path,
        target_id: str,
        seeds: Sequence[int],
        score_fusion: str = "median",
        cost_per_seed: float = 1.0,
        docking_config: Mapping[str, object] | None = None,
        resume: bool = False,
    ) -> None:
        normalized_seeds = tuple(int(seed) for seed in seeds)
        if not normalized_seeds or len(set(normalized_seeds)) != len(normalized_seeds):
            raise ValueError("seeds must be a non-empty sequence of unique integers")
        if score_fusion != "median":
            raise ValueError("only median score fusion is supported")
        if not math.isfinite(float(cost_per_seed)) or float(cost_per_seed) <= 0:
            raise ValueError("cost_per_seed must be positive and finite")
        if not target_id:
            raise ValueError("target_id must be non-empty")
        self.adapter = adapter
        self.data_root = Path(data_root).resolve()
        self.target_id = target_id
        self.seeds = normalized_seeds
        self.score_fusion = score_fusion
        self.cost_per_seed = float(cost_per_seed)
        self.docking_config = dict(docking_config or {})
        self.resume = resume

    def execute(
        self,
        *,
        tasks: Sequence[Task],
        ligand_manifest: Sequence[Mapping[str, object]],
        receptor_manifest: Sequence[Mapping[str, object]],
        output_directory: str | Path,
    ) -> dict[Task, TaskExecutionResult]:
        selected = tuple(_task(item) for item in tasks)
        if not selected:
            raise TaskExecutionError("at least one selected task is required")
        if len(set(selected)) != len(selected):
            raise TaskExecutionError("selected tasks contain duplicates")
        ligands = {str(row.get("ligand_id", "")): dict(row) for row in ligand_manifest}
        receptors = {str(row.get("receptor_id", row.get("conformer_id", ""))): dict(row) for row in receptor_manifest}
        if any(not identifier for identifier in ligands):
            raise TaskExecutionError("ligand manifest contains an empty ligand_id")
        if any(not identifier for identifier in receptors):
            raise TaskExecutionError("receptor manifest contains an empty receptor_id")
        if len(ligands) != len(ligand_manifest):
            raise TaskExecutionError("ligand manifest contains duplicate IDs")
        if len(receptors) != len(receptor_manifest):
            raise TaskExecutionError("receptor manifest contains duplicate IDs")
        for row in (*ligands.values(), *receptors.values()):
            _reject_hidden(row, "active docking manifest")
        missing_ligands = sorted({ligand for ligand, _ in selected} - set(ligands))
        missing_receptors = sorted({receptor for _, receptor in selected} - set(receptors))
        if missing_ligands or missing_receptors:
            raise TaskExecutionError(
                f"selected tasks reference unknown IDs: ligands={missing_ligands}, receptors={missing_receptors}"
            )

        root = Path(output_directory).resolve()
        root.mkdir(parents=True, exist_ok=True)
        by_receptor: dict[str, list[str]] = {}
        for ligand_id, receptor_id in selected:
            by_receptor.setdefault(receptor_id, []).append(ligand_id)
        seed_scores: dict[Task, dict[int, float]] = {task: {} for task in selected}
        selected_ligand_ids = {ligand_id for ligand_id, _ in selected}
        for receptor_id in sorted(by_receptor):
            receptor = receptors[receptor_id]
            receptor_path = _rooted(receptor.get("receptor_pdbqt"), self.data_root)
            receptor_ligands = [ligands[ligand_id] for ligand_id in sorted(by_receptor[receptor_id])]
            for seed in self.seeds:
                seed_directory = root / f"receptor_{safe_filename(receptor_id)}" / f"seed_{seed}"
                score_table = seed_directory / "scores.csv"
                rows = self.adapter.run_batch(
                    target_id=self.target_id,
                    receptor_id=receptor_id,
                    receptor_path=receptor_path,
                    ligands=receptor_ligands,
                    seed=seed,
                    output_dir=seed_directory,
                    score_table=score_table,
                    config=self.docking_config,
                    root=self.data_root,
                    resume=self.resume,
                )
                self._collect_seed_rows(
                    rows,
                    receptor_id=receptor_id,
                    selected_ligand_ids=set(by_receptor[receptor_id]),
                    seed=seed,
                    seed_scores=seed_scores,
                )
        if set(seed_scores) != set(selected) or any(
            len(scores) != len(self.seeds) for scores in seed_scores.values()
        ):
            raise TaskExecutionError("one or more selected tasks are missing seed results")

        results: dict[Task, TaskExecutionResult] = {}
        for task in selected:
            scores = dict(sorted(seed_scores[task].items()))
            fused = float(median(scores.values()))
            if not math.isfinite(fused):
                raise TaskExecutionError(f"fused score is not finite for task: {task}")
            results[task] = TaskExecutionResult(
                task=task,
                seed_scores=scores,
                fused_score=fused,
                cost=len(self.seeds) * self.cost_per_seed,
            )
        self._write_audit(root, results)
        del selected_ligand_ids
        return results

    @staticmethod
    def _collect_seed_rows(
        rows: Sequence[Mapping[str, object]],
        *,
        receptor_id: str,
        selected_ligand_ids: set[str],
        seed: int,
        seed_scores: dict[Task, dict[int, float]],
    ) -> None:
        seen: set[str] = set()
        for row in rows:
            _reject_hidden(row, "docking score row")
            ligand_id = str(row.get("ligand_id", ""))
            row_receptor = str(row.get("receptor_id", ""))
            if ligand_id not in selected_ligand_ids or row_receptor != receptor_id:
                raise TaskExecutionError(
                    f"adapter returned an unselected or mismatched task: {ligand_id}/{row_receptor}"
                )
            if ligand_id in seen:
                raise TaskExecutionError(f"duplicate score row for task: {(ligand_id, receptor_id)}")
            seen.add(ligand_id)
            row_seed = row.get("seed", seed)
            try:
                if int(row_seed) != seed:
                    raise ValueError
                score = float(row.get("docking_score"))
            except (TypeError, ValueError) as exc:
                raise TaskExecutionError(
                    f"invalid score row for {(ligand_id, receptor_id)} at seed {seed}"
                ) from exc
            if not math.isfinite(score) or str(row.get("status", "ok")) != "ok":
                raise TaskExecutionError(
                    f"non-finite or failed docking score for {(ligand_id, receptor_id)} at seed {seed}"
                )
            seed_scores[(ligand_id, receptor_id)][seed] = score
        missing = sorted(selected_ligand_ids - seen)
        if missing:
            raise TaskExecutionError(f"missing score rows for seed {seed}: {missing}")

    @staticmethod
    def _write_audit(root: Path, results: Mapping[Task, TaskExecutionResult]) -> None:
        records = [
            {
                "ligand_id": task[0],
                "receptor_id": task[1],
                "seed_scores": {str(seed): score for seed, score in result.seed_scores.items()},
                "fused_score": result.fused_score,
                "cost": result.cost,
            }
            for task, result in sorted(results.items())
        ]
        write_json(root / "task_results.json", {"tasks": records})
