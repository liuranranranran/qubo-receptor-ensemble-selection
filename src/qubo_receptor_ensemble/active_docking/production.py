"""Resumable production workflow built on the canonical prepare stage."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

from ..docking_adapters import get_docking_adapter
from ..experiment import prepare_experiment_inputs
from ..io import file_sha256, write_json
from .acquisition import AcquisitionConfig, PosteriorAcquisitionEvaluator
from .executor import SelectedTaskExecutor
from .manifest_bridge import (
    ActiveManifestBridgeResult,
    build_active_manifests,
    read_evaluation_labels,
)
from .production_config import ActiveProductionConfig
from .qubo import BatchConstraints, build_batch_qubo
from .replay import _candidate_pool, _final_evaluation, _make_predictor
from .solvers import solve_batch_qubo
from .state import PartialObservationState, StateError, Task
from .warm_start import WarmStartConfig, plan_warm_start


class ProductionRunError(RuntimeError):
    """Raised when a production active run cannot safely continue or resume."""


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    write_json(temporary, value)
    os.replace(temporary, path)


def _canonical_digest(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _prepared_fingerprint(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(path.name.encode("utf-8"))
        digest.update(file_sha256(path).encode("ascii"))
    return digest.hexdigest()


def _active_constraints(
    tasks: Sequence[Task],
    state: PartialObservationState,
    config: Mapping[str, object],
    budget: float,
) -> BatchConstraints:
    raw = config.get("constraints", {})
    if not isinstance(raw, Mapping):
        raise ProductionRunError("constraints must be an object")
    return BatchConstraints(
        budget=budget,
        task_costs={task: state.cost_for(task) for task in tasks},
        max_per_ligand=raw.get("max_per_ligand"),
        max_per_receptor=raw.get("max_per_receptor"),
        max_per_scaffold=raw.get("max_per_scaffold"),
        ligand_scaffolds={
            str(row["ligand_id"]): str(
                row.get("scaffold", row.get("scaffold_smiles", "__unknown__"))
            )
            for row in state.ligand_manifest
        },
        receptor_activation_cost=raw.get("receptor_activation_cost", 0.0),
        penalty=float(raw.get("penalty", 10.0)),
        cost_unit=raw.get("cost_unit"),
        equal_cost=bool(raw.get("equal_cost", False)),
        coefficient_scale=float(raw.get("coefficient_scale", 1.0)),
    )


class ActiveProductionRunner:
    """Run prepare, warm-start, active rounds, and final evaluation independently."""

    def __init__(self, config: ActiveProductionConfig, *, adapter: object | None = None) -> None:
        self.config = config
        self.adapter = adapter or get_docking_adapter(config.base_config.data)
        self._state: PartialObservationState | None = None
        self._manifests: ActiveManifestBridgeResult | None = None
        self._prepared_paths: tuple[Path, Path] | None = None
        self._prepared_input_fingerprint: str | None = None
        self._rounds: list[dict[str, object]] = []
        self._task_sequence: list[Task] = []

    @property
    def state(self) -> PartialObservationState:
        if self._state is None:
            raise ProductionRunError("active run has not been initialized")
        return self._state

    @property
    def manifests(self) -> ActiveManifestBridgeResult:
        if self._manifests is None:
            raise ProductionRunError("active run manifests have not been initialized")
        return self._manifests

    def prepare(self, *, resume: bool = False, overwrite: bool = False) -> dict[str, object]:
        """Invoke the old canonical prepare stage and record its output boundary."""
        prepared = prepare_experiment_inputs(
            self.config.base_config, resume=resume, overwrite=overwrite
        )
        if not isinstance(prepared, Mapping):
            raise ProductionRunError("canonical prepare returned an invalid context")
        return {
            "status": "completed",
            "prepared_run_directory": str(self.config.prepared_run_directory),
            "prepared_ligand_manifest": str(
                self.config.prepared_run_directory / "prepared_ligands.csv"
            ),
            "selected_receptor_manifest": str(
                self.config.prepared_run_directory / "selected_receptors.csv"
            ),
            "canonical_prepare": {
                str(key): str(value) if isinstance(value, Path) else value
                for key, value in prepared.items()
                if key not in {"ligands", "receptors"}
            },
        }

    def initialize(
        self,
        *,
        resume: bool = False,
        overwrite: bool = False,
    ) -> dict[str, object]:
        """Load prepared inputs, execute deterministic warm-start, and checkpoint."""
        if resume:
            self._load_checkpoint()
            return self._summary(status="initialized")
        if self.config.active_run_directory.exists() and any(
            (self.config.active_run_directory / name).is_file()
            for name in ("state.json", "run_metadata.json")
        ):
            if not overwrite:
                raise ProductionRunError(
                    "active run already exists; use resume=True or overwrite=True"
                )
        ligand_path, receptor_path = self._find_prepared_manifests()
        self._prepared_paths = (ligand_path, receptor_path)
        self._prepared_input_fingerprint = _prepared_fingerprint(self._prepared_paths)
        self._manifests = build_active_manifests(
            ligand_path,
            receptor_path,
            data_root=self.config.data_root,
            baseline_receptor=self.config.baseline_receptor,
            cluster_threshold_angstrom=float(
                self.config.data["receptor_cluster"]["threshold_angstrom"]
            ),
        )
        self.config.active_run_directory.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(
            self.config.active_run_directory / "active_manifest.json",
            {
                "ligands": self.manifests.ligands,
                "receptors": self.manifests.receptors,
                "receptor_cluster_distances": self.manifests.receptor_cluster_distances,
            },
        )
        _atomic_write_json(
            self.config.active_run_directory / "run_metadata.json",
            {
                "workflow": self.config.workflow,
                "config_fingerprint": self.config.fingerprint,
                "prepared_input_fingerprint": self._prepared_input_fingerprint,
                "prepared_run_directory": str(self.config.prepared_run_directory),
                "active_run_directory": str(self.config.active_run_directory),
                "real_docking_executed": False,
                "quantum_hardware_used": False,
            },
        )
        warm = self.config.data["warm_start"]
        assert isinstance(warm, Mapping)
        warm_tasks = plan_warm_start(
            self.manifests.ligands,
            self.manifests.receptors,
            WarmStartConfig(
                baseline_receptor=str(warm["baseline_receptor"]),
                cluster_fraction=float(warm["cluster_fraction"]),
                min_ligands_per_cluster=int(warm["min_ligands_per_cluster"]),
                random_seed=int(warm["random_seed"]),
            ),
        )
        all_tasks = {
            (str(ligand["ligand_id"]), str(receptor["receptor_id"]))
            for ligand in self.manifests.ligands
            for receptor in self.manifests.receptors
        }
        task_cost = len(self.config.docking_seeds) * self.config.cost_per_seed
        task_costs = {task: task_cost for task in all_tasks}
        self._state = PartialObservationState(
            ligand_manifest=[dict(row) for row in self.manifests.ligands],
            receptor_manifest=[dict(row) for row in self.manifests.receptors],
            candidate_tasks=all_tasks,
            task_costs=task_costs,
            warm_start_state={
                "strategy": "fixed_label_free",
                "planned_tasks": [list(task) for task in warm_tasks],
                "completed": False,
            },
            scaffold_metadata={
                str(row["ligand_id"]): str(row["scaffold"])
                for row in self.manifests.ligands
            },
            receptor_cluster_metadata={
                str(row["receptor_id"]): str(row["cluster"])
                for row in self.manifests.receptors
            },
        )
        if not set(warm_tasks).issubset(all_tasks):
            raise ProductionRunError("warm-start selected a task outside the candidate set")
        warm_results = self._executor().execute(
            tasks=warm_tasks,
            ligand_manifest=self.manifests.ligands,
            receptor_manifest=self.manifests.receptors,
            output_directory=self.config.active_run_directory / "warm_start",
        )
        self._state.reveal({task: result.fused_score for task, result in warm_results.items()})
        self._state.warm_start_state["completed"] = True
        _atomic_write_json(
            self.config.active_run_directory / "warm_start.json",
            {
                "strategy": "fixed_label_free",
                "tasks": [list(task) for task in warm_tasks],
                "task_cost": task_cost,
                "total_cost": sum(result.cost for result in warm_results.values()),
                "docking_seeds": list(self.config.docking_seeds),
                "score_fusion": self.config.score_fusion,
            },
        )
        metadata_path = self.config.active_run_directory / "run_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="ascii"))
        metadata["real_docking_executed"] = True
        _atomic_write_json(metadata_path, metadata)
        self._save_state()
        return self._summary(status="initialized")

    def run(
        self,
        *,
        resume: bool = False,
        overwrite: bool = False,
        max_rounds: int | None = None,
    ) -> dict[str, object]:
        """Run active rounds until budget, candidate exhaustion, or a stop rule."""
        if resume:
            self._load_checkpoint()
        elif self._state is None:
            self.initialize(overwrite=overwrite)
        limit = max_rounds
        if limit is None:
            stop = self.config.data.get("stop", {})
            limit = int(stop.get("max_rounds", 1000)) if isinstance(stop, Mapping) else 1000
        if limit < 0:
            raise ProductionRunError("max_rounds must be non-negative")
        active_rounds = 0
        stop_reason = "budget_exhausted"
        while active_rounds < limit:
            if self.state.docking_cost >= self.config.total_budget - 1e-9:
                stop_reason = "budget_exhausted"
                break
            available = self.state.unfinished_tasks()
            if not available:
                stop_reason = "candidate_exhausted"
                break
            predictor = _make_predictor(self.config.data, self.state)
            acquisition = self.config.data["acquisition"]
            assert isinstance(acquisition, Mapping)
            evaluator = PosteriorAcquisitionEvaluator(
                self.state,
                predictor,
                AcquisitionConfig(
                    top_q=int(acquisition["top_q"]),
                    monte_carlo_samples=int(acquisition["monte_carlo_samples"]),
                    risk_lambda=float(acquisition["risk_lambda"]),
                    utility_mode=str(acquisition["utility_mode"]),
                    random_seed=self._round_seed(),
                ),
            )
            pool = _candidate_pool(self.state, predictor, self.config.data)
            available_pool = tuple(task for task in pool if task in available)
            if not available_pool:
                stop_reason = "candidate_pool_exhausted"
                break
            remaining_budget = min(
                self.config.batch_budget,
                self.config.total_budget - self.state.docking_cost,
            )
            constraints = _active_constraints(
                available_pool, self.state, self.config.data, remaining_budget
            )
            values = evaluator.all_task_values(available_pool)
            interactions = evaluator.interaction_matrix(available_pool)
            qubo = build_batch_qubo(
                available_pool,
                values,
                interactions,
                constraints,
                batch_interaction_weight=float(acquisition["batch_interaction_weight"]),
            )
            solver_config = self.config.data["solver"]
            assert isinstance(solver_config, Mapping)
            backend = str(self.config.data["strategy"])
            solver_result = solve_batch_qubo(
                qubo,
                backend=backend,
                random_seed=self._round_seed(),
                time_budget_seconds=float(solver_config["time_budget_seconds"]),
            )
            selected = tuple(task for task in solver_result.tasks if task in available_pool)
            if not selected:
                stop_reason = "no_feasible_or_positive_batch"
                break
            if not qubo.validate_tasks(selected).is_feasible:
                raise ProductionRunError("solver returned an infeasible selected task set")
            round_index = len(self._rounds)
            execution = self._executor().execute(
                tasks=selected,
                ligand_manifest=self.manifests.ligands,
                receptor_manifest=self.manifests.receptors,
                output_directory=self.config.active_run_directory / f"round_{round_index:03d}",
            )
            self.state.reveal({task: result.fused_score for task, result in execution.items()})
            self._task_sequence.extend(selected)
            solver_audit = solver_result.as_dict()
            solver_audit["metadata"] = {
                key: value
                for key, value in solver_result.metadata.items()
                if key != "solver_time_seconds"
            }
            audit = {
                "round": round_index,
                "candidate_pool": [list(task) for task in pool],
                "available_tasks": [list(task) for task in available],
                "selected_tasks": [list(task) for task in selected],
                "revealed_tasks": [list(task) for task in selected],
                "predicted_values": {f"{task[0]}||{task[1]}": values[task] for task in available_pool},
                "prediction_variances": {
                    f"{task[0]}||{task[1]}": evaluator.predictions[task].variance
                    for task in available_pool
                },
                "solver": solver_audit,
                "qubo_fingerprint": qubo.fingerprint,
                "observed_task_count_after": len(self.state.observed_scores),
                "docking_cost_after": self.state.docking_cost,
                "real_docking_executed": True,
            }
            self._rounds.append(audit)
            self._write_round_and_checkpoint(round_index, audit)
            active_rounds += 1
        else:
            stop_reason = "round_limit"
        return self._summary(status="completed", stop_reason=stop_reason)

    def finalize(self) -> dict[str, object]:
        """Evaluate the final ranking; this is the only production label boundary."""
        if self._state is None:
            self._load_checkpoint()
        labels = read_evaluation_labels(self._prepared_ligand_path())
        predictor = _make_predictor(self.config.data, self.state)
        evaluation = _final_evaluation(
            self.state,
            predictor,
            labels,
            [str(item) for item in self.config.data.get("evaluation", {}).get("metrics", [])]
            if isinstance(self.config.data.get("evaluation"), Mapping)
            else [],
        )
        output = {
            "workflow": self.config.workflow,
            "evaluation": evaluation,
            "hidden_labels_used_only_for_final_evaluation": True,
            "real_docking_executed": True,
        }
        _atomic_write_json(self.config.active_run_directory / "evaluation.json", output)
        return output

    def _executor(self) -> SelectedTaskExecutor:
        return SelectedTaskExecutor(
            adapter=self.adapter,
            data_root=self.config.data_root,
            target_id=str(self.config.base_config.data["target_id"]),
            seeds=self.config.docking_seeds,
            score_fusion=self.config.score_fusion,
            cost_per_seed=self.config.cost_per_seed,
            docking_config=self.config.base_config.data,
            resume=True,
        )

    def _round_seed(self) -> int:
        return int(self.config.data["random_seed"]) + len(self._rounds)

    def _prepared_ligand_path(self) -> Path:
        return self._find_prepared_manifests()[0]

    def _find_prepared_manifests(self) -> tuple[Path, Path]:
        prepared = self.config.prepared_run_directory
        ligand = prepared / "prepared_ligands.csv"
        receptor = prepared / "selected_receptors.csv"
        base_paths = getattr(self.config.base_config, "paths", {})
        if not ligand.is_file() and isinstance(base_paths, Mapping):
            candidate = base_paths.get("prepared_ligand_manifest")
            if isinstance(candidate, Path):
                ligand = candidate
        if not receptor.is_file() and isinstance(base_paths, Mapping):
            candidate = base_paths.get("selected_receptor_manifest")
            if isinstance(candidate, Path):
                receptor = candidate
        if not ligand.is_file() or not receptor.is_file():
            missing = [str(path) for path in (ligand, receptor) if not path.is_file()]
            if not missing:
                missing = [str(prepared)]
            raise FileNotFoundError(f"prepared active-docking manifests are missing: {missing}")
        return ligand.resolve(), receptor.resolve()

    def _save_state(self) -> None:
        _atomic_write_json(self.config.active_run_directory / "state.json", self.state.to_dict())

    def _write_round_and_checkpoint(self, round_index: int, audit: Mapping[str, object]) -> None:
        round_path = self.config.active_run_directory / f"round_{round_index:03d}.json"
        _atomic_write_json(round_path, dict(audit))
        self._save_state()
        _atomic_write_json(
            self.config.active_run_directory / "run_summary.json",
            self._summary(status="running"),
        )

    def _load_checkpoint(self) -> None:
        active = self.config.active_run_directory
        metadata_path = active / "run_metadata.json"
        state_path = active / "state.json"
        manifest_path = active / "active_manifest.json"
        if not metadata_path.is_file() or not state_path.is_file() or not manifest_path.is_file():
            raise ProductionRunError("active checkpoint is incomplete")
        metadata = json.loads(metadata_path.read_text(encoding="ascii"))
        if metadata.get("config_fingerprint") != self.config.fingerprint:
            raise ProductionRunError("active checkpoint config fingerprint differs")
        ligand_path, receptor_path = self._find_prepared_manifests()
        prepared_fingerprint = _prepared_fingerprint((ligand_path, receptor_path))
        if metadata.get("prepared_input_fingerprint") != prepared_fingerprint:
            raise ProductionRunError("active checkpoint prepared input fingerprint differs")
        manifest = json.loads(manifest_path.read_text(encoding="ascii"))
        if not isinstance(manifest, Mapping):
            raise ProductionRunError("active manifest checkpoint is invalid")
        try:
            self._manifests = ActiveManifestBridgeResult(
                ligands=[dict(row) for row in manifest["ligands"]],
                receptors=[dict(row) for row in manifest["receptors"]],
                receptor_cluster_distances={
                    str(key): dict(value)
                    for key, value in manifest.get("receptor_cluster_distances", {}).items()
                },
            )
            self._state = PartialObservationState.from_dict(
                json.loads(state_path.read_text(encoding="ascii"))
            )
        except (KeyError, TypeError, ValueError, StateError) as exc:
            raise ProductionRunError("active checkpoint state is invalid") from exc
        self._prepared_paths = (ligand_path, receptor_path)
        self._prepared_input_fingerprint = prepared_fingerprint
        self._rounds = []
        self._task_sequence = []
        for round_path in sorted(active.glob("round_*.json")):
            audit = json.loads(round_path.read_text(encoding="ascii"))
            if not isinstance(audit, Mapping):
                raise ProductionRunError(f"invalid round checkpoint: {round_path}")
            normalized = dict(audit)
            self._rounds.append(normalized)
            for task in normalized.get("selected_tasks", []):
                if isinstance(task, list) and len(task) == 2:
                    self._task_sequence.append((str(task[0]), str(task[1])))

    def _summary(self, *, status: str, stop_reason: str | None = None) -> dict[str, object]:
        result: dict[str, object] = {
            "workflow": self.config.workflow,
            "status": status,
            "prepare_stage": "canonical_full_workflow_prepare_output",
            "config_fingerprint": self.config.fingerprint,
            "prepared_run_directory": str(self.config.prepared_run_directory),
            "active_run_directory": str(self.config.active_run_directory),
            "task_sequence": tuple(self._task_sequence),
            "rounds": tuple(self._rounds),
            "observed_task_count": len(self.state.observed_scores) if self._state else 0,
            "docking_cost": self.state.docking_cost if self._state else 0.0,
            "real_docking_executed": bool(self._task_sequence or self._state),
            "quantum_hardware_used": False,
        }
        if stop_reason is not None:
            result["stop_reason"] = stop_reason
        return result
