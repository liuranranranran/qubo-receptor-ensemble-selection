"""Configuration contract for the production active-docking workflow.

The active workflow owns its policy and output paths while reusing only the
canonical full-workflow preparation configuration as an input contract.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ..full_workflow import ConfigError as FullConfigError
from ..full_workflow import FullExperimentConfig, load_full_experiment_config


class ActiveProductionConfigError(ValueError):
    """Raised when a production active-docking config is unsafe or incomplete."""


@dataclass(frozen=True)
class ActiveProductionConfig:
    """Resolved active workflow configuration and its immutable input contract."""

    path: Path
    data_root: Path
    data: dict[str, object]
    base_config: FullExperimentConfig
    prepared_run_directory: Path
    active_run_directory: Path
    baseline_receptor: str
    fingerprint: str

    @property
    def workflow(self) -> str:
        return str(self.data["workflow"])

    @property
    def docking_seeds(self) -> tuple[int, ...]:
        docking = self.data["docking"]
        assert isinstance(docking, dict)
        return tuple(int(seed) for seed in docking["seeds"])

    @property
    def score_fusion(self) -> str:
        docking = self.data["docking"]
        assert isinstance(docking, dict)
        return str(docking["score_fusion"])

    @property
    def cost_per_seed(self) -> float:
        docking = self.data["docking"]
        assert isinstance(docking, dict)
        return float(docking["cost_per_seed"])

    @property
    def total_budget(self) -> float:
        budget = self.data["budget"]
        assert isinstance(budget, dict)
        return float(budget["total_cost"])

    @property
    def batch_budget(self) -> float:
        budget = self.data["budget"]
        assert isinstance(budget, dict)
        return float(budget["batch_cost"])


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ActiveProductionConfigError(f"{name} must be an object")
    return value


def _finite_number(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ActiveProductionConfigError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ActiveProductionConfigError(f"{name} must be numeric") from exc
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive and finite" if positive else "finite"
        raise ActiveProductionConfigError(f"{name} must be {qualifier}")
    return result


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ActiveProductionConfigError(f"{name} must be a positive integer")
    return value


def _resolve_path(value: object, root: Path, name: str) -> Path:
    if isinstance(value, Path):
        path = value
    elif isinstance(value, str) and value.strip():
        path = Path(value)
    else:
        raise ActiveProductionConfigError(f"{name} must be a non-empty path")
    return (path if path.is_absolute() else root / path).resolve()


def validate_active_production_config(config: Mapping[str, object]) -> None:
    """Validate active-only settings without changing the canonical config."""
    if config.get("schema_version") != "1.0":
        raise ActiveProductionConfigError("schema_version must be 1.0")
    if config.get("workflow") != "active_ligand_receptor_docking":
        raise ActiveProductionConfigError(
            "workflow must be active_ligand_receptor_docking"
        )
    if not isinstance(config.get("base_experiment_config"), str):
        raise ActiveProductionConfigError("base_experiment_config is required")

    baseline = config.get("baseline_receptor")
    if not isinstance(baseline, str) or not baseline.strip():
        raise ActiveProductionConfigError("baseline_receptor must be non-empty")
    if str(config.get("target_id", "MK14")).upper() == "MK14" and baseline != "11OY":
        raise ActiveProductionConfigError(
            "baseline_receptor for MK14 must be the real receptor ID 11OY"
        )

    cluster = _mapping(config.get("receptor_cluster"), "receptor_cluster")
    if cluster.get("method") != "aligned_ca_rmsd":
        raise ActiveProductionConfigError(
            "receptor_cluster.method must be aligned_ca_rmsd"
        )
    _finite_number(
        cluster.get("threshold_angstrom"),
        "receptor_cluster.threshold_angstrom",
        positive=True,
    )

    warm = _mapping(config.get("warm_start"), "warm_start")
    if warm.get("baseline_receptor") != baseline:
        raise ActiveProductionConfigError(
            "warm_start.baseline_receptor must match baseline_receptor"
        )
    fraction = _finite_number(warm.get("cluster_fraction"), "warm_start.cluster_fraction")
    if not 0.0 <= fraction <= 1.0:
        raise ActiveProductionConfigError(
            "warm_start.cluster_fraction must be between 0 and 1"
        )
    minimum = warm.get("min_ligands_per_cluster")
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 0:
        raise ActiveProductionConfigError(
            "warm_start.min_ligands_per_cluster must be a non-negative integer"
        )
    if isinstance(warm.get("random_seed"), bool) or not isinstance(warm.get("random_seed"), int):
        raise ActiveProductionConfigError("warm_start.random_seed must be an integer")

    docking = _mapping(config.get("docking"), "docking")
    seeds = docking.get("seeds")
    if (
        not isinstance(seeds, list)
        or len(seeds) != 3
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ActiveProductionConfigError(
            "docking.seeds must contain exactly three unique integers"
        )
    if docking.get("score_fusion") != "median":
        raise ActiveProductionConfigError("docking.score_fusion must be median")
    _finite_number(docking.get("cost_per_seed"), "docking.cost_per_seed", positive=True)

    budget = _mapping(config.get("budget"), "budget")
    _finite_number(budget.get("total_cost"), "budget.total_cost", positive=True)
    _finite_number(budget.get("batch_cost"), "budget.batch_cost", positive=True)

    candidate_cap = _positive_int(config.get("candidate_cap"), "candidate_cap")
    del candidate_cap
    constraints = _mapping(config.get("constraints"), "constraints")
    for key in ("max_per_ligand", "max_per_receptor", "max_per_scaffold"):
        value = constraints.get(key)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise ActiveProductionConfigError(
                f"constraints.{key} must be a non-negative integer or null"
            )
    _finite_number(constraints.get("penalty"), "constraints.penalty", positive=True)
    _finite_number(constraints.get("cost_unit"), "constraints.cost_unit", positive=True)
    _finite_number(constraints.get("coefficient_scale"), "constraints.coefficient_scale", positive=True)
    if not isinstance(constraints.get("equal_cost"), bool):
        raise ActiveProductionConfigError("constraints.equal_cost must be boolean")
    _finite_number(
        constraints.get("receptor_activation_cost", 0.0),
        "constraints.receptor_activation_cost",
    )
    if float(constraints.get("receptor_activation_cost", 0.0)) < 0.0:
        raise ActiveProductionConfigError("constraints.receptor_activation_cost must be non-negative")

    predictor = _mapping(config.get("predictor"), "predictor")
    if predictor.get("model") != "bayesian_residual":
        raise ActiveProductionConfigError("predictor.model must be bayesian_residual")
    if predictor.get("baseline_receptor") != baseline:
        raise ActiveProductionConfigError(
            "predictor.baseline_receptor must match baseline_receptor"
        )
    _positive_int(predictor.get("posterior_samples"), "predictor.posterior_samples")

    acquisition = _mapping(config.get("acquisition"), "acquisition")
    _positive_int(acquisition.get("top_q"), "acquisition.top_q")
    _positive_int(acquisition.get("monte_carlo_samples"), "acquisition.monte_carlo_samples")
    _finite_number(acquisition.get("risk_lambda"), "acquisition.risk_lambda")
    if acquisition.get("utility_mode") not in {"ranking_score", "information_gain"}:
        raise ActiveProductionConfigError("acquisition.utility_mode is unsupported")
    _finite_number(
        acquisition.get("batch_interaction_weight"),
        "acquisition.batch_interaction_weight",
    )

    solver = _mapping(config.get("solver"), "solver")
    if not isinstance(solver.get("backend"), str) or not solver["backend"].strip():
        raise ActiveProductionConfigError("solver.backend must be non-empty")
    _finite_number(
        solver.get("time_budget_seconds"),
        "solver.time_budget_seconds",
        positive=True,
    )
    strategy = config.get("strategy")
    if not isinstance(strategy, str) or not strategy.strip():
        raise ActiveProductionConfigError("strategy must be non-empty")
    if strategy != solver["backend"]:
        raise ActiveProductionConfigError("strategy must match solver.backend")

    prediction_gate = _mapping(config.get("prediction_gate"), "prediction_gate")
    if not isinstance(prediction_gate.get("required"), bool):
        raise ActiveProductionConfigError("prediction_gate.required must be boolean")
    stop = _mapping(config.get("stop"), "stop")
    _positive_int(stop.get("max_rounds"), "stop.max_rounds")
    seed = config.get("random_seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ActiveProductionConfigError("random_seed must be an integer")

    problem = config.get("problem")
    if isinstance(problem, Mapping) and problem.get("type") == "receptor_subset":
        raise ActiveProductionConfigError(
            "active workflow must not define the old receptor-subset problem"
        )


def load_active_production_config(
    path: str | Path,
    *,
    data_root: str | Path | None = None,
    prepared_run_directory: str | Path | None = None,
) -> ActiveProductionConfig:
    """Load active settings and validate the referenced full prepare contract."""
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ActiveProductionConfigError(f"invalid JSON: {config_path}") from exc
    if not isinstance(value, dict):
        raise ActiveProductionConfigError("config JSON root must be an object")
    validate_active_production_config(value)

    root = Path(data_root or Path.cwd()).resolve()
    base_path = _resolve_path(value["base_experiment_config"], config_path.parent, "base_experiment_config")
    try:
        base_config = load_full_experiment_config(base_path, data_root=root)
    except (FullConfigError, FileNotFoundError) as exc:
        raise ActiveProductionConfigError(
            f"invalid base_experiment_config: {base_path}"
        ) from exc

    prepared = _resolve_path(
        prepared_run_directory
        if prepared_run_directory is not None
        else value.get("prepared_run_directory"),
        root,
        "prepared_run_directory",
    )
    active = _resolve_path(value.get("active_run_directory"), root, "active_run_directory")
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    base_payload = json.dumps(base_config.data, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha256(
        (payload + "\n" + base_payload).encode("utf-8")
    ).hexdigest()
    return ActiveProductionConfig(
        path=config_path,
        data=json.loads(json.dumps(value)),
        data_root=root,
        base_config=base_config,
        prepared_run_directory=prepared,
        active_run_directory=active,
        baseline_receptor=str(value["baseline_receptor"]),
        fingerprint=fingerprint,
    )
