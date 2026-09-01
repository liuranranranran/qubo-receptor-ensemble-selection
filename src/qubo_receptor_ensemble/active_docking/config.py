"""Independent configuration contract for masked active-docking replay."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


class ActiveDockingConfigError(ValueError):
    """Raised when an active-docking configuration is unsafe or incomplete."""


@dataclass(frozen=True)
class ActiveDockingConfig:
    path: Path
    data: dict[str, object]
    artifact_output_directory: Path

    @property
    def workflow(self) -> str:
        return str(self.data["workflow"])

    @property
    def strategies(self) -> tuple[str, ...]:
        raw = self.data.get("strategies", [str(self.data.get("solver", {}).get("backend", "exact"))])
        return tuple(str(item) for item in raw)


_ALLOWED_BACKENDS = {
    "exact",
    "value_greedy",
    "greedy",
    "greedy_one_swap",
    "greedy+one_swap",
    "simulated_annealing",
    "quantum_compatible_simulator",
    "quantum_compatible",
    "random",
    "receptor_round_robin",
}
_ALLOWED_METRICS = {"bedroc20", "pr_auc", "ef1", "ef5", "roc_auc", "mae", "rmse"}


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ActiveDockingConfigError(f"{name} must be an object")
    return value


def _positive_number(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ActiveDockingConfigError(f"{name} must be positive")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ActiveDockingConfigError(f"{name} must be positive") from exc
    if result <= 0 or not math.isfinite(result):
        raise ActiveDockingConfigError(f"{name} must be positive and finite")
    return result


def validate_active_docking_config(config: Mapping[str, object]) -> None:
    if config.get("schema_version") != "1.0":
        raise ActiveDockingConfigError("schema_version must be 1.0")
    if config.get("workflow") != "masked_active_docking_replay":
        raise ActiveDockingConfigError("workflow must be masked_active_docking_replay; real docking is not enabled")
    seed = config.get("random_seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ActiveDockingConfigError("random_seed must be an integer")
    warm = _mapping(config.get("warm_start"), "warm_start")
    if not str(warm.get("baseline_receptor", "")):
        raise ActiveDockingConfigError("warm_start.baseline_receptor is required")
    if float(warm.get("cluster_fraction", 0.0)) < 0 or float(warm.get("cluster_fraction", 0.0)) > 1:
        raise ActiveDockingConfigError("warm_start.cluster_fraction must be between 0 and 1")
    predictor = _mapping(config.get("predictor"), "predictor")
    if not str(predictor.get("baseline_receptor", warm["baseline_receptor"])):
        raise ActiveDockingConfigError("predictor.baseline_receptor is required")
    posterior_samples = predictor.get("posterior_samples", 64)
    if isinstance(posterior_samples, bool) or not isinstance(posterior_samples, int) or posterior_samples <= 0:
        raise ActiveDockingConfigError("predictor.posterior_samples must be a positive integer")
    acquisition = _mapping(config.get("acquisition"), "acquisition")
    for key in ("top_q", "monte_carlo_samples"):
        value = acquisition.get(key, 64 if key == "monte_carlo_samples" else 10)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ActiveDockingConfigError(f"acquisition.{key} must be a positive integer")
    _positive_number(acquisition.get("risk_lambda", 0.0) or 1e-12, "acquisition.risk_lambda")
    candidate_cap = config.get("candidate_cap")
    if isinstance(candidate_cap, bool) or not isinstance(candidate_cap, int) or candidate_cap <= 0:
        raise ActiveDockingConfigError("candidate_cap must be a positive integer")
    budget = _mapping(config.get("budget"), "budget")
    _positive_number(budget.get("total_cost"), "budget.total_cost")
    _positive_number(budget.get("batch_cost"), "budget.batch_cost")
    constraints = _mapping(config.get("constraints"), "constraints")
    for key in ("max_per_ligand", "max_per_receptor", "max_per_scaffold"):
        value = constraints.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise ActiveDockingConfigError(f"constraints.{key} must be a non-negative integer or null")
    solver = _mapping(config.get("solver", {}), "solver")
    backends = list(config.get("strategies", [solver.get("backend", "exact")]))
    if not backends or any(str(item) not in _ALLOWED_BACKENDS for item in backends):
        raise ActiveDockingConfigError(f"strategies must use supported backends: {sorted(_ALLOWED_BACKENDS)}")
    evaluation = _mapping(config.get("evaluation"), "evaluation")
    metrics = evaluation.get("metrics")
    if not isinstance(metrics, list) or not metrics or any(str(metric) not in _ALLOWED_METRICS for metric in metrics):
        raise ActiveDockingConfigError(f"evaluation.metrics must be a non-empty list from {sorted(_ALLOWED_METRICS)}")
    mask_strategy = str(config.get("replay_mask_strategy", "scaffold_cluster"))
    if mask_strategy not in {"random", "scaffold_cluster"}:
        raise ActiveDockingConfigError("replay_mask_strategy must be random or scaffold_cluster")


def load_active_docking_config(path: str | Path) -> ActiveDockingConfig:
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ActiveDockingConfigError(f"invalid JSON: {config_path}") from exc
    if not isinstance(value, dict):
        raise ActiveDockingConfigError("config JSON root must be an object")
    validate_active_docking_config(value)
    output = Path(str(value.get("artifact_output_directory", "results/active_docking")))
    if not output.is_absolute():
        output = config_path.parent.parent.parent / output
    return ActiveDockingConfig(path=config_path, data=dict(value), artifact_output_directory=output.resolve())
