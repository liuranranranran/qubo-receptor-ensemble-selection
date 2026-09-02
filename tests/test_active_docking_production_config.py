from __future__ import annotations

import json
from pathlib import Path

import pytest

from qubo_receptor_ensemble.active_docking.production_config import (
    ActiveProductionConfigError,
    load_active_production_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO_ROOT / "configs" / "experiments" / "mk14_adaptive_remote.json"


def _write_config(tmp_path: Path, **overrides: object) -> Path:
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "workflow": "active_ligand_receptor_docking",
        "base_experiment_config": str(BASE_CONFIG),
        "prepared_run_directory": "results/runs/mk14_adaptive_remote",
        "active_run_directory": "results/runs/mk14_adaptive_remote/active_docking",
        "baseline_receptor": "11OY",
        "receptor_cluster": {
            "method": "aligned_ca_rmsd",
            "threshold_angstrom": 2.0,
        },
        "warm_start": {
            "baseline_receptor": "11OY",
            "cluster_fraction": 0.1,
            "min_ligands_per_cluster": 1,
            "random_seed": 20260901,
        },
        "docking": {
            "seeds": [20260821, 20260822, 20260823],
            "score_fusion": "median",
            "cost_per_seed": 1.0,
        },
        "budget": {"total_cost": 6000.0, "batch_cost": 300.0},
        "strategy": "quantum_compatible_simulator",
        "candidate_cap": 64,
        "constraints": {
            "max_per_ligand": 2,
            "max_per_receptor": 32,
            "max_per_scaffold": 4,
            "receptor_activation_cost": 0.0,
            "penalty": 25.0,
            "cost_unit": 0.001,
            "equal_cost": True,
            "coefficient_scale": 1.0,
        },
        "predictor": {
            "model": "bayesian_residual",
            "baseline_receptor": "11OY",
            "posterior_samples": 32,
        },
        "acquisition": {
            "top_q": 20,
            "monte_carlo_samples": 32,
            "risk_lambda": 0.1,
            "utility_mode": "ranking_score",
            "batch_interaction_weight": 1.0,
        },
        "solver": {"backend": "quantum_compatible_simulator", "time_budget_seconds": 30.0},
        "prediction_gate": {"required": False},
        "stop": {"max_rounds": 1000},
        "random_seed": 20260901,
    }
    payload.update(overrides)
    path = tmp_path / "active.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_production_config_resolves_old_full_config_and_separate_run_directory(tmp_path: Path) -> None:
    config = load_active_production_config(_write_config(tmp_path), data_root=tmp_path)

    assert config.workflow == "active_ligand_receptor_docking"
    assert config.base_config.workflow_mode == "full"
    assert config.baseline_receptor == "11OY"
    assert config.base_config.data["docking"]["seeds"] == [20260821, 20260822, 20260823]
    assert config.prepared_run_directory == (
        tmp_path / "results" / "runs" / "mk14_adaptive_remote"
    ).resolve()
    assert config.active_run_directory == (
        tmp_path / "results" / "runs" / "mk14_adaptive_remote" / "active_docking"
    ).resolve()
    assert len(config.fingerprint) == 64


def test_production_config_accepts_path_override_for_prepared_run_directory(tmp_path: Path) -> None:
    override = tmp_path / "prepared-override"

    config = load_active_production_config(
        _write_config(tmp_path),
        data_root=tmp_path,
        prepared_run_directory=override,
    )

    assert config.prepared_run_directory == override.resolve()


def test_production_config_rejects_seed_or_fusion_drift(tmp_path: Path) -> None:
    payload = json.loads(_write_config(tmp_path).read_text(encoding="utf-8"))
    payload["docking"] = {
        "seeds": [1],
        "score_fusion": "mean",
        "cost_per_seed": 1.0,
    }
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ActiveProductionConfigError, match="median|seed"):
        load_active_production_config(path, data_root=tmp_path)


def test_production_config_rejects_receptor_subset_problem_and_wrong_baseline(tmp_path: Path) -> None:
    payload = json.loads(_write_config(tmp_path).read_text(encoding="utf-8"))
    payload["baseline_receptor"] = "r0"
    path = tmp_path / "invalid-baseline.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ActiveProductionConfigError, match="baseline_receptor"):
        load_active_production_config(path, data_root=tmp_path)
