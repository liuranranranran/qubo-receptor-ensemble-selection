from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

from qubo_receptor_ensemble.active_docking.production import ActiveProductionRunner
from qubo_receptor_ensemble.active_docking.production_config import ActiveProductionConfig


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_pdb(path: Path, offset: float) -> None:
    path.write_text(
        "".join(
            f"ATOM  {index:5d}  CA  ALA A{index:4d}    {offset + index:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 20.00           C\n"
            for index in range(1, 4)
        )
        + "END\n",
        encoding="ascii",
    )


class FakeAdapter:
    name = "fake"

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, tuple[str, ...]]] = []

    def run_batch(self, *, receptor_id: str, ligands: list[dict[str, object]], seed: int, **kwargs: object) -> list[dict[str, object]]:
        del kwargs
        ligand_ids = tuple(str(row["ligand_id"]) for row in ligands)
        self.calls.append((receptor_id, seed, ligand_ids))
        return [
            {
                "ligand_id": ligand_id,
                "receptor_id": receptor_id,
                "docking_score": -10.0 - index - (0.5 if receptor_id == "R2" else 0.0) - seed / 1000.0,
                "status": "ok",
                "seed": seed,
            }
            for index, ligand_id in enumerate(ligand_ids)
        ]


def _config(root: Path) -> ActiveProductionConfig:
    data = {
        "workflow": "active_ligand_receptor_docking",
        "target_id": "MK14",
        "receptor_cluster": {"method": "aligned_ca_rmsd", "threshold_angstrom": 2.0},
        "docking": {"seeds": [1, 2, 3], "score_fusion": "median", "cost_per_seed": 1.0},
        "warm_start": {"baseline_receptor": "11OY", "cluster_fraction": 0.0, "min_ligands_per_cluster": 0, "random_seed": 7},
        "predictor": {"model": "bayesian_residual", "baseline_receptor": "11OY", "posterior_samples": 8},
        "acquisition": {"top_q": 1, "monte_carlo_samples": 8, "risk_lambda": 0.0, "utility_mode": "ranking_score", "batch_interaction_weight": 1.0},
        "candidate_cap": 4,
        "budget": {"total_cost": 9.0, "batch_cost": 3.0},
        "constraints": {"max_per_ligand": 1, "max_per_receptor": 4, "max_per_scaffold": 2, "receptor_activation_cost": 0.0, "penalty": 20.0, "cost_unit": 0.001, "equal_cost": True, "coefficient_scale": 1.0},
        "solver": {"backend": "exact", "time_budget_seconds": 10.0},
        "strategy": "exact",
        "prediction_gate": {"required": False},
        "stop": {"max_rounds": 10},
        "random_seed": 7,
    }
    base = SimpleNamespace(
        data_root=root,
        data={"target_id": "MK14", "docking": {"engine": "unidock", "seeds": [1, 2, 3]}},
        paths={},
        workflow_mode="full",
    )
    return ActiveProductionConfig(
        path=root / "active.json",
        data_root=root,
        data=data,
        base_config=base,
        prepared_run_directory=root / "prepared",
        active_run_directory=root / "prepared" / "active_docking",
        baseline_receptor="11OY",
        fingerprint="b" * 64,
    )


def _prepared(root: Path) -> None:
    prepared = root / "prepared"
    _write_csv(
        prepared / "prepared_ligands.csv",
        [
            {"ligand_id": "L1", "smiles": "CCO", "label": "active", "selection_role": "train", "pdbqt_path": "L1.pdbqt"},
            {"ligand_id": "L2", "smiles": "c1ccccc1", "label": "decoy", "selection_role": "train", "pdbqt_path": "L2.pdbqt"},
        ],
    )
    _write_pdb(prepared / "11OY.pdb", 0.0)
    _write_pdb(prepared / "R2.pdb", 10.0)
    _write_csv(
        prepared / "selected_receptors.csv",
        [
            {"conformer_id": "11OY", "receptor_pdb": "prepared/11OY.pdb", "receptor_pdbqt": "11OY.pdbqt"},
            {"conformer_id": "R2", "receptor_pdb": "prepared/R2.pdb", "receptor_pdbqt": "R2.pdbqt"},
        ],
    )


def test_production_smoke_consumes_prepare_outputs_and_replays_deterministically(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    _prepared(first_root)
    _prepared(second_root)
    first_adapter = FakeAdapter()
    second_adapter = FakeAdapter()

    first = ActiveProductionRunner(_config(first_root), adapter=first_adapter)
    second = ActiveProductionRunner(_config(second_root), adapter=second_adapter)
    first_result = first.run(max_rounds=1)
    second_result = second.run(max_rounds=1)

    assert first_result["prepare_stage"] == "canonical_full_workflow_prepare_output"
    assert first_result["task_sequence"] == second_result["task_sequence"]
    assert len(first_adapter.calls) == 6
    assert all(len(call[2]) <= 2 for call in first_adapter.calls)
    assert (first.config.active_run_directory / "state.json").is_file()
    assert (first.config.active_run_directory / "round_000.json").is_file()
    state = json.loads(
        (first.config.active_run_directory / "state.json").read_text(encoding="utf-8")
    )
    assert all(
        key.lower() not in {"label", "active", "decoy", "hidden_label"}
        for key in state
    )
