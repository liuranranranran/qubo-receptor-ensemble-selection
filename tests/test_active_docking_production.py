from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from qubo_receptor_ensemble.active_docking.production import (
    ActiveProductionRunner,
    ProductionRunError,
)
from qubo_receptor_ensemble.active_docking.production_config import ActiveProductionConfig


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_pdb(path: Path, offset: float = 0.0) -> None:
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
        names = tuple(str(row["ligand_id"]) for row in ligands)
        self.calls.append((receptor_id, seed, names))
        receptor_offset = 0.0 if receptor_id == "11OY" else 0.5
        return [
            {
                "ligand_id": ligand_id,
                "receptor_id": receptor_id,
                "docking_score": -10.0 - index - receptor_offset - seed / 1000.0,
                "status": "ok",
                "seed": seed,
            }
            for index, ligand_id in enumerate(names)
        ]


class ConfigCaptureAdapter(FakeAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.configs: list[dict[str, object]] = []

    def run_batch(self, *, config: dict[str, object], **kwargs: object) -> list[dict[str, object]]:
        self.configs.append(config)
        return super().run_batch(config=config, **kwargs)


def _config(tmp_path: Path, *, total_cost: float = 12.0) -> ActiveProductionConfig:
    data = {
        "workflow": "active_ligand_receptor_docking",
        "target_id": "MK14",
        "receptor_cluster": {
            "method": "aligned_ca_rmsd",
            "threshold_angstrom": 2.0,
        },
        "docking": {
            "seeds": [1, 2, 3],
            "score_fusion": "median",
            "cost_per_seed": 1.0,
        },
        "warm_start": {
            "baseline_receptor": "11OY",
            "cluster_fraction": 0.0,
            "min_ligands_per_cluster": 0,
            "random_seed": 7,
        },
        "predictor": {
            "model": "bayesian_residual",
            "baseline_receptor": "11OY",
            "posterior_samples": 8,
        },
        "acquisition": {
            "top_q": 1,
            "monte_carlo_samples": 8,
            "risk_lambda": 0.0,
            "utility_mode": "ranking_score",
            "batch_interaction_weight": 1.0,
        },
        "candidate_cap": 4,
        "budget": {"total_cost": total_cost, "batch_cost": 3.0},
        "constraints": {
            "max_per_ligand": 1,
            "max_per_receptor": 4,
            "max_per_scaffold": 2,
            "receptor_activation_cost": 0.0,
            "penalty": 20.0,
            "cost_unit": 0.001,
            "equal_cost": True,
            "coefficient_scale": 1.0,
        },
        "solver": {"backend": "exact", "time_budget_seconds": 10.0},
        "strategy": "exact",
        "random_seed": 7,
        "prediction_gate": {"required": False},
        "stop": {"max_rounds": 10},
    }
    base = SimpleNamespace(
        data_root=tmp_path,
        data={
            "target_id": "MK14",
            "docking": {
                "engine": "unidock",
                "executable": "unidock",
                "seeds": [1, 2, 3],
                "parameters": {"exhaustiveness": 17},
            },
        },
        paths={},
        workflow_mode="full",
    )
    return ActiveProductionConfig(
        path=tmp_path / "active.json",
        data_root=tmp_path,
        data=data,
        base_config=base,
        prepared_run_directory=tmp_path / "prepared",
        active_run_directory=tmp_path / "prepared" / "active_docking",
        baseline_receptor="11OY",
        fingerprint="f" * 64,
    )


def _prepared_inputs(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    _write_csv(
        prepared / "prepared_ligands.csv",
        [
            {"ligand_id": "L1", "smiles": "CCO", "label": "active", "selection_role": "train", "pdbqt_path": "L1.pdbqt"},
            {"ligand_id": "L2", "smiles": "c1ccccc1", "label": "decoy", "selection_role": "train", "pdbqt_path": "L2.pdbqt"},
        ],
    )
    _write_pdb(prepared / "11OY.pdb")
    _write_pdb(prepared / "R2.pdb", 10.0)
    _write_csv(
        prepared / "selected_receptors.csv",
        [
            {"conformer_id": "11OY", "receptor_pdb": "prepared/11OY.pdb", "receptor_pdbqt": "11OY.pdbqt"},
            {"conformer_id": "R2", "receptor_pdb": "prepared/R2.pdb", "receptor_pdbqt": "R2.pdbqt"},
        ],
    )
    (prepared / "docking_box.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "method": "ligand_bounds",
                "box": {
                    "center_x": 1.0,
                    "center_y": 2.0,
                    "center_z": 3.0,
                    "size_x": 22.0,
                    "size_y": 23.0,
                    "size_z": 28.0,
                },
            }
        ),
        encoding="utf-8",
    )


def test_prepare_delegates_to_canonical_prepare(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    calls: list[tuple[object, bool, bool]] = []

    def fake_prepare(base_config: object, *, resume: bool, overwrite: bool) -> dict[str, object]:
        calls.append((base_config, resume, overwrite))
        return {"ligands": [], "receptors": []}

    monkeypatch.setattr(
        "qubo_receptor_ensemble.active_docking.production.prepare_experiment_inputs",
        fake_prepare,
    )
    result = ActiveProductionRunner(config, adapter=FakeAdapter()).prepare(resume=True, overwrite=True)

    assert calls == [(config.base_config, True, True)]
    assert result["prepared_run_directory"] == str(config.prepared_run_directory)


def test_production_runner_reveals_only_selected_tasks_and_resumes(tmp_path: Path) -> None:
    _prepared_inputs(tmp_path)
    adapter = FakeAdapter()
    config = _config(tmp_path)
    runner = ActiveProductionRunner(config, adapter=adapter)

    runner.initialize()
    assert len(adapter.calls) == 3
    assert runner.state.completed_tasks() == (("L1", "11OY"), ("L2", "11OY"))
    assert runner.state.docking_cost == pytest.approx(6.0)

    result = runner.run(max_rounds=2)
    assert len(result["task_sequence"]) == 2
    assert runner.state.docking_cost == pytest.approx(12.0)
    assert all(task[1] == "R2" for task in result["task_sequence"])
    state_text = (config.active_run_directory / "state.json").read_text(encoding="utf-8")
    state_payload = json.loads(state_text)
    assert "hidden_score" not in state_text
    assert all(
        key.lower() not in {"label", "active", "decoy", "hidden_label"}
        for key in state_payload
    )
    assert all(
        value not in {"active", "decoy"}
        for value in state_payload.get("scaffold_metadata", {}).values()
    )

    call_count = len(adapter.calls)
    resumed = ActiveProductionRunner(config, adapter=adapter).run(resume=True, max_rounds=2)
    assert resumed["task_sequence"] == result["task_sequence"]
    assert len(adapter.calls) == call_count


def test_production_runner_passes_prepared_box_to_warm_and_active_docking(
    tmp_path: Path,
) -> None:
    _prepared_inputs(tmp_path)
    adapter = ConfigCaptureAdapter()
    runner = ActiveProductionRunner(_config(tmp_path), adapter=adapter)

    runner.initialize()
    runner.run(max_rounds=1)

    assert adapter.configs
    for config in adapter.configs:
        docking = config["docking"]
        assert isinstance(docking, dict)
        assert docking["executable"] == "unidock"
        assert docking["parameters"] == {"exhaustiveness": 17}
        assert docking["box"] == {
            "center_x": 1.0,
            "center_y": 2.0,
            "center_z": 3.0,
            "size_x": 22.0,
            "size_y": 23.0,
            "size_z": 28.0,
        }


def test_production_runner_reports_round_stages_and_pair_progress(
    tmp_path: Path,
) -> None:
    _prepared_inputs(tmp_path)
    messages: list[str] = []
    runner = ActiveProductionRunner(
        _config(tmp_path),
        adapter=FakeAdapter(),
        progress=messages.append,
    )

    runner.run(max_rounds=1)

    expected_stages = (
        "predictor",
        "candidate_pool",
        "acquisition",
        "batch_interaction",
        "qubo",
        "solver",
        "docking",
        "checkpoint",
    )
    for stage in expected_stages:
        assert f"[active][round=000][stage={stage}] start" in messages
        assert any(
            message.startswith(f"[active][round=000][stage={stage}] done elapsed=")
            for message in messages
        )
    assert any(
        message.startswith("[active][round=000][stage=batch_interaction] progress pairs=1/1")
        for message in messages
    )


def test_production_runner_persists_completed_run_summary(
    tmp_path: Path,
) -> None:
    _prepared_inputs(tmp_path)
    runner = ActiveProductionRunner(_config(tmp_path), adapter=FakeAdapter())

    result = runner.run(max_rounds=1)

    persisted = json.loads(
        (runner.config.active_run_directory / "run_summary.json").read_text(encoding="utf-8")
    )
    assert result["status"] == "completed"
    assert result["stop_reason"] == "round_limit"
    assert persisted["status"] == "completed"
    assert persisted["stop_reason"] == "round_limit"


def test_production_runner_rejects_missing_prepared_box_before_docking(
    tmp_path: Path,
) -> None:
    _prepared_inputs(tmp_path)
    (tmp_path / "prepared" / "docking_box.json").unlink()
    adapter = FakeAdapter()

    with pytest.raises(ProductionRunError, match="docking_box"):
        ActiveProductionRunner(_config(tmp_path), adapter=adapter).initialize()

    assert adapter.calls == []


def test_production_runner_rejects_invalid_prepared_box_before_docking(
    tmp_path: Path,
) -> None:
    _prepared_inputs(tmp_path)
    (tmp_path / "prepared" / "docking_box.json").write_text(
        json.dumps({"box": {"center_x": 1.0}}),
        encoding="utf-8",
    )
    adapter = FakeAdapter()

    with pytest.raises(ProductionRunError, match="docking_box"):
        ActiveProductionRunner(_config(tmp_path), adapter=adapter).initialize()

    assert adapter.calls == []


def test_production_runner_rejects_resume_when_config_fingerprint_changes(tmp_path: Path) -> None:
    _prepared_inputs(tmp_path)
    config = _config(tmp_path, total_cost=9.0)
    ActiveProductionRunner(config, adapter=FakeAdapter()).initialize()
    changed = replace(config, fingerprint="e" * 64)

    with pytest.raises(ProductionRunError, match="fingerprint"):
        ActiveProductionRunner(changed, adapter=FakeAdapter()).run(resume=True)


def test_required_prediction_gate_blocks_docking_until_passed(tmp_path: Path) -> None:
    _prepared_inputs(tmp_path)
    config = _config(tmp_path)
    config.data["prediction_gate"] = {"required": True}
    adapter = FakeAdapter()

    with pytest.raises(ProductionRunError, match="prediction gate"):
        ActiveProductionRunner(config, adapter=adapter).initialize()
    assert adapter.calls == []

    gate = config.active_run_directory / "prediction_gate.json"
    gate.parent.mkdir(parents=True, exist_ok=True)
    gate.write_text(json.dumps({"passed": True}), encoding="utf-8")
    ActiveProductionRunner(config, adapter=adapter).initialize()
    assert len(adapter.calls) == 3
