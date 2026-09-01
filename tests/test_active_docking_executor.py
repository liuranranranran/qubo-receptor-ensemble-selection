from __future__ import annotations

from pathlib import Path

import pytest

from qubo_receptor_ensemble.active_docking.executor import (
    SelectedTaskExecutor,
    TaskExecutionError,
)


class FakeAdapter:
    name = "fake"

    def __init__(self, missing: tuple[str, int] | None = None) -> None:
        self.calls: list[tuple[str, int, tuple[str, ...]]] = []
        self.missing = missing

    def run_batch(self, *, receptor_id: str, ligands: list[dict[str, str]], seed: int, **kwargs: object) -> list[dict[str, object]]:
        del kwargs
        self.calls.append((receptor_id, seed, tuple(row["ligand_id"] for row in ligands)))
        rows: list[dict[str, object]] = []
        for ligand in ligands:
            if self.missing == (ligand["ligand_id"], seed):
                continue
            rows.append({
                "ligand_id": ligand["ligand_id"],
                "receptor_id": receptor_id,
                "docking_score": -10.0 - float(seed) / 1000.0 - (0.1 if ligand["ligand_id"] == "L2" else 0.0),
                "status": "ok",
                "seed": seed,
            })
        return rows


def _executor(tmp_path: Path, adapter: FakeAdapter) -> SelectedTaskExecutor:
    return SelectedTaskExecutor(
        adapter=adapter,
        data_root=tmp_path,
        target_id="MK14",
        seeds=(1, 2, 3),
        score_fusion="median",
        cost_per_seed=1.0,
    )


def test_executor_submits_only_selected_tasks_for_each_seed_and_fuses_median(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    executor = _executor(tmp_path, adapter)
    results = executor.execute(
        tasks=(("L1", "R1"), ("L2", "R1")),
        ligand_manifest=[
            {"ligand_id": "L1", "pdbqt_path": "L1.pdbqt"},
            {"ligand_id": "L2", "pdbqt_path": "L2.pdbqt"},
            {"ligand_id": "UNSELECTED", "pdbqt_path": "U.pdbqt"},
        ],
        receptor_manifest=[{"receptor_id": "R1", "receptor_pdbqt": "R1.pdbqt"}],
        output_directory=tmp_path / "round_000",
    )

    assert len(adapter.calls) == 3
    assert all(call[0] == "R1" and call[2] == ("L1", "L2") for call in adapter.calls)
    assert results[("L1", "R1")].fused_score == pytest.approx(-10.002)
    assert results[("L2", "R1")].fused_score == pytest.approx(-10.102)
    assert results[("L1", "R1")].cost == pytest.approx(3.0)
    assert results[("L1", "R1")].seed_scores == {1: -10.001, 2: -10.002, 3: -10.003}


def test_executor_rejects_incomplete_seed_results(tmp_path: Path) -> None:
    adapter = FakeAdapter(missing=("L1", 2))
    executor = _executor(tmp_path, adapter)

    with pytest.raises(TaskExecutionError, match="missing"):
        executor.execute(
            tasks=(("L1", "R1"),),
            ligand_manifest=[{"ligand_id": "L1", "pdbqt_path": "L1.pdbqt"}],
            receptor_manifest=[{"receptor_id": "R1", "receptor_pdbqt": "R1.pdbqt"}],
            output_directory=tmp_path / "round_000",
        )

