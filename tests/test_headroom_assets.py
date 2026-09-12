"""D1 asset loading and verification."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from qubo_receptor_ensemble.headroom.assets import (
    AssetError,
    AssetSpec,
    load_asset_specs,
    load_matrix_panel,
    load_problem_panel,
    load_target_panel,
    sha256_records,
    verify_panel,
)


def write_matrix(path: Path, ligands: int = 12, receptors: int = 3, target: str = "TST") -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["target_id", "ligand_id", "label", "selection_role", "R1", "R2", "R3"][: 4 + receptors])
        for index in range(ligands):
            label = "active" if index % 3 == 0 else "decoy"
            writer.writerow([target, f"L{index:04d}", label, "development_train"] + [f"{-8.0 - index * 0.01 - column:.4f}" for column in range(receptors)])


def write_manifest(path: Path, ligands: int = 12, missing_at: int | None = None) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ligand_id", "scaffold_smiles", "outer_fold"])
        for index in range(ligands):
            scaffold = "" if index == missing_at else f"c1ccccc1{index % 3}"
            writer.writerow([f"L{index:04d}", scaffold, index % 3])


def test_matrix_and_manifest_loader_produces_a_verified_panel(headroom_workspace: Path) -> None:
    matrix = headroom_workspace / "matrix.csv"
    manifest = headroom_workspace / "prepared.csv"
    write_matrix(matrix)
    write_manifest(manifest)
    spec = AssetSpec(target_id="TST", matrix=matrix, manifest=manifest)
    panel = load_matrix_panel(spec, matrix, manifest)
    report = verify_panel(panel)
    assert report["status"] == "ok"
    assert panel.n_ligands == 12 and panel.n_receptors == 3
    assert panel.receptor_ids == ("R1", "R2", "R3")
    assert panel.ligand_ids == tuple(sorted(panel.ligand_ids))
    assert set(panel.folds.tolist()) == {0, 1, 2}


def test_missing_scaffold_is_backfilled_and_counted(headroom_workspace: Path) -> None:
    matrix = headroom_workspace / "matrix.csv"
    manifest = headroom_workspace / "prepared.csv"
    write_matrix(matrix)
    write_manifest(manifest, missing_at=2)
    panel = load_matrix_panel(AssetSpec(target_id="TST", matrix=matrix, manifest=manifest), matrix, manifest)
    report = verify_panel(panel, strict=False)
    assert report["status"] == "ok"
    assert report["backfilled_scaffold_rows"] == 1
    assert all(str(value).strip() for value in panel.scaffolds)


def test_missing_ligand_in_manifest_is_rejected(headroom_workspace: Path) -> None:
    matrix = headroom_workspace / "matrix.csv"
    manifest = headroom_workspace / "prepared.csv"
    write_matrix(matrix, ligands=8)
    write_manifest(manifest, ligands=6)
    with pytest.raises(AssetError):
        load_matrix_panel(AssetSpec(target_id="TST", matrix=matrix, manifest=manifest), matrix, manifest)


def test_bad_label_and_missing_fold_are_rejected(headroom_workspace: Path) -> None:
    matrix = headroom_workspace / "matrix.csv"
    manifest = headroom_workspace / "prepared.csv"
    write_matrix(matrix)
    write_manifest(manifest)
    text = matrix.read_text(encoding="utf-8").replace("active", "inactive", 1)
    matrix.write_text(text, encoding="utf-8")
    with pytest.raises(AssetError):
        load_matrix_panel(AssetSpec(target_id="TST", matrix=matrix, manifest=manifest), matrix, manifest)

    write_matrix(matrix)
    rows = manifest.read_text(encoding="utf-8").splitlines()
    rows[1] = rows[1].rsplit(",", 1)[0]  # drop outer_fold for one ligand
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(AssetError):
        load_matrix_panel(AssetSpec(target_id="TST", matrix=matrix, manifest=manifest), matrix, manifest)


def write_problem(path: Path, receptors=("P1", "P2"), backfill: Path | None = None) -> None:
    rows = []
    for index in range(9):
        row = {
            "P1": -8.0 - index * 0.01,
            "P2": -7.5 - index * 0.02,
            "ligand_id": f"P{index:04d}",
            "label": "active" if index % 3 == 0 else "decoy",
            "target_id": "PRB",
            "scaffold_smiles": f"CC{index % 2}",
            "outer_fold": index % 3,
        }
        rows.append(row)
    path.write_text(
        json.dumps({"rows": rows, "problem_config": {"receptor_ids": list(receptors)}}),
        encoding="utf-8",
    )


def test_problem_json_carrier_is_supported(headroom_workspace: Path) -> None:
    problem = headroom_workspace / "problem.json"
    write_problem(problem)
    spec = AssetSpec(target_id="PRB", problem_json=problem)
    panel = load_problem_panel(spec, problem)
    report = verify_panel(panel)
    assert report["status"] == "ok"
    assert panel.source == "problem_json"
    assert panel.receptor_ids == ("P1", "P2")


def test_problem_json_carrier_can_backfill_scaffold_and_fold(headroom_workspace: Path) -> None:
    donor = headroom_workspace / "donor.json"
    write_problem(donor)
    payload = json.loads(donor.read_text(encoding="utf-8"))
    for row in payload["rows"]:
        row.pop("scaffold_smiles")
        row.pop("outer_fold")
    target = headroom_workspace / "target.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    spec = AssetSpec(target_id="PRB", problem_json=target, backfill_problem_json=donor)
    panel = load_problem_panel(spec, target, backfill_path=donor)
    report = verify_panel(panel, strict=False)
    assert report["status"] == "ok"
    assert report["backfilled_scaffold_rows"] == 9
    assert report["backfilled_fold_rows"] == 9
    assert set(panel.folds.tolist()) == {0, 1, 2}


def test_loader_prefers_matrix_over_problem_json(headroom_workspace: Path) -> None:
    matrix = headroom_workspace / "matrix.csv"
    manifest = headroom_workspace / "prepared.csv"
    problem = headroom_workspace / "problem.json"
    write_matrix(matrix)
    write_manifest(manifest)
    write_problem(problem)
    spec = AssetSpec(target_id="TST", matrix=matrix, manifest=manifest, problem_json=problem)
    panel = load_target_panel(spec)
    assert panel.source == "matrix_csv"


def test_verify_panel_flags_nan_scores(headroom_workspace: Path) -> None:
    matrix = headroom_workspace / "matrix.csv"
    manifest = headroom_workspace / "prepared.csv"
    write_matrix(matrix)
    write_manifest(manifest)
    panel = load_matrix_panel(AssetSpec(target_id="TST", matrix=matrix, manifest=manifest), matrix, manifest)
    broken = type(panel)(**{**panel.__dict__, "scores": np.where(np.arange(12)[:, None] == 3, np.nan, panel.scores)})
    report = verify_panel(broken, strict=False)
    assert report["status"] == "failed"
    assert any("non-finite" in problem or "NaN" in problem for problem in report["problems"])


def test_asset_spec_placeholders_are_expanded(headroom_workspace: Path) -> None:
    config = headroom_workspace / "e1_assets.json"
    config.write_text(
        json.dumps(
            {
                "schema": "e1_assets_v1",
                "roots": {"root": str(headroom_workspace)},
                "targets": [
                    {"target_id": "TST", "matrix": "{root}/matrix.csv", "manifest": "{root}/prepared.csv"}
                ],
            }
        ),
        encoding="utf-8",
    )
    roots, specs = load_asset_specs(config)
    assert roots["root"] == str(headroom_workspace)
    assert specs[0].matrix == headroom_workspace / "matrix.csv"


def test_sha256_records_skips_missing_paths(headroom_workspace: Path) -> None:
    present = headroom_workspace / "present.txt"
    present.write_text("data", encoding="utf-8")
    records = sha256_records([present, headroom_workspace / "missing.txt"])
    assert len(records) == 1
    record = next(iter(records.values()))
    assert len(str(record["sha256"])) == 64