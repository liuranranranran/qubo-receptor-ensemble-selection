"""T6: reproduction is bit-identical for fixed seeds and inputs, plus runner smoke."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from qubo_receptor_ensemble.headroom import runner


def write_dataset(root: Path, ligands: int = 120, receptors: int = 6, folds: int = 3) -> tuple[Path, Path]:
    rng = np.random.default_rng(17)
    matrix_path = root / "primary_matrix.csv"
    manifest_path = root / "prepared_ligands.csv"
    with matrix_path.open("w", encoding="utf-8", newline="") as matrix_handle, manifest_path.open(
        "w", encoding="utf-8", newline=""
    ) as manifest_handle:
        matrix_writer = csv.writer(matrix_handle)
        manifest_writer = csv.writer(manifest_handle)
        receptor_ids = [f"R{index:02d}" for index in range(receptors)]
        matrix_writer.writerow(["target_id", "ligand_id", "label", "selection_role", *receptor_ids])
        manifest_writer.writerow(["ligand_id", "scaffold_smiles", "outer_fold"])
        scaffold_count = 6
        labels = ["decoy"] * ligands
        for scaffold_id in range(scaffold_count):
            members = [index for index in range(ligands) if index % scaffold_count == scaffold_id]
            for position, member in enumerate(members):
                if position % 3 == 0:
                    labels[member] = "active"
        for index in range(ligands):
            fold = index % folds
            label = labels[index]
            scores = rng.normal(-8.0, 1.0, size=receptors)
            if label == "active":
                scores[0] -= 1.2
            matrix_writer.writerow(
                ["SYN", f"L{index:04d}", label, "development_train", *[f"{value:.4f}" for value in scores]]
            )
            manifest_writer.writerow([f"L{index:04d}", f"scaffold_{index % scaffold_count:02d}", fold])
    return matrix_path, manifest_path


def write_assets_config(root: Path, matrix: Path, manifest: Path) -> Path:
    path = root / "e1_assets.json"
    path.write_text(
        json.dumps(
            {
                "schema": "e1_assets_v1",
                "roots": {"root": str(root)},
                "targets": [
                    {
                        "target_id": "SYN",
                        "role": "primary",
                        "matrix": "{root}/primary_matrix.csv",
                        "manifest": "{root}/prepared_ligands.csv",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def write_prereg(root: Path) -> Path:
    path = root / "prereg.json"
    path.write_text(
        json.dumps(
            {
                "schema": "e1_headroom_v1",
                "frozen_at": "2026-09-11T00:00:00Z",
                "targets": ["SYN"],
                "k_range": [1, 2],
                "fusion_family": ["mean", "min", "max", "zmean", "gmean", "hmean", "ranksum", "rrf"],
                "fusion_params": {"rrf_k": 60, "gmean_shift_rule": "1.0 - min(train_scores)"},
                "primary_cells": {"k": [2], "phi": "all"},
                "primary_metric": "pr_auc",
                "bedroc_alpha": 20.0,
                "bootstrap": {"unit": "scaffold_cluster", "iterations": 25, "seed": 0},
                "gate_g1": {"no_go_below": 0.10, "go_above": 0.30, "require_lower_bound": True},
            }
        ),
        encoding="utf-8",
    )
    return path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_small(root: Path, prereg: Path, assets: Path):
    return runner.run_e1(
        prereg_path=prereg,
        assets_path=assets,
        output_dir=root / "results" / "headroom" / "e1_synth",
        jobs=1,
        resume=False,
        k_list=(1, 2),
        bootstrap_iterations=25,
        top_m=25,
        permutations=10,
        perm_ks=(2,),
        inner_fold_count=2,
        skip_figures=True,
        verbose=False,
    )


def test_run_e1_products_are_reproducible(headroom_workspace: Path) -> None:
    matrix, manifest = write_dataset(headroom_workspace)
    assets = write_assets_config(headroom_workspace, matrix, manifest)
    prereg = write_prereg(headroom_workspace)
    first = run_small(headroom_workspace, prereg, assets)
    run_dir = Path(first["paths"]["root"])
    products = [
        run_dir / "cell_metrics.csv",
        run_dir / "headroom_map.csv",
        run_dir / "bootstrap_report.json",
        run_dir / "gate_g1.json",
        run_dir / "phi_selection.json",
        run_dir / "permutations.json",
    ]
    before = {path.name: sha256(path) for path in products}
    assert all(path.is_file() for path in products)

    second = run_small(headroom_workspace, prereg, assets)
    after = {path.name: sha256(path) for path in products}
    assert before == after
    assert first["gate"]["decision"] == second["gate"]["decision"]

    shard_files = sorted((run_dir / "cells").glob("*.json"))
    assert len(shard_files) == 3 * 8  # folds x fusion rules
    payload = json.loads(shard_files[0].read_text(encoding="utf-8"))
    assert payload["schema"] == "e1_shard_v1"


def test_checkpoint_compatibility_allows_superset_k_list(headroom_workspace: Path) -> None:
    base = runner.HeadroomConfig(k_list=(1, 2, 3), bootstrap_iterations=10)
    payload = {"run_id": "r", "config": base.as_dict(), "config_hash": runner._config_hash(base)}
    narrower = runner.HeadroomConfig(k_list=(1, 2), bootstrap_iterations=10)
    wider = runner.HeadroomConfig(k_list=(1, 2, 3, 4), bootstrap_iterations=10)
    other = runner.HeadroomConfig(k_list=(1, 2, 3), bootstrap_iterations=99)
    assert runner.checkpoint_compatible(payload, narrower, "r")
    assert not runner.checkpoint_compatible(payload, wider, "r")
    assert not runner.checkpoint_compatible(payload, other, "r")
    assert not runner.checkpoint_compatible(payload, narrower, "other-run")


def test_preregistration_schema_is_enforced(headroom_workspace: Path) -> None:
    bad = headroom_workspace / "bad.json"
    bad.write_text(json.dumps({"schema": "nope"}), encoding="utf-8")
    with pytest.raises(runner.RunnerError):
        runner.load_preregistration(bad)


def test_cli_parses_the_documented_invocation() -> None:
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts" / "headroom_scan.py"
    spec = importlib.util.spec_from_file_location("headroom_scan", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.build_parser().parse_args(
        [
            "run",
            "--prereg",
            "p.json",
            "--assets",
            "a.json",
            "--output-dir",
            "out",
            "--jobs",
            "24",
            "--resume",
            "--targets",
            "MK14,PPARA",
            "--quick",
        ]
    )
    assert args.command == "run" and args.jobs == 24 and args.resume
    assert module._parse_targets(args.targets) == ("MK14", "PPARA")
    assert module._parse_ints("2,3") == (2, 3)

def test_parallel_map_is_lazy_so_checkpoints_stay_incremental() -> None:
    """Regression: an eager map would defer every checkpoint until the end."""

    produced: list[int] = []

    def worker(item):
        produced.append(int(item["index"]))
        return item["index"]

    items = [{"index": index} for index in range(4)]
    generator = runner.parallel_map(1, worker, items, verbose=False)
    first = next(generator)
    assert first == 0
    assert produced == [0]  # only the first shard ran: it can be checkpointed now
    assert list(generator) == [1, 2, 3]
    assert produced == [0, 1, 2, 3]
