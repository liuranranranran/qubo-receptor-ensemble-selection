"""E1 runner: sharded exact scan, checkpoints, products, gate and figures.

Entry point is ``scripts/headroom_scan.py``.  The runner keeps the
pre-registered knobs frozen (hash-checked), shards work by
``(target, fold, phi)``, writes resumable JSON checkpoints per shard and then
builds the DoD products:

``input_manifest.json``  D1 verification + SHA-256 of every input file
``cell_metrics.csv``     per (target, fold, phi, k, method) metrics + headroom
``headroom_map.csv``     target x phi x k map of the nested headroom ratio
``bootstrap_report.json``noise floor / MDE per cell
``phi_selection.json``   train-only inner-CV phi selection + k* displacement
``gate_g1.json``         frozen G1 decision with its evidence chain
``run_manifest.json``    commit, environment, seeds, input hashes
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np

from ..io import file_sha256, write_csv, write_json
from . import assets as asset_module
from . import __name__ as _package_name  # noqa: F401 - keeps the module import explicit
from .assets import AssetSpec, LigandPanel, load_asset_specs, load_target_panel, sha256_records, verify_panel
from .bootstrap import percentile
from .fusion import FusionScorer, FUSION_NAMES, FusionSpec, fit_fusion
from .gate import evaluate_gate
from .headroom import HeadroomConfig, scan_shard, scaffold_allocation
from .metrics_fast import METRIC_NAMES, metric_values
from .subsets import iter_combinations, utility

PREREG_SCHEMA = "e1_headroom_v1"
RUN_SCHEMA = "e1_run_v1"

REQUESTED_METRICS: tuple[str, ...] = ("pr_auc", "bedroc20", "roc_auc", "ef1", "ef5", "ef10")


class RunnerError(RuntimeError):
    """Raised when the E1 run cannot proceed as pre-registered."""


@dataclass(frozen=True)
class RunPaths:
    """Canonical product paths under ``results/headroom/<run_id>``."""

    root: Path
    cells: Path
    figures: Path

    @property
    def input_manifest(self) -> Path:
        return self.root / "input_manifest.json"

    @property
    def cell_metrics(self) -> Path:
        return self.root / "cell_metrics.csv"

    @property
    def headroom_map(self) -> Path:
        return self.root / "headroom_map.csv"

    @property
    def bootstrap_report(self) -> Path:
        return self.root / "bootstrap_report.json"

    @property
    def phi_selection(self) -> Path:
        return self.root / "phi_selection.json"

    @property
    def gate_g1(self) -> Path:
        return self.root / "gate_g1.json"

    @property
    def run_manifest(self) -> Path:
        return self.root / "run_manifest.json"

    @property
    def figure_headroom(self) -> Path:
        return self.figures / "fig_headroom_map.png"

    @property
    def figure_phi(self) -> Path:
        return self.figures / "fig_phi_ranking.png"


def build_run_paths(output_dir: Path) -> RunPaths:
    root = Path(output_dir)
    return RunPaths(root=root, cells=root / "cells", figures=root / "figures")


def load_preregistration(path: Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RunnerError(f"pre-registration root must be an object: {path}")
    if payload.get("schema") != PREREG_SCHEMA:
        raise RunnerError(f"pre-registration schema must be {PREREG_SCHEMA}")
    for key in ("targets", "k_range", "fusion_family", "primary_cells", "gate_g1"):
        if key not in payload:
            raise RunnerError(f"pre-registration is missing {key}")
    k_range = payload["k_range"]
    if (
        not isinstance(k_range, Sequence)
        or len(k_range) != 2
        or int(k_range[0]) < 1
        or int(k_range[1]) < int(k_range[0])
    ):
        raise RunnerError("k_range must be [k_min, k_max]")
    payload["_sha256"] = file_sha256(Path(path))
    payload["_path"] = Path(path).as_posix()
    return payload


def headroom_config_from_prereg(
    prereg: Mapping[str, object],
    *,
    k_list: Sequence[int] | None = None,
    bootstrap_iterations: int | None = None,
    top_m: int | None = None,
    permutations: int | None = None,
    perm_ks: Sequence[int] | None = None,
    with_train_oracle: bool = True,
    max_subsets_per_k: int | None = None,
) -> HeadroomConfig:
    k_min, k_max = (int(value) for value in prereg["k_range"])
    bootstrap = prereg.get("bootstrap")
    if not isinstance(bootstrap, Mapping):
        raise RunnerError("pre-registration is missing bootstrap")
    primary = prereg["primary_cells"]
    if not isinstance(primary, Mapping):
        raise RunnerError("pre-registration is missing primary_cells")
    return HeadroomConfig(
        k_list=tuple(int(value) for value in (k_list or range(k_min, k_max + 1))),
        metric=str(prereg.get("primary_metric", "pr_auc")),
        alpha=float(prereg.get("bedroc_alpha", 20.0)),
        top_m=int(top_m if top_m is not None else 2000),
        permutations=int(permutations if permutations is not None else 200),
        perm_ks=tuple(int(value) for value in (perm_ks or primary.get("k", (2, 3)))),
        bootstrap_iterations=int(
            bootstrap_iterations if bootstrap_iterations is not None else bootstrap.get("iterations", 2000)
        ),
        bootstrap_seed=int(bootstrap.get("seed", 0)),
        with_train_oracle=bool(with_train_oracle),
        max_subsets_per_k=max_subsets_per_k,
    )


def _config_hash(config: HeadroomConfig) -> str:
    payload = json.dumps(config.as_dict(), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def shard_filename(target_id: str, fold: int, phi: str) -> str:
    safe = str(target_id).replace("/", "_").replace("\\", "_")
    return f"{safe}_{int(fold):02d}_{phi}.json"


def parallel_map(
    jobs: int,
    function,
    items: Sequence[Mapping[str, object]],
    verbose: bool = True,
) -> Iterator[object]:
    """Lazily yield worker results, with a serial fallback.

    Results are yielded as soon as each shard finishes (``as_completed``) or,
    in the serial fallback, one by one.  This matters for ``--resume``: the
    caller writes a checkpoint per yielded payload, so an interrupted run only
    loses the shard that was still in flight instead of the whole battery.

    Some sandboxes forbid the anonymous pipes used by
    ``ProcessPoolExecutor`` on Windows (``PermissionError: [WinError 5]``);
    the fallback keeps the protocol identical and only loses wall-clock time.
    """
    payloads = [dict(item) for item in items]
    if jobs and jobs > 1 and len(payloads) > 1:
        try:
            with ProcessPoolExecutor(max_workers=int(jobs)) as pool:
                futures = [pool.submit(function, payload) for payload in payloads]
                for future in as_completed(futures):
                    yield future.result()
            return
        except (OSError, PermissionError, RuntimeError, ImportError) as exc:  # pragma: no cover
            if verbose:
                print(f"[parallel] process pool unavailable ({exc}); running serially", flush=True)
    for payload in payloads:
        yield function(payload)


def run_shard_task(task: Mapping[str, object]) -> dict[str, object]:
    """Worker: full battery for one (target, fold, phi) shard."""
    panel: LigandPanel = task["panel"]  # type: ignore[assignment]
    fold = int(task["fold"])
    spec = FusionSpec(
        name=str(task["phi"]),
        rrf_k=int(task.get("rrf_k", 60)),
        shift_rule=str(task.get("shift_rule", "1.0 - min(train_scores)")),
    )
    config = HeadroomConfig(**dict(task["config"]))  # type: ignore[arg-type]
    train_index = np.flatnonzero(panel.folds != fold)
    test_index = np.flatnonzero(panel.folds == fold)
    started = time.time()
    payload = scan_shard(
        target_id=str(task["target_id"]),
        fold=fold,
        spec=spec,
        train_scores=panel.scores[train_index],
        train_labels=panel.labels[train_index],
        train_scaffolds=[panel.scaffolds[index] for index in train_index],
        test_scores=panel.scores[test_index],
        test_labels=panel.labels[test_index],
        test_scaffolds=[panel.scaffolds[index] for index in test_index],
        config=config,
        seed=int(task.get("seed", 0)),
    )
    payload["run_id"] = str(task.get("run_id", ""))
    payload["role"] = str(task.get("role", "primary"))
    payload["config_hash"] = _config_hash(config)
    payload["elapsed_seconds"] = round(time.time() - started, 3)
    payload["test_ligand_ids"] = [panel.ligand_ids[index] for index in test_index]
    return payload


def run_permutation_task(task: Mapping[str, object]) -> dict[str, object]:
    """Worker: label-permutation inflation correction for one primary cell."""
    from .headroom import permutation_null

    spec = FusionSpec(
        name=str(task["phi"]),
        rrf_k=int(task.get("rrf_k", 60)),
        shift_rule=str(task.get("shift_rule", "1.0 - min(train_scores)")),
    )
    config = HeadroomConfig(**dict(task["config"]))  # type: ignore[arg-type]
    frozen = fit_fusion(spec, np.asarray(task["train_scores"], dtype=np.float64))
    scorer = FusionScorer(frozen, np.asarray(task["test_scores"], dtype=np.float64))
    null = permutation_null(
        scorer,
        [int(mask) for mask in task["top_masks"]],  # type: ignore[index]
        np.asarray(task["test_labels"], dtype=np.float64),
        config,
        seed=int(task.get("seed", 0)),
    )
    h_raw = float(task["h_raw"])
    u_test_oracle = float(task["u_test_oracle"])
    h_perm = h_raw - float(null["q95"]) if math.isfinite(float(null["q95"])) else None
    return {
        "target_id": str(task["target_id"]),
        "fold": int(task["fold"]),
        "phi": str(task["phi"]),
        "k": int(task["k"]),
        "h_raw": h_raw,
        "null": null,
        "h_perm": h_perm,
        "h_perm_alt": (
            u_test_oracle - float(null["q95"]) if math.isfinite(float(null["q95"])) else None
        ),
    }


def run_phi_selection_task(task: Mapping[str, object]) -> dict[str, object]:
    """Worker: inner-CV, train-only phi selection for one (target, fold)."""
    panel: LigandPanel = task["panel"]  # type: ignore[assignment]
    fold = int(task["fold"])
    config = HeadroomConfig(**dict(task["config"]))  # type: ignore[arg-type]
    inner_fold_count = int(task.get("inner_fold_count", 3))
    phis = tuple(str(value) for value in task["phis"])  # type: ignore[index]
    train_index = np.flatnonzero(panel.folds != fold)
    train_labels = panel.labels[train_index]
    train_scores = panel.scores[train_index]
    assignment = scaffold_allocation(
        [panel.scaffolds[index] for index in train_index], inner_fold_count
    )
    inner_scores: dict[str, dict[int, list[float]]] = {
        phi: {k: [] for k in config.k_list} for phi in phis
    }
    inner_sizes: list[dict[str, int]] = []
    for inner in range(inner_fold_count):
        held_mask = assignment == inner
        part_mask = ~held_mask
        held_rows = np.flatnonzero(held_mask)
        part_rows = np.flatnonzero(part_mask)
        inner_sizes.append(
            {
                "held": int(held_rows.size),
                "train": int(part_rows.size),
                "held_active": int(train_labels[held_rows].sum()),
            }
        )
        if held_rows.size == 0 or part_rows.size == 0:
            raise RunnerError("inner fold split produced an empty side")
        for phi_name in phis:
            spec = FusionSpec(
                name=phi_name,
                rrf_k=int(task.get("rrf_k", 60)),
                shift_rule=str(task.get("shift_rule", "1.0 - min(train_scores)")),
            )
            frozen = fit_fusion(spec, train_scores[part_rows])
            scorer_part = FusionScorer(frozen, train_scores[part_rows])
            scorer_held = FusionScorer(frozen, train_scores[held_rows])
            for k in config.k_list:
                best_value = float("-inf")
                best_columns: tuple[int, ...] = ()
                for columns in iter_combinations(scorer_part.n_receptors, k):
                    value = utility(
                        scorer_part.score_columns(columns),
                        train_labels[part_rows],
                        config.metric,
                        config.alpha,
                    )
                    if value > best_value:
                        best_value = value
                        best_columns = columns
                held_value = utility(
                    scorer_held.score_columns(best_columns),
                    train_labels[held_rows],
                    config.metric,
                    config.alpha,
                )
                inner_scores[phi_name][k].append(held_value)
    means = {
        phi: {
            int(k): (float(np.mean(values)) if values else float("nan"))
            for k, values in per_k.items()
        }
        for phi, per_k in inner_scores.items()
    }
    selected: dict[str, str] = {}
    for k in config.k_list:
        best_phi = None
        best_value = float("-inf")
        for phi_name in phis:
            value = means[phi_name][int(k)]
            if best_phi is None or value > best_value:
                best_phi = phi_name
                best_value = value
        selected[str(int(k))] = str(best_phi)
    return {
        "target_id": str(task["target_id"]),
        "fold": fold,
        "inner_fold_count": inner_fold_count,
        "inner_fold_sizes": inner_sizes,
        "means": means,
        "detail": inner_scores,
        "selected_phi_by_k": selected,
    }


def _rank_columns(panel: LigandPanel, columns: Sequence[int]) -> list[str]:
    return [panel.receptor_ids[int(column)] for column in columns]
def _columns_text(panel: LigandPanel, columns: Sequence[int] | None) -> str:
    if not columns:
        return ""
    return "+".join(_rank_columns(panel, columns))


def bootstrap_key(target_id: str, fold: int, phi: str, k: int) -> str:
    return f"{target_id}|{int(fold)}|{phi}|{int(k)}"


def aggregate_products(
    shards: Iterable[Mapping[str, object]],
    prereg: Mapping[str, object],
    run_id: str,
    panels: Mapping[str, LigandPanel],
    permutations: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Build the long cell metrics, the headroom map and the bootstrap report."""
    cells: list[dict[str, object]] = []
    map_rows: list[dict[str, object]] = []
    bootstrap_config = prereg.get("bootstrap") or {}
    bootstrap_report: dict[str, object] = {
        "schema": "e1_bootstrap_v1",
        "unit": "scaffold_cluster",
        "iterations": int(bootstrap_config.get("iterations", 2000)),
        "seed": int(bootstrap_config.get("seed", 0)),
        "per_cell": {},
    }
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = {}

    for shard in shards:
        target_id = str(shard["target_id"])
        if target_id not in panels:
            continue
        phi = str(shard["phi"])
        fold = int(shard["fold"])
        role = str(shard.get("role", "primary"))
        panel = panels[target_id]
        fold_mask = panel.folds == fold
        test_count = int(fold_mask.sum())
        test_active = int(panel.labels[fold_mask].sum())
        for cell in shard["cells"]:
            k = int(cell["k"])
            perm = None if permutations is None else permutations.get(bootstrap_key(target_id, fold, phi, k))
            null = None if perm is None else perm.get("null")
            noise = cell["noise_floor"]
            common = {
                "target_id": target_id,
                "role": role,
                "fold": fold,
                "phi": phi,
                "k": k,
                "n_lig": test_count,
                "n_active": test_active,
                "n_subsets": int(cell["n_subsets"]),
                "metric": str(prereg.get("primary_metric", "pr_auc")),
                "h_raw": float(cell["h_raw"]),
                "h_nested": float(cell["h_nested"]),
                "h_ab": float(cell["h_ab"]),
                "h_ba": float(cell["h_ba"]),
                "h_nested_lower95": float(cell["h_nested_lower95"]),
                "noise_floor": float(noise["se"]) if noise.get("se") is not None else None,
                "noise_ci95_low": noise.get("ci95_low"),
                "noise_ci95_high": noise.get("ci95_high"),
                "mde": noise.get("mde"),
                "ratio": float(cell["ratio"]),
                "h_perm": None if perm is None else perm.get("h_perm"),
                "h_perm_alt": None if perm is None else perm.get("h_perm_alt"),
                "perm_null_q95": None if not isinstance(null, Mapping) else null.get("q95"),
                "u_test_ref": float(cell["u_test_ref"]),
                "ref_source": str(cell["ref_source"]),
                "ref_columns": _columns_text(panel, cell["ref_columns"]),
                "u_test_oracle": float(cell["u_test_oracle"]),
                "oracle_columns": _columns_text(panel, cell["oracle_columns"]),
                "u_train_oracle": cell.get("u_train_oracle"),
                "u_test_of_train_selected": cell.get("u_test_of_train_selected"),
                "columns_a": _columns_text(panel, cell.get("columns_a")),
                "columns_b": _columns_text(panel, cell.get("columns_b")),
                "columns_train_oracle": _columns_text(panel, cell.get("columns_train_oracle")),
                "seed_tag": f"bootstrap_seed={bootstrap_config.get('seed', 0)}",
                "run_id": run_id,
            }
            method_specs = [
                ("oracle", float(cell["u_test_oracle"]), cell["oracle_columns"], cell["metric_oracle"]),
                ("ref", float(cell["u_test_ref"]), cell["ref_columns"], cell["metric_ref"]),
                ("greedy", float(cell["u_test_greedy"]), cell["greedy_columns"], cell["metric_greedy"]),
                ("single", float(cell["u_test_single"]), cell["single_columns"], cell["metric_single"]),
            ]
            if cell.get("metric_train_selected") is not None:
                method_specs.append(
                    (
                        "train_selected",
                        float(cell["u_test_of_train_selected"]),
                        cell["columns_train_oracle"],
                        cell["metric_train_selected"],
                    )
                )
            for method, u_test, columns, metrics in method_specs:
                row = dict(common)
                row.update(
                    {
                        "method": method,
                        "u_test": float(u_test),
                        "selected_receptor_ids": _columns_text(panel, columns),
                    }
                )
                row.update({name: metrics.get(name) for name in REQUESTED_METRICS})
                cells.append(row)
            bootstrap_report["per_cell"][bootstrap_key(target_id, fold, phi, k)] = {
                "target_id": target_id,
                "fold": fold,
                "phi": phi,
                "k": k,
                "noise_floor": dict(noise),
                "nested": dict(cell["nested_bootstrap"]),
                "h_nested_lower95": float(cell["h_nested_lower95"]),
            }
            grouped.setdefault((target_id, phi, k), []).append({"cell": cell, "role": role, "fold": fold})

    for (target_id, phi, k), entries in sorted(grouped.items()):
        payloads = [entry["cell"] for entry in entries]
        role = str(entries[0]["role"])
        h_nested = np.asarray([float(entry["h_nested"]) for entry in payloads])
        lower = np.asarray([float(entry["h_nested_lower95"]) for entry in payloads])
        noise = np.asarray(
            [
                float(entry["noise_floor"]["se"])
                if entry["noise_floor"].get("se") is not None
                else float("nan")
                for entry in payloads
            ]
        )
        h_raw = np.asarray([float(entry["h_raw"]) for entry in payloads])
        h_perm_values = []
        for entry in entries:
            record = None if permutations is None else permutations.get(
                bootstrap_key(target_id, int(entry["fold"]), phi, k)
            )
            h_perm_values.append(
                float(record["h_perm"]) if record is not None and record.get("h_perm") is not None else float("nan")
            )
        h_perm = np.asarray(h_perm_values)
        go_folds = 0
        for entry in payloads:
            se = entry["noise_floor"].get("se")
            if se is not None and float(entry["h_nested_lower95"]) > float(se) and float(entry["ratio"]) > 1.0:
                go_folds += 1
        finite_noise = noise[np.isfinite(noise)]
        noise_mean = float(finite_noise.mean()) if finite_noise.size else float("nan")
        h_mean = float(np.nanmean(h_nested)) if np.isfinite(h_nested).any() else float("nan")
        ratio_mean = h_mean / noise_mean if np.isfinite(noise_mean) and noise_mean > 0 else float("nan")
        map_rows.append(
            {
                "target_id": target_id,
                "role": role,
                "phi": phi,
                "k": k,
                "n_folds": len(payloads),
                "h_nested": h_mean,
                "h_lower95": float(np.nanmean(lower)) if np.isfinite(lower).any() else float("nan"),
                "noise_floor": noise_mean,
                "ratio": ratio_mean,
                "h_raw": float(np.nanmean(h_raw)) if np.isfinite(h_raw).any() else float("nan"),
                "h_perm": float(np.nanmean(h_perm)) if np.isfinite(h_perm).any() else None,
                "go_folds": go_folds,
                "verdict": (
                    "GO"
                    if (
                        np.isfinite(h_mean)
                        and np.isfinite(noise_mean)
                        and h_mean > noise_mean
                        and ratio_mean > 1.0
                    )
                    else "NO-GO"
                ),
            }
        )
    return {"cells": cells, "map": map_rows, "bootstrap": bootstrap_report}


def k_star_analysis(
    cells: Sequence[Mapping[str, object]],
    prereg: Mapping[str, object],
) -> dict[str, object]:
    """Oracle/train-selected k* per target and the best-phi displacement."""
    targets = [str(value) for value in prereg.get("targets", ())]
    fusion_family = [str(value) for value in prereg.get("fusion_family", ())]
    primary_ks = [int(value) for value in (prereg.get("primary_cells") or {}).get("k", (2, 3))]  # type: ignore[union-attr]
    all_ks = sorted({int(cell["k"]) for cell in cells})
    output: dict[str, object] = {}
    for target in targets:
        target_cells = [cell for cell in cells if str(cell["target_id"]) == target and str(cell["method"]) == "oracle"]
        train_cells = [cell for cell in cells if str(cell["target_id"]) == target and str(cell["method"]) == "train_selected"]
        if not target_cells:
            continue
        folds = sorted({int(cell["fold"]) for cell in target_cells})
        k_star_oracle: dict[str, int] = {}
        k_star_train: dict[str, int] = {}
        phi_mean_nested: dict[str, float] = {}
        for phi in fusion_family:
            for store, source, key in (
                (k_star_oracle, target_cells, "u_test_oracle"),
                (k_star_train, train_cells, "u_test"),
            ):
                values = {}
                for k in all_ks:
                    entries = [
                        float(cell[key])
                        for cell in source
                        if str(cell["phi"]) == phi and int(cell["k"]) == k and cell.get(key) is not None
                    ]
                    values[k] = float(np.mean(entries)) if entries else float("nan")
                finite = {k: v for k, v in values.items() if np.isfinite(v)}
                store[phi] = max(finite, key=lambda k: (finite[k], -k)) if finite else -1
            nested_values = [
                float(cell["h_nested"])
                for cell in target_cells
                if str(cell["phi"]) == phi and int(cell["k"]) in primary_ks
            ]
            phi_mean_nested[phi] = float(np.mean(nested_values)) if nested_values else float("nan")
        best_phi = max(fusion_family, key=lambda phi: (phi_mean_nested.get(phi, float("nan")), -fusion_family.index(phi)))
        output[target] = {
            "folds": folds,
            "k_star_oracle_by_phi": {phi: int(k_star_oracle[phi]) for phi in fusion_family},
            "k_star_train_selected_by_phi": {phi: int(k_star_train[phi]) for phi in fusion_family},
            "phi_mean_nested": phi_mean_nested,
            "best_phi": best_phi,
            "k_star_oracle_mean_phi": int(k_star_oracle.get("mean", -1)),
            "k_star_oracle_best_phi": int(k_star_oracle.get(best_phi, -1)),
            "displacement_best_vs_mean": int(k_star_oracle.get(best_phi, -1)) - int(k_star_oracle.get("mean", -1)),
            "displacement_train_best_vs_mean": int(k_star_train.get(best_phi, -1)) - int(k_star_train.get("mean", -1)),
        }
    return output


def write_products(
    paths: RunPaths,
    aggregated: Mapping[str, object],
    phi_selection: Mapping[str, object],
    k_star: Mapping[str, object],
    prereg: Mapping[str, object],
    run_id: str,
) -> dict[str, object]:
    write_csv(paths.cell_metrics, list(aggregated["cells"]))  # type: ignore[arg-type]
    write_csv(paths.headroom_map, list(aggregated["map"]))  # type: ignore[arg-type]
    write_json(paths.bootstrap_report, aggregated["bootstrap"])
    write_json(
        paths.phi_selection,
        {
            "schema": "e1_phi_selection_v1",
            "method": "inner-CV (3 scaffold folds of the training fold), oracle subset on inner-train, utility on inner-held",
            "targets": phi_selection,
            "k_star": k_star,
        },
    )
    gate = evaluate_gate(
        list(aggregated["cells"]),  # type: ignore[arg-type]
        prereg,
        phi_selection=phi_selection,
        k_displacement=k_star,
    )
    gate["evidence_paths"] = {
        "cell_metrics": paths.cell_metrics.as_posix(),
        "headroom_map": paths.headroom_map.as_posix(),
        "bootstrap_report": paths.bootstrap_report.as_posix(),
        "phi_selection": paths.phi_selection.as_posix(),
        "run_manifest": paths.run_manifest.as_posix(),
        "figures": [paths.figure_headroom.as_posix(), paths.figure_phi.as_posix()],
    }
    gate["run_id"] = run_id
    write_json(paths.gate_g1, gate)
    return gate

def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def git_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def environment_record() -> dict[str, object]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
    }


def _load_panels(
    specs: Sequence[AssetSpec],
) -> tuple[dict[str, LigandPanel], list[dict[str, object]]]:
    panels: dict[str, LigandPanel] = {}
    verification: list[dict[str, object]] = []
    for spec in specs:
        try:
            panel = load_target_panel(spec)
        except asset_module.AssetError as exc:
            verification.append(
                {
                    "target_id": spec.target_id,
                    "asset_key": spec.target_id,
                    "role": spec.role,
                    "status": "missing",
                    "error": str(exc),
                    "spec": spec.as_dict(),
                }
            )
            continue
        report = verify_panel(panel, strict=False)
        report["role"] = spec.role
        report["asset_key"] = spec.target_id
        report["spec"] = spec.as_dict()
        verification.append(report)
        if report["status"] == "ok":
            panels[spec.target_id] = panel
    return panels, verification


def _shard_tasks(
    panels: Mapping[str, LigandPanel],
    specs: Sequence[AssetSpec],
    config: HeadroomConfig,
    prereg: Mapping[str, object],
) -> list[dict[str, object]]:
    roles = {spec.target_id: spec.role for spec in specs}
    tasks: list[dict[str, object]] = []
    rrf_k = int(((prereg.get("fusion_params") or {}).get("rrf_k", 60)))  # type: ignore[union-attr]
    shift_rule = str(
        ((prereg.get("fusion_params") or {}).get("gmean_shift_rule", "1.0 - min(train_scores)"))  # type: ignore[union-attr]
    )
    for target_id, panel in panels.items():
        for fold in panel.fold_ids():
            for phi in prereg["fusion_family"]:  # type: ignore[index]
                tasks.append(
                    {
                        "target_id": target_id,
                        "role": roles.get(target_id, "primary"),
                        "fold": int(fold),
                        "phi": str(phi),
                        "rrf_k": rrf_k,
                        "shift_rule": shift_rule,
                        "config": config.as_dict(),
                        "panel": panel,
                    }
                )
    return tasks


def checkpoint_compatible(
    payload: Mapping[str, object],
    config: HeadroomConfig,
    run_id: str,
) -> bool:
    """A checkpoint is reusable when its frozen knobs match and its k_list covers ours."""
    if payload.get("run_id") != run_id:
        return False
    stored = payload.get("config")
    if not isinstance(stored, Mapping):
        return False
    if str(payload.get("config_hash")) == _config_hash(config):
        return True
    keys = ("metric", "alpha", "top_m", "permutations", "bootstrap_iterations", "bootstrap_seed")
    for key in keys:
        if stored.get(key) != config.as_dict().get(key):
            return False
    stored_ks = {int(value) for value in stored.get("k_list", [])}
    requested_ks = {int(value) for value in config.k_list}
    if not requested_ks.issubset(stored_ks):
        return False
    stored_perm = {int(value) for value in stored.get("perm_ks", [])}
    requested_perm = {int(value) for value in config.perm_ks}
    return requested_perm.issubset(stored_perm)


def _run_shards(
    tasks: Sequence[Mapping[str, object]],
    paths: RunPaths,
    config: HeadroomConfig,
    jobs: int,
    resume: bool,
    run_id: str,
    verbose: bool = True,
) -> dict[tuple[str, int, str], dict[str, object]]:
    paths.cells.mkdir(parents=True, exist_ok=True)
    config_hash = _config_hash(config)
    results: dict[tuple[str, int, str], dict[str, object]] = {}
    pending: list[Mapping[str, object]] = []
    for task in tasks:
        key = (str(task["target_id"]), int(task["fold"]), str(task["phi"]))
        path = paths.cells / shard_filename(*key)
        if resume and path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict) and checkpoint_compatible(payload, config, run_id):
                results[key] = payload
                continue
        pending.append(task)

    def handle(payload: dict[str, object]) -> None:
        key = (str(payload["target_id"]), int(payload["fold"]), str(payload["phi"]))
        path = paths.cells / shard_filename(*key)
        write_json(path, payload)
        results[key] = payload
        if verbose:
            print(
                f"[scan] {key[0]} fold={key[1]} phi={key[2]} "
                f"k={len(payload.get('cells', []))} {payload.get('elapsed_seconds')}s",
                flush=True,
            )

    for payload in parallel_map(jobs, run_shard_task, pending, verbose=verbose):
        handle(payload)
    return results


def _run_permutations(
    shards: Sequence[Mapping[str, object]],
    panels: Mapping[str, LigandPanel],
    config: HeadroomConfig,
    prereg: Mapping[str, object],
    jobs: int,
    verbose: bool = True,
    existing_keys: set[str] | None = None,
) -> dict[str, dict[str, object]]:
    rrf_k = int((prereg.get("fusion_params") or {}).get("rrf_k", 60))  # type: ignore[union-attr]
    shift_rule = str(
        (prereg.get("fusion_params") or {}).get("gmean_shift_rule", "1.0 - min(train_scores)")  # type: ignore[union-attr]
    )
    tasks: list[dict[str, object]] = []
    for shard in shards:
        target_id = str(shard["target_id"])
        fold = int(shard["fold"])
        panel = panels[target_id]
        train_index = np.flatnonzero(panel.folds != fold)
        test_index = np.flatnonzero(panel.folds == fold)
        for cell in shard["cells"]:
            k = int(cell["k"])
            if k not in set(config.perm_ks) or not cell.get("top_masks"):
                continue
            if existing_keys and bootstrap_key(target_id, fold, str(shard["phi"]), k) in existing_keys:
                continue
            tasks.append(
                {
                    "target_id": target_id,
                    "fold": fold,
                    "phi": str(shard["phi"]),
                    "k": k,
                    "rrf_k": rrf_k,
                    "shift_rule": shift_rule,
                    "top_masks": cell["top_masks"],
                    "train_scores": panel.scores[train_index],
                    "test_scores": panel.scores[test_index],
                    "test_labels": panel.labels[test_index],
                    "h_raw": float(cell["h_raw"]),
                    "u_test_oracle": float(cell["u_test_oracle"]),
                    "config": config.as_dict(),
                    "seed": int(config.bootstrap_seed),
                }
            )
    results: dict[str, dict[str, object]] = {}
    if not tasks:
        return results
    for payload in parallel_map(jobs, run_permutation_task, tasks, verbose=verbose):
        results[bootstrap_key(payload["target_id"], payload["fold"], payload["phi"], payload["k"])] = payload
    if verbose:
        print(f"[perm] {len(results)} cells corrected", flush=True)
    return results


def _run_phi_selection(
    panels: Mapping[str, LigandPanel],
    roles: Mapping[str, str],
    config: HeadroomConfig,
    prereg: Mapping[str, object],
    inner_fold_count: int,
    jobs: int,
    verbose: bool = True,
) -> dict[str, dict[str, object]]:
    rrf_k = int((prereg.get("fusion_params") or {}).get("rrf_k", 60))  # type: ignore[union-attr]
    shift_rule = str(
        (prereg.get("fusion_params") or {}).get("gmean_shift_rule", "1.0 - min(train_scores)")  # type: ignore[union-attr]
    )
    eligible = {
        target_id
        for target_id, role in roles.items()
        if role in {"primary", "secondary"}
    }
    tasks = [
        {
            "target_id": target_id,
            "fold": int(fold),
            "phis": list(prereg["fusion_family"]),  # type: ignore[index]
            "rrf_k": rrf_k,
            "shift_rule": shift_rule,
            "inner_fold_count": int(inner_fold_count),
            "config": config.as_dict(),
            "panel": panel,
        }
        for target_id, panel in panels.items()
        if target_id in eligible
        for fold in panel.fold_ids()
    ]
    output: dict[str, dict[str, object]] = {target_id: {} for target_id in eligible if target_id in panels}
    records: list[dict[str, object]] = []
    records = [dict(record) for record in parallel_map(jobs, run_phi_selection_task, tasks, verbose=verbose)]
    for record in records:
        output[str(record["target_id"])][str(record["fold"])] = record
    if verbose:
        print(f"[phi] {len(records)} fold records", flush=True)
    return output


def write_figures(
    paths: RunPaths,
    cells: Sequence[Mapping[str, object]],
    map_rows: Sequence[Mapping[str, object]],
    k_star: Mapping[str, Mapping[str, object]],
    prereg: Mapping[str, object],
) -> list[str]:
    """Two pre-registered figures: headroom map and phi ranking / k* shift."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency path
        return [f"matplotlib unavailable: {exc}"]
    paths.figures.mkdir(parents=True, exist_ok=True)
    fusion_family = [str(value) for value in prereg["fusion_family"]]
    targets = [str(value) for value in prereg["targets"]]
    k_min, k_max = (int(value) for value in prereg["k_range"])
    ks = list(range(k_min, k_max + 1))
    ratio_lookup = {
        (str(row["target_id"]), str(row["phi"]), int(row["k"])): float(row["ratio"])
        for row in map_rows
    }
    fig, axes = plt.subplots(
        nrows=len(targets), ncols=1, figsize=(9.0, 1.7 * len(targets) + 1.0), squeeze=False
    )
    for ax, target in zip(axes[:, 0], targets):
        grid = np.full((len(fusion_family), len(ks)), np.nan)
        for i, phi in enumerate(fusion_family):
            for j, k in enumerate(ks):
                value = ratio_lookup.get((target, phi, k))
                if value is not None and np.isfinite(value):
                    grid[i, j] = value
        image = ax.imshow(grid, cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
        ax.set_title(f"{target}: H_nested / noise_floor", fontsize=9)
        ax.set_yticks(range(len(fusion_family)), fusion_family, fontsize=7)
        ax.set_xticks(range(len(ks)), [f"k={k}" for k in ks], fontsize=7)
        for i in range(len(fusion_family)):
            for j in range(len(ks)):
                value = grid[i, j]
                if np.isfinite(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6)
        fig.colorbar(image, ax=ax, fraction=0.02, pad=0.01)
    fig.tight_layout()
    fig.savefig(paths.figure_headroom, dpi=160)
    plt.close(fig)

    primary_ks = [int(value) for value in (prereg.get("primary_cells") or {}).get("k", (2, 3))]  # type: ignore[union-attr]
    units: dict[tuple[str, int, int], dict[str, float]] = {}
    for cell in cells:
        if str(cell["method"]) != "oracle":
            continue
        if int(cell["k"]) not in primary_ks:
            continue
        key = (str(cell["target_id"]), int(cell["fold"]), int(cell["k"]))
        units.setdefault(key, {})[str(cell["phi"])] = float(cell["h_nested"])
    ranks: list[int] = []
    for values in units.values():
        if "mean" not in values:
            continue
        ordered = sorted(values.items(), key=lambda item: (-item[1], item[0]))
        rank = next(index for index, (phi, _) in enumerate(ordered, start=1) if phi == "mean")
        ranks.append(rank)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12.0, 4.4))
    axes[0].hist(ranks, bins=np.arange(0.5, len(fusion_family) + 1.5), color="#4477aa", edgecolor="white")
    axes[0].set_title("rank of mean among the 8 fusion rules\n(per target x fold x k unit)")
    axes[0].set_xlabel("rank (1 = best)")
    axes[0].set_ylabel("units")
    axes[0].axvline(float(np.mean(ranks)) if ranks else 0.0, color="crimson", linestyle="--")
    labels = []
    mean_ks = []
    best_ks = []
    for target in targets:
        block = k_star.get(target)
        if not block:
            continue
        labels.append(target)
        mean_ks.append(float(block.get("k_star_oracle_mean_phi", np.nan)))
        best_ks.append(float(block.get("k_star_oracle_best_phi", np.nan)))
    axis = axes[1]
    positions = np.arange(len(labels))
    axis.scatter(mean_ks, positions, label="k* under mean", color="#4477aa", zorder=3)
    axis.scatter(best_ks, positions, label="k* under best phi", color="#ee6677", zorder=3)
    for position, left, right in zip(positions, mean_ks, best_ks):
        axis.plot([left, right], [position, position], color="#bbbbbb", zorder=1)
    axis.set_yticks(positions, labels)
    axis.set_xlabel("oracle k* (mean over folds)")
    axis.set_title("k* displacement: mean vs best phi")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(paths.figure_phi, dpi=160)
    plt.close(fig)
    return [paths.figure_headroom.as_posix(), paths.figure_phi.as_posix()]

def input_paths_for(specs: Sequence[AssetSpec], prereg_path: Path, assets_path: Path) -> list[Path]:
    """Every declared input file of the run (for the SHA-256 manifest)."""
    paths: list[Path] = [Path(prereg_path), Path(assets_path)]
    for spec in specs:
        paths.extend(
            path
            for path in (spec.matrix, spec.manifest, spec.problem_json, spec.backfill_problem_json)
            if path
        )
    return paths


def write_input_manifest(
    paths: RunPaths,
    *,
    prereg: Mapping[str, object],
    assets_path: Path,
    roots: Mapping[str, str],
    verification: Sequence[Mapping[str, object]],
    panels: Mapping[str, LigandPanel],
    primary_targets: Sequence[str],
) -> None:
    missing = [target for target in primary_targets if target not in panels]
    specs = [entry.get("spec") for entry in verification]
    input_paths: list[Path] = [Path(str(prereg["_path"])), Path(assets_path)]
    for spec in specs:
        if not isinstance(spec, Mapping):
            continue
        for key in ("matrix", "manifest", "problem_json", "backfill_problem_json"):
            raw = spec.get(key)
            if isinstance(raw, str) and raw:
                input_paths.append(Path(raw))
    write_json(
        paths.input_manifest,
        {
            "schema": "e1_input_manifest_v1",
            "run_id": paths.root.name,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "preregistration": {
                "path": prereg.get("_path"),
                "sha256": prereg.get("_sha256"),
                "schema": prereg.get("schema"),
            },
            "assets_config": Path(assets_path).as_posix(),
            "roots": dict(roots),
            "targets": [dict(entry) for entry in verification],
            "primary_targets_available": sorted(panels),
            "primary_targets_missing": missing,
            "files_sha256": sha256_records(input_paths),
        },
    )


def load_checkpoints(paths: RunPaths) -> list[dict[str, object]]:
    """Load every shard checkpoint on disk, ordered by (target, fold, phi)."""
    payloads = []
    if not paths.cells.is_dir():
        return payloads
    for path in sorted(paths.cells.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("schema") == "e1_shard_v1":
            payloads.append(payload)
    payloads.sort(key=lambda item: (str(item["target_id"]), int(item["fold"]), str(item["phi"])))
    return payloads


def run_e1(
    *,
    prereg_path: Path,
    assets_path: Path,
    output_dir: Path,
    root_overrides: Mapping[str, str] | None = None,
    jobs: int = 1,
    resume: bool = False,
    targets: Sequence[str] | None = None,
    k_list: Sequence[int] | None = None,
    bootstrap_iterations: int | None = None,
    top_m: int | None = None,
    permutations: int | None = None,
    perm_ks: Sequence[int] | None = None,
    inner_fold_count: int = 3,
    skip_perm: bool = False,
    skip_phi_selection: bool = False,
    skip_figures: bool = False,
    with_train_oracle: bool = True,
    max_subsets_per_k: int | None = None,
    allow_missing_primary: bool = False,
    verbose: bool = True,
) -> dict[str, object]:
    """Run the frozen E1 protocol end to end and write every DoD product."""
    started = time.time()
    prereg = load_preregistration(Path(prereg_path))
    paths = build_run_paths(Path(output_dir))
    run_id = paths.root.name
    paths.root.mkdir(parents=True, exist_ok=True)
    roots, specs = load_asset_specs(Path(assets_path), root_overrides)
    if targets:
        wanted = {str(value) for value in targets}
        specs = [spec for spec in specs if spec.target_id in wanted]
    if not specs:
        raise RunnerError("no asset entries selected for this run")

    panels, verification = _load_panels(specs)
    primary_targets = [str(value) for value in prereg["targets"]]
    missing = [target for target in primary_targets if target not in panels]
    if missing and not allow_missing_primary:
        raise RunnerError(
            f"pre-registered primary targets are unavailable: {missing}; "
            "pass --allow-missing-primary for a sensitivity-only battery"
        )

    config = headroom_config_from_prereg(
        prereg,
        k_list=k_list,
        bootstrap_iterations=bootstrap_iterations,
        top_m=top_m,
        permutations=permutations,
        perm_ks=perm_ks,
        with_train_oracle=with_train_oracle,
        max_subsets_per_k=max_subsets_per_k,
    )
    input_paths = input_paths_for(specs, Path(prereg_path), Path(assets_path))
    write_input_manifest(
        paths,
        prereg=prereg,
        assets_path=Path(assets_path),
        roots=roots,
        verification=verification,
        panels=panels,
        primary_targets=primary_targets,
    )
    if verbose:
        for report in verification:
            if report.get("status") == "ok":
                print(
                    f"[assets] {report['target_id']}: {report['ligand_count']} ligands x "
                    f"{report['receptor_count']} receptors, folds={report['fold_count']}, "
                    f"scaffolds={report['scaffold_count']} ({report['source']})",
                    flush=True,
                )
            else:
                print(f"[assets] {report['target_id']}: {report['status']} {report.get('error', '')}", flush=True)

    tasks = _shard_tasks(panels, specs, config, prereg)
    shards = _run_shards(tasks, paths, config, jobs, resume, run_id, verbose=verbose)
    ordered_shards = [shards[key] for key in sorted(shards)]
    permutations_path = paths.root / "permutations.json"
    permutations_map: dict[str, dict[str, object]] = {}
    if permutations_path.is_file():
        permutations_map = dict(
            json.loads(permutations_path.read_text(encoding="utf-8")).get("per_cell", {})
        )
    if not skip_perm:
        permutations_map.update(
            _run_permutations(
                ordered_shards,
                panels,
                config,
                prereg,
                jobs,
                verbose=verbose,
                existing_keys=set(permutations_map),
            )
        )
    if permutations_map:
        write_json(permutations_path, {"schema": "e1_permutations_v1", "per_cell": permutations_map})

    phi_selection: dict[str, dict[str, object]] = {}
    if paths.phi_selection.is_file():
        phi_selection = dict(
            json.loads(paths.phi_selection.read_text(encoding="utf-8")).get("targets", {})
        )
    if not skip_phi_selection and panels:
        roles = {spec.target_id: spec.role for spec in specs}
        phi_selection.update(
            _run_phi_selection(panels, roles, config, prereg, inner_fold_count, jobs, verbose=verbose)
        )

    aggregated = aggregate_products(ordered_shards, prereg, run_id, panels, permutations_map)
    k_star = k_star_analysis(aggregated["cells"], prereg)
    gate = write_products(paths, aggregated, phi_selection, k_star, prereg, run_id)
    if not skip_figures:
        write_figures(paths, aggregated["cells"], aggregated["map"], k_star, prereg)

    run_manifest = {
        "schema": RUN_SCHEMA,
        "run_id": run_id,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 3),
        "git_commit": git_commit(repository_root()),
        "environment": environment_record(),
        "preregistration": {
            "path": prereg["_path"],
            "sha256": prereg["_sha256"],
            "schema": prereg.get("schema"),
        },
        "assets_config": Path(assets_path).as_posix(),
        "config": config.as_dict(),
        "config_hash": _config_hash(config),
        "seed": int(config.bootstrap_seed),
        "shard_count": len(ordered_shards),
        "targets": {
            report.get("target_id"): {
                "role": report.get("role"),
                "status": report.get("status"),
                "source": report.get("source"),
                "source_paths": report.get("source_paths"),
                "ligand_count": report.get("ligand_count"),
                "receptor_count": report.get("receptor_count"),
                "receptor_order": report.get("receptor_order"),
                "fold_count": report.get("fold_count"),
                "per_fold": report.get("per_fold"),
                "scaffold_count": report.get("scaffold_count"),
                "backfilled_scaffold_rows": report.get("backfilled_scaffold_rows"),
                "backfilled_fold_rows": report.get("backfilled_fold_rows"),
                "problems": report.get("problems"),
            }
            for report in verification
        },
        "input_sha256": sha256_records(input_paths),
        "products": {
            "input_manifest": paths.input_manifest.as_posix(),
            "cell_metrics": paths.cell_metrics.as_posix(),
            "headroom_map": paths.headroom_map.as_posix(),
            "bootstrap_report": paths.bootstrap_report.as_posix(),
            "phi_selection": paths.phi_selection.as_posix(),
            "gate_g1": paths.gate_g1.as_posix(),
            "figures": [paths.figure_headroom.as_posix(), paths.figure_phi.as_posix()],
        },
        "gate": {
            "decision": gate.get("decision"),
            "fraction_cells_go": gate.get("fraction_cells_go"),
            "fraction_cells_go_fold_oracle_phi": gate.get("fraction_cells_go_fold_oracle_phi"),
            "fraction_cells_go_train_selected_phi": gate.get("fraction_cells_go_train_selected_phi"),
        },
        "quick_mode": bool(skip_perm or skip_phi_selection),
        "notes": list(prereg.get("notes", [])) if isinstance(prereg.get("notes"), list) else [],
    }
    write_json(paths.run_manifest, run_manifest)
    if verbose:
        print(
            f"[done] {run_id}: gate={gate.get('decision')} "
            f"go(train-selected)={gate.get('fraction_cells_go_train_selected_phi')} "
            f"in {run_manifest['elapsed_seconds']}s",
            flush=True,
        )
    return {
        "run_id": run_id,
        "paths": {
            "root": paths.root.as_posix(),
            "cell_metrics": paths.cell_metrics.as_posix(),
            "headroom_map": paths.headroom_map.as_posix(),
            "gate_g1": paths.gate_g1.as_posix(),
        },
        "gate": gate,
        "run_manifest": run_manifest,
        "input_manifest": paths.input_manifest.as_posix(),
    }


def assemble_products(
    *,
    prereg_path: Path,
    assets_path: Path,
    output_dir: Path,
    root_overrides: Mapping[str, str] | None = None,
    jobs: int = 1,
    inner_fold_count: int | None = None,
    skip_figures: bool = False,
    targets: Sequence[str] | None = None,
    verbose: bool = True,
) -> dict[str, object]:
    """Rebuild every product from shard checkpoints, filling missing passes.

    Shard checkpoints are the source of truth.  Permutation corrections and
    train-only phi selections are merged into ``permutations.json`` /
    ``phi_selection.json`` and only the missing cells are computed, so the
    assemble step is resumable and cheap.
    """
    prereg = load_preregistration(Path(prereg_path))
    paths = build_run_paths(Path(output_dir))
    run_id = paths.root.name
    roots, specs = load_asset_specs(Path(assets_path), root_overrides)
    selected = {
        str(value) for value in (targets or ())
    }
    if selected:
        specs = [spec for spec in specs if spec.target_id in selected]
    panels, _ = _load_panels(specs)
    shards = [
        shard
        for shard in load_checkpoints(paths)
        if not selected or str(shard["target_id"]) in selected
    ]
    if not shards:
        raise RunnerError(f"no shard checkpoints under {paths.cells}")
    stored = shards[0].get("config")
    if not isinstance(stored, Mapping):
        raise RunnerError("shard checkpoints are missing their frozen config")
    config = HeadroomConfig(**dict(stored))

    permutations_path = paths.root / "permutations.json"
    permutations_map: dict[str, dict[str, object]] = {}
    if permutations_path.is_file():
        permutations_map = dict(
            json.loads(permutations_path.read_text(encoding="utf-8")).get("per_cell", {})
        )
    permutations_map.update(
        _run_permutations(
            shards,
            panels,
            config,
            prereg,
            jobs,
            verbose=verbose,
            existing_keys=set(permutations_map),
        )
    )
    write_json(permutations_path, {"schema": "e1_permutations_v1", "per_cell": permutations_map})

    roles = {spec.target_id: spec.role for spec in specs}
    phi_selection: dict[str, dict[str, object]] = {}
    if paths.phi_selection.is_file():
        phi_selection = dict(
            json.loads(paths.phi_selection.read_text(encoding="utf-8")).get("targets", {})
        )
    eligible = {
        target_id: panel
        for target_id, panel in panels.items()
        if roles.get(target_id) in {"primary", "secondary"}
    }
    missing_phi = {
        target_id: panel
        for target_id, panel in eligible.items()
        if not phi_selection.get(target_id)
    }
    if missing_phi:
        folds = int(
            inner_fold_count
            if inner_fold_count is not None
            else (prereg.get("phi_selection") or {}).get("inner_fold_count", 3)  # type: ignore[union-attr]
        )
        phi_selection.update(
            _run_phi_selection(
                missing_phi, roles, config, prereg, folds, jobs, verbose=verbose
            )
        )

    verification = []
    for spec in specs:
        if spec.target_id not in panels:
            verification.append(
                {"target_id": spec.target_id, "asset_key": spec.target_id, "role": spec.role, "status": "missing"}
            )
            continue
        report = verify_panel(panels[spec.target_id], strict=False)
        report["role"] = spec.role
        report["asset_key"] = spec.target_id
        report["spec"] = spec.as_dict()
        verification.append(report)
    write_input_manifest(
        paths,
        prereg=prereg,
        assets_path=Path(assets_path),
        roots=roots,
        verification=verification,
        panels=panels,
        primary_targets=[str(value) for value in prereg["targets"]],
    )

    aggregated = aggregate_products(shards, prereg, run_id, panels, permutations_map)
    k_star = k_star_analysis(aggregated["cells"], prereg)
    gate = write_products(paths, aggregated, phi_selection, k_star, prereg, run_id)
    if not skip_figures:
        write_figures(paths, aggregated["cells"], aggregated["map"], k_star, prereg)
    write_json(
        paths.run_manifest,
        {
            "schema": RUN_SCHEMA,
            "run_id": run_id,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "mode": "assemble",
            "git_commit": git_commit(repository_root()),
            "environment": environment_record(),
            "preregistration": {
                "path": prereg.get("_path"),
                "sha256": prereg.get("_sha256"),
                "schema": prereg.get("schema"),
            },
            "config": config.as_dict(),
            "config_hash": _config_hash(config),
            "shard_count": len(shards),
            "permutation_cells": len(permutations_map),
            "phi_records": len(phi_selection),
            "k_coverage": {
                str(target): sorted({int(cell["k"]) for shard in shards if str(shard["target_id"]) == str(target) for cell in shard["cells"]})
                for target in sorted({str(shard["target_id"]) for shard in shards})
            },
            "notes": list(prereg.get("notes", []))
            + [
                "pool30 runs at k<=4 (pre-registered contingency in the E1 plan section 10)",
                "H_perm as pre-registered is dominated by the absolute metric level; G1 uses H_nested only (see docs/headroom_scan_zh.md section 6.4)",
                "sensitivity: only the MK14 min-aggregation matrix is available locally; independent seed matrices and alternative scaffold splits were not run",
            ],
            "targets": {
                str(entry.get("asset_key") or entry.get("target_id")): {
                    "role": entry.get("role"),
                    "status": entry.get("status"),
                    "source": entry.get("source"),
                    "source_paths": entry.get("source_paths"),
                    "ligand_count": entry.get("ligand_count"),
                    "receptor_count": entry.get("receptor_count"),
                    "receptor_order": entry.get("receptor_order"),
                    "fold_count": entry.get("fold_count"),
                    "scaffold_count": entry.get("scaffold_count"),
                    "backfilled_scaffold_rows": entry.get("backfilled_scaffold_rows"),
                    "backfilled_fold_rows": entry.get("backfilled_fold_rows"),
                    "problems": entry.get("problems"),
                }
                for entry in verification
            },
            "products": {
                "input_manifest": paths.input_manifest.as_posix(),
                "cell_metrics": paths.cell_metrics.as_posix(),
                "headroom_map": paths.headroom_map.as_posix(),
                "bootstrap_report": paths.bootstrap_report.as_posix(),
                "phi_selection": paths.phi_selection.as_posix(),
                "gate_g1": paths.gate_g1.as_posix(),
                "figures": [paths.figure_headroom.as_posix(), paths.figure_phi.as_posix()],
            },
            "gate": {
                "decision": gate.get("decision"),
                "fraction_cells_go": gate.get("fraction_cells_go"),
                "fraction_cells_go_fold_oracle_phi": gate.get("fraction_cells_go_fold_oracle_phi"),
                "fraction_cells_go_train_selected_phi": gate.get("fraction_cells_go_train_selected_phi"),
            },
        },
    )
    if verbose:
        print(
            f"[assemble] {run_id}: {len(shards)} shards, {len(permutations_map)} perm cells, "
            f"{len(phi_selection)} phi records, gate={gate.get('decision')}",
            flush=True,
        )
    return {
        "run_id": run_id,
        "shard_count": len(shards),
        "permutation_cells": len(permutations_map),
        "phi_records": len(phi_selection),
        "gate": gate,
    }


def rebuild_products(**kwargs) -> dict[str, object]:
    """Backwards-compatible alias of :func:`assemble_products`."""
    return assemble_products(**kwargs)
