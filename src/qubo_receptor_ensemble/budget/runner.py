"""E2 runner: sharded budget simulation, checkpoints, products and figures."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..headroom.assets import (
    LigandPanel,
    load_asset_specs,
    load_target_panel,
    sha256_records,
    verify_panel,
)
from ..headroom.fusion import FusionScorer, FusionSpec, fit_fusion
from ..headroom.runner import environment_record, git_commit, parallel_map, repository_root
from ..io import file_sha256, write_csv, write_json
from .features import aggregate_fold_features, build_law_summary, fold_train_features
from .fusion_ragged import build_ragged_fusion, fuse_ragged
from .law import (
    BudgetConfig,
    FoldEvaluation,
    curve_rows,
    detect_b_star,
    evaluate_gate_g2,
    macro_bootstrap_delta,
    paired_fold_deltas,
)
from .metrics import BUDGET_METRICS, budget_metric_values
from .policies import BudgetSimulator, PolicySpec, policy_family, receptor_order, run_policy

PREREG_SCHEMA = "e2_budget_v1"
RUN_SCHEMA = "e2_run_v1"


class BudgetRunnerError(RuntimeError):
    """Raised when an E2 run cannot proceed as pre-registered."""


@dataclass(frozen=True)
class BudgetPaths:
    root: Path
    figures: Path

    @property
    def cells(self) -> Path:
        return self.root / "cells"

    @property
    def input_manifest(self) -> Path:
        return self.root / "input_manifest.json"

    @property
    def budget_cells(self) -> Path:
        return self.root / "budget_cells.csv"

    @property
    def budget_curves(self) -> Path:
        return self.root / "budget_curves.csv"

    @property
    def policy_comparisons(self) -> Path:
        return self.root / "policy_comparisons.csv"

    @property
    def b_star(self) -> Path:
        return self.root / "b_star.json"

    @property
    def law_features(self) -> Path:
        return self.root / "law_features.csv"

    @property
    def law_summary(self) -> Path:
        return self.root / "law_summary.json"

    @property
    def gate_g2(self) -> Path:
        return self.root / "gate_g2.json"

    @property
    def run_manifest(self) -> Path:
        return self.root / "run_manifest.json"

    @property
    def figure_curves(self) -> Path:
        return self.figures / "fig_budget_curves.png"

    @property
    def figure_b_star(self) -> Path:
        return self.figures / "fig_b_star.png"

    @property
    def figure_law(self) -> Path:
        return self.figures / "fig_law.png"


def build_paths(output_dir: Path) -> BudgetPaths:
    root = Path(output_dir)
    return BudgetPaths(root=root, figures=root / "figures")


def load_preregistration(path: Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != PREREG_SCHEMA:
        raise BudgetRunnerError(f"pre-registration schema must be {PREREG_SCHEMA}")
    for key in ("targets", "budgets", "fusion_family", "policies", "primary_metric", "gate_g2"):
        if key not in payload:
            raise BudgetRunnerError(f"pre-registration is missing {key}")
    payload["_sha256"] = file_sha256(Path(path))
    payload["_path"] = Path(path).as_posix()
    return payload


def config_from_prereg(prereg: Mapping[str, object], **overrides) -> BudgetConfig:
    bootstrap = prereg.get("bootstrap") if isinstance(prereg.get("bootstrap"), Mapping) else {}
    oracle = prereg.get("oracle") if isinstance(prereg.get("oracle"), Mapping) else {}
    base = BudgetConfig(
        budgets=tuple(int(value) for value in prereg.get("budgets", ())),
        fusions=tuple(str(value) for value in prereg.get("fusion_family", ())),
        primary_metric=str(prereg.get("primary_metric", "pr_auc")),
        alpha=float(prereg.get("bedroc_alpha", 20.0)),
        bootstrap_iterations=int(bootstrap.get("iterations", 2000)),
        bootstrap_seed=int(bootstrap.get("seed", 0)),
        oracle_greedy_budgets=tuple(int(value) for value in oracle.get("greedy_budgets", (1200, 4800))),
        oracle_greedy_max_steps=oracle.get("greedy_max_steps"),
        baseline=str(prereg.get("baseline", "s1_width")),
    )
    if not overrides:
        return base
    payload = base.as_dict()
    payload.update({key: value for key, value in overrides.items() if value is not None})
    return BudgetConfig(
        budgets=tuple(int(value) for value in payload["budgets"]),
        fusions=tuple(str(value) for value in payload["fusions"]),
        primary_metric=str(payload["primary_metric"]),
        alpha=float(payload["alpha"]),
        bootstrap_iterations=int(payload["bootstrap_iterations"]),
        bootstrap_seed=int(payload["bootstrap_seed"]),
        oracle_greedy_budgets=tuple(int(value) for value in payload["oracle_greedy_budgets"]),
        oracle_greedy_max_steps=payload["oracle_greedy_max_steps"],
        baseline=str(payload["baseline"]),
    )


def evaluate_shard(
    *,
    target_id: str,
    role: str,
    fold: int,
    panel: LigandPanel,
    phi: str,
    config: BudgetConfig,
    policies: Sequence[PolicySpec],
) -> dict[str, object]:
    """Simulate every policy x budget of one (target, fold, phi) shard."""
    train = np.flatnonzero(panel.folds != fold)
    test = np.flatnonzero(panel.folds == fold)
    train_scores = panel.scores[train]
    labels = panel.labels[test]
    eval_scaffolds = tuple(panel.scaffolds[index] for index in test)
    frozen = fit_fusion(FusionSpec(name=phi), train_scores)
    # The budget is spent on the whole 600-ligand library (as in the plan);
    # the held-out fold is only the evaluation subset.
    ragged = build_ragged_fusion(frozen, panel.scores)
    train_scorer = FusionScorer(frozen, train_scores)
    order = receptor_order(train_scorer, panel.labels[train], config.primary_metric, config.alpha)
    features = fold_train_features(
        frozen, train_scores, panel.labels[train], config.primary_metric, config.alpha
    )
    library_scaffolds = {str(scaffold) for scaffold in panel.scaffolds}
    features.update(
        {
            "receptor_count": float(panel.n_receptors),
            "ligand_count": float(panel.n_ligands),
            "scaffold_count": float(len(library_scaffolds)),
            "mean_scaffold_size": float(panel.n_ligands / max(len(library_scaffolds), 1)),
            "train_activity_rate": float(np.asarray(panel.labels[train], dtype=np.float64).mean()),
        }
    )

    baseline_fused: dict[int, np.ndarray] = {}
    cells: list[dict[str, object]] = []
    fused_by_key: dict[str, list[float]] = {}
    for policy in policies:
        for budget in config.budgets:
            simulator = BudgetSimulator(
                panel.scores,
                ragged,
                budget,
                metric=config.primary_metric,
                alpha=config.alpha,
                eval_index=test,
                eval_labels=labels,
            )
            diagnostics = run_policy(
                policy,
                simulator,
                order=order,
                scaffolds=tuple(panel.scaffolds),
                labels=labels if policy.oracle == "metric" else None,
                greedy_max_steps=(
                    config.oracle_greedy_max_steps
                    if policy.oracle == "metric" and budget in set(config.oracle_greedy_budgets)
                    else (0 if policy.oracle == "metric" else None)
                ),
            )
            fused = simulator.fused()
            eval_fused = fused[test]
            values = budget_metric_values(eval_fused, labels, BUDGET_METRICS, config.alpha)
            summary = simulator.summary()
            if policy.name == config.baseline:
                baseline_fused[budget] = eval_fused
            row: dict[str, object] = {
                "target_id": target_id,
                "role": role,
                "fold": int(fold),
                "phi": phi,
                "policy": policy.name,
                "budget": int(budget),
                **summary,
                **values,
                "stages": "|".join(str(item) for item in diagnostics.get("stages", ())),
            }
            cells.append(row)
            fused_by_key[f"{policy.name}|{budget}"] = [float(value) for value in eval_fused]

    for row in cells:
        reference = baseline_fused.get(int(row["budget"]))
        if reference is None:
            row[f"ref_{config.primary_metric}"] = None
            continue
        row[f"ref_{config.primary_metric}"] = float(
            budget_metric_values(reference, labels, [config.primary_metric], config.alpha)[config.primary_metric]
        )
    return {
        "schema": "e2_shard_v1",
        "target_id": target_id,
        "role": role,
        "fold": int(fold),
        "phi": phi,
        "config": config.as_dict(),
        "receptor_order": [int(value) for value in order],
        "test_ligand_count": int(test.size),
        "features": features,
        "cells": cells,
        "labels": [float(value) for value in labels],
        "scaffolds": list(eval_scaffolds),
        "fused": fused_by_key,
    }


def config_from_shards(shards: Sequence[Mapping[str, object]]) -> BudgetConfig:
    """Recover the frozen BudgetConfig from checkpoints (report path)."""
    payloads = [shard.get("config") for shard in shards if isinstance(shard.get("config"), Mapping)]
    if not payloads:
        raise BudgetRunnerError("no shard carries a config payload")
    first = dict(payloads[0])
    for payload in payloads[1:]:
        if dict(payload) != first:
            raise BudgetRunnerError("shard checkpoints disagree on the frozen config")
    return BudgetConfig(**first)  # type: ignore[arg-type]


def _shard_key(target_id: str, fold: int, phi: str) -> str:
    return f"{str(target_id).replace('/', '_')}_{int(fold):02d}_{phi}.json"


def _config_hash(config: BudgetConfig) -> str:
    return hashlib.sha256(json.dumps(config.as_dict(), sort_keys=True).encode("utf-8")).hexdigest().upper()


def run_e2(
    *,
    prereg_path: Path,
    assets_path: Path,
    output_dir: Path,
    jobs: int = 1,
    resume: bool = False,
    targets: Sequence[str] | None = None,
    budgets: Sequence[int] | None = None,
    fusions: Sequence[str] | None = None,
    policies: Sequence[str] | None = None,
    folds: Sequence[int] | None = None,
    verbose: bool = True,
) -> dict[str, object]:
    """Run the frozen E2 battery end to end and write every product."""
    started = time.time()
    prereg = load_preregistration(Path(prereg_path))
    paths = build_paths(Path(output_dir))
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.figures.mkdir(parents=True, exist_ok=True)
    run_id = paths.root.name
    config = config_from_prereg(
        prereg,
        budgets=tuple(int(value) for value in budgets) if budgets else None,
        fusions=tuple(str(value) for value in fusions) if fusions else None,
    )
    policy_names = tuple(str(value) for value in prereg.get("policies", ())) if policies is None else tuple(policies)
    policy_specs = policy_family(policy_names or None)
    roots, specs = load_asset_specs(Path(assets_path))
    if targets:
        wanted = {str(value) for value in targets}
        specs = [spec for spec in specs if spec.target_id in wanted]
    panels: dict[str, LigandPanel] = {}
    verification: list[dict[str, object]] = []
    for spec in specs:
        try:
            panel = load_target_panel(spec)
        except Exception as exc:  # noqa: BLE001 - reported in the manifest
            verification.append({"asset_key": spec.target_id, "role": spec.role, "status": "missing", "error": str(exc)})
            continue
        report = verify_panel(panel, strict=False)
        report["asset_key"] = spec.target_id
        report["role"] = spec.role
        verification.append(report)
        if report["status"] == "ok":
            panels[spec.target_id] = panel
    roles = {spec.target_id: spec.role for spec in specs}
    if not panels:
        raise BudgetRunnerError("no readable assets for this run")

    tasks: list[dict[str, object]] = []
    for target_id, panel in panels.items():
        for fold in panel.fold_ids():
            if folds and int(fold) not in set(int(value) for value in folds):
                continue
            for phi in config.fusions:
                tasks.append(
                    {
                        "target_id": target_id,
                        "role": roles.get(target_id, "primary"),
                        "fold": int(fold),
                        "phi": str(phi),
                        "panel": panel,
                        "config": config.as_dict(),
                        "policies": [spec.as_dict() for spec in policy_specs],
                        "run_id": run_id,
                    }
                )

    def compute(task: Mapping[str, object]) -> dict[str, object]:
        payload = evaluate_shard(
            target_id=str(task["target_id"]),
            role=str(task["role"]),
            fold=int(task["fold"]),
            panel=task["panel"],  # type: ignore[arg-type]
            phi=str(task["phi"]),
            config=BudgetConfig(**dict(task["config"])),  # type: ignore[arg-type]
            policies=tuple(PolicySpec(**dict(item)) for item in task["policies"]),  # type: ignore[index]
        )
        payload["run_id"] = str(task["run_id"])
        payload["config_hash"] = _config_hash(BudgetConfig(**dict(task["config"])))  # type: ignore[arg-type]
        return payload

    config_hash = _config_hash(config)
    shards: list[dict[str, object]] = []
    pending: list[Mapping[str, object]] = []
    for task in tasks:
        path = paths.cells / _shard_key(str(task["target_id"]), int(task["fold"]), str(task["phi"]))
        if resume and path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict) and payload.get("config_hash") == config_hash and payload.get("run_id") == run_id:
                shards.append(payload)
                continue
        pending.append(task)
    for payload in parallel_map(jobs, compute, pending, verbose=verbose):
        path = paths.cells / _shard_key(str(payload["target_id"]), int(payload["fold"]), str(payload["phi"]))
        write_json(path, payload)
        shards.append(payload)
        if verbose:
            print(
                f"[budget] {payload['target_id']} fold={payload['fold']} phi={payload['phi']} "
                f"rows={len(payload['cells'])}",
                flush=True,
            )
    input_paths: list[Path] = [Path(prereg_path), Path(assets_path)]
    for panel in panels.values():
        input_paths.extend(Path(value) for value in panel.source_paths.values())
    write_json(
        paths.input_manifest,
        {
            "schema": "e2_input_manifest_v1",
            "run_id": run_id,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "preregistration": {"path": prereg["_path"], "sha256": prereg["_sha256"]},
            "assets_config": {
                "path": Path(assets_path).as_posix(),
                "sha256": file_sha256(Path(assets_path)),
            },
            "roots": roots,
            "policies": list(policy_names),
            "targets": verification,
            "files_sha256": sha256_records(input_paths),
            "config": config.as_dict(),
        },
    )
    products = assemble_products(
        paths=paths,
        shards=shards,
        prereg=prereg,
        config=config,
        run_id=run_id,
        panels=panels,
        elapsed=time.time() - started,
        verification=verification,
        policies=policy_names,
        assets_path=Path(assets_path),
    )
    if verbose:
        gate = products["gate_g2"]
        print(f"[done] {run_id}: G2={gate.get('decision')} in {products['elapsed_seconds']}s", flush=True)
    return products


def assemble_products(
    *,
    paths: BudgetPaths,
    shards: Sequence[Mapping[str, object]],
    prereg: Mapping[str, object],
    config: BudgetConfig,
    run_id: str,
    panels: Mapping[str, LigandPanel],
    elapsed: float | None = None,
    verification: Sequence[Mapping[str, object]] = (),
    with_bootstrap: bool = True,
    policies: Sequence[str] | None = None,
    assets_path: Path | None = None,
) -> dict[str, object]:
    """Aggregate shard checkpoints into the E2 products and figures."""
    assembly_started = time.time()
    cells: list[dict[str, object]] = []
    for shard in shards:
        cells.extend(dict(row) for row in shard["cells"])  # type: ignore[index]
    curves = curve_rows(cells, config.primary_metric)

    comparisons: list[dict[str, object]] = []
    by_key: dict[tuple[str, str, int], dict[str, list[dict[str, object]]]] = {}
    for row in cells:
        key = (str(row["target_id"]), str(row["phi"]), int(row["budget"]))
        by_key.setdefault(key, {}).setdefault(str(row["policy"]), []).append(row)
    evaluations_by_target: dict[str, list[FoldEvaluation]] = {}
    for shard in shards:
        target_id = str(shard["target_id"])
        fused = {tuple(str(key).split("|")): np.asarray(value, dtype=np.float64) for key, value in dict(shard["fused"]).items()}  # type: ignore[union-attr]
        labels = np.asarray(shard["labels"], dtype=np.float64)
        scaffolds = tuple(str(value) for value in shard["scaffolds"])  # type: ignore[index]
        policy_fused = {
            (policy, str(shard["phi"]), int(budget)): vector for (policy, budget), vector in fused.items()
        }
        evaluations_by_target.setdefault(target_id, []).append(
            FoldEvaluation(
                fold=int(shard["fold"]),
                labels=labels,
                scaffolds=scaffolds,
                policy_fused=policy_fused,
            )
        )

    primary_targets = [str(value) for value in prereg.get("targets", ())]
    for (target_id, phi, budget), policies in sorted(by_key.items()):
        baseline_entries = policies.get(config.baseline, [])
        if not baseline_entries:
            continue
        for policy, entries in sorted(policies.items()):
            if policy == config.baseline:
                continue
            shared_folds, deltas = paired_fold_deltas(entries, baseline_entries, config.primary_metric)
            if not shared_folds:
                continue
            row: dict[str, object] = {
                "target_id": target_id,
                "phi": phi,
                "policy": policy,
                "budget": int(budget),
                "n_folds": len(deltas),
                "folds": "|".join(str(fold) for fold in shared_folds),
                "mean_delta": float(np.mean(deltas)),
                "worst_fold_delta": float(np.min(deltas)),
                "positive_folds": int(sum(1 for value in deltas if value > 0)),
            }
            if with_bootstrap and target_id in evaluations_by_target:
                report = macro_bootstrap_delta(
                    evaluations_by_target[target_id],
                    policy,
                    phi,
                    int(budget),
                    config.baseline,
                    primary_metric=config.primary_metric,
                    alpha=config.alpha,
                    iterations=config.bootstrap_iterations,
                    seed=config.bootstrap_seed,
                )
                row.update({key: value for key, value in report.items() if key != "per_fold_se"})
            comparisons.append(row)

    b_star = detect_b_star(curves, primary_metric=config.primary_metric, baseline=config.baseline)
    gate = evaluate_gate_g2(comparisons, b_star, prereg)
    resolved_policies = (
        tuple(str(value) for value in policies)
        if policies
        else tuple(sorted({str(row["policy"]) for row in cells}))
    )
    prerereg_config = config_from_prereg(prereg)
    gate["policies"] = list(resolved_policies)
    gate["config_matches_preregistration"] = config.as_dict() == prerereg_config.as_dict()
    law_rows = _collect_law_rows(shards, b_star, comparisons, config.budgets)

    write_csv(paths.budget_cells, cells)
    write_csv(paths.budget_curves, curves)
    write_csv(paths.policy_comparisons, comparisons)
    if law_rows:
        write_csv(paths.law_features, law_rows)
        write_json(paths.law_summary, build_law_summary(law_rows, config.budgets))
    write_json(
        paths.b_star,
        {
            "schema": "e2_b_star_v1",
            "primary_metric": config.primary_metric,
            "baseline": config.baseline,
            "targets": b_star,
        },
    )
    gate["run_id"] = run_id
    write_json(paths.gate_g2, gate)
    figures = write_figures(paths, curves, b_star, prereg, config)
    if law_rows:
        figures.extend(write_law_figure(paths, law_rows, config))
    assembly_elapsed = time.time() - assembly_started
    manifest = {
        "schema": RUN_SCHEMA,
        "run_id": run_id,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(float(elapsed if elapsed is not None else 0.0) + assembly_elapsed, 3),
        "shard_elapsed_seconds": None if elapsed is None else round(float(elapsed), 3),
        "assembly_elapsed_seconds": round(assembly_elapsed, 3),
        "git_commit": git_commit(repository_root()),
        "environment": environment_record(),
        "preregistration": {"path": prereg.get("_path"), "sha256": prereg.get("_sha256")},
        "assets_config": (
            {"path": assets_path.as_posix(), "sha256": file_sha256(assets_path)}
            if assets_path is not None and assets_path.is_file()
            else None
        ),
        "policies": list(resolved_policies),
        "config": config.as_dict(),
        "config_matches_preregistration": gate.get("config_matches_preregistration"),
        "shard_count": len(shards),
        "cell_rows": len(cells),
        "comparisons": len(comparisons),
        "targets": {entry.get("asset_key"): {"status": entry.get("status"), "role": entry.get("role")} for entry in verification},
        "products": {
            "budget_cells": paths.budget_cells.as_posix(),
            "budget_curves": paths.budget_curves.as_posix(),
            "policy_comparisons": paths.policy_comparisons.as_posix(),
            "b_star": paths.b_star.as_posix(),
            "law_features": paths.law_features.as_posix() if law_rows else None,
            "law_summary": paths.law_summary.as_posix() if law_rows else None,
            "gate_g2": paths.gate_g2.as_posix(),
            "figures": figures,
        },
        "gate": {"decision": gate.get("decision"), "fraction_width_locked": gate.get("fraction_width_locked")},
    }
    write_json(paths.run_manifest, manifest)
    return {
        "run_id": run_id,
        "elapsed_seconds": manifest["elapsed_seconds"],
        "shard_count": len(shards),
        "cell_rows": len(cells),
        "gate_g2": gate,
        "b_star": b_star,
        "paths": manifest["products"],
    }


def _matched_budget_gains(
    comparisons: Sequence[Mapping[str, object]],
    target_id: str,
    phi: str,
    budgets: Sequence[int],
    baseline: str,
) -> dict[str, object]:
    """Best non-baseline delta at the first and the largest budget (matched budget)."""
    output: dict[str, object] = {}
    if not budgets:
        return output
    for label, budget in (("best_gain_first_budget", min(budgets)), ("best_gain_max_budget", max(budgets))):
        rows = [
            row
            for row in comparisons
            if str(row.get("target_id")) == target_id
            and str(row.get("phi")) == phi
            and int(row.get("budget")) == int(budget)
            and str(row.get("policy")) != baseline
        ]
        if not rows:
            continue
        best = max(rows, key=lambda row: float(row.get("mean_delta", float("-inf"))))
        output[label] = float(best.get("mean_delta"))
        output[f"{label}_policy"] = str(best.get("policy"))
    return output


def _collect_law_rows(
    shards: Sequence[Mapping[str, object]],
    b_star: Mapping[str, Mapping[str, object]],
    comparisons: Sequence[Mapping[str, object]] = (),
    budgets: Sequence[int] = (),
) -> list[dict[str, object]]:
    """Mean train-fold features per (target, phi), merged with B* and gains."""
    grouped: dict[tuple[str, str], list[Mapping[str, float]]] = {}
    for shard in shards:
        features = shard.get("features")
        if not isinstance(features, Mapping):
            continue
        key = (str(shard["target_id"]), str(shard["phi"]))
        grouped.setdefault(key, []).append(features)  # type: ignore[arg-type]
    rows: list[dict[str, object]] = []
    for (target_id, phi), feature_list in sorted(grouped.items()):
        block = b_star.get(f"{target_id}|{phi}", {})
        rows.append(
            {
                "target_id": target_id,
                "phi": phi,
                "b_star": block.get("b_star"),
                "width_locked": block.get("width_locked"),
                **aggregate_fold_features(feature_list),
                **_matched_budget_gains(comparisons, target_id, phi, budgets, "s1_width"),
            }
        )
    return rows


def write_law_figure(
    paths: BudgetPaths,
    law_rows: Sequence[Mapping[str, object]],
    config: BudgetConfig,
) -> list[str]:
    """B* against the two candidate law features (exploratory figure)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        return [f"matplotlib unavailable: {exc}"]
    ceiling = max(config.budgets) + 600
    panels = (
        ("best_single_train_pr_auc", "B* vs best single train PR-AUC"),
        ("pair_gain_train_pr_auc", "B* vs pair-gain train PR-AUC (complementarity)"),
    )
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 7.0), squeeze=False)
    for axis, (name, title) in zip(axes[:, 0], panels):
        for row in law_rows:
            if name not in row:
                continue
            value = float(row[name])
            if not np.isfinite(value):
                continue
            y_value = float(row["b_star"]) if row.get("b_star") is not None else float(ceiling)
            axis.scatter(value, y_value, color="#4477aa")
            axis.annotate(
                f"{row['target_id']}|{row['phi']}",
                (value, y_value),
                fontsize=6,
                textcoords="offset points",
                xytext=(3, 3),
            )
        axis.set_title(title, fontsize=9)
        axis.set_xlabel(name)
        axis.set_ylabel(f"B* (width-locked = {ceiling})")
        axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(paths.figure_law, dpi=160)
    plt.close(fig)
    return [paths.figure_law.as_posix()]


def write_figures(
    paths: BudgetPaths,
    curves: Sequence[Mapping[str, object]],
    b_star: Mapping[str, Mapping[str, object]],
    prereg: Mapping[str, object],
    config: BudgetConfig,
) -> list[str]:
    """Two figures: benefit-budget curves and the B* switching map."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        return [f"matplotlib unavailable: {exc}"]
    preferred = [str(value) for value in prereg.get("targets", ())]
    present = {str(row["target_id"]) for row in curves}
    targets = [target for target in preferred if target in present] + sorted(present.difference(preferred))
    phi = config.fusions[0] if config.fusions else "mean"
    policies = sorted({str(row["policy"]) for row in curves})
    fig, axes = plt.subplots(len(targets), 1, figsize=(9, 2.2 * len(targets) + 1), squeeze=False)
    for axis, target in zip(axes[:, 0], targets):
        block = [row for row in curves if str(row["target_id"]) == target and str(row["phi"]) == phi]
        for policy in policies:
            points = sorted(
                (int(row["budget"]), float(row[f"{config.primary_metric}_mean"]), float(row.get("coverage_mean", 0.0)))
                for row in block
                if str(row["policy"]) == policy
            )
            if not points:
                continue
            axis.plot([item[0] for item in points], [item[1] for item in points], marker="o", label=policy, linewidth=1.2)
        axis.set_title(f"{target} | phi={phi} | {config.primary_metric}", fontsize=9)
        axis.set_xlabel("budget (docking jobs)")
        axis.set_ylabel(config.primary_metric)
        axis.legend(fontsize=6, ncol=3)
        axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(paths.figure_curves, dpi=160)
    plt.close(fig)

    keys = sorted(b_star)
    fig, axis = plt.subplots(figsize=(9, max(3.0, 0.35 * len(keys))))
    labels = [key.split("|")[0] for key in keys]
    positions = np.arange(len(keys))
    values = [
        int(block.get("b_star")) if block.get("b_star") is not None else max(config.budgets) + 600
        for block in (b_star[key] for key in keys)
    ]
    axis.scatter(values, positions, color="#4477aa")
    for position, value in zip(positions, values):
        axis.text(value, position, f"  {value}", va="center", fontsize=7)
    axis.set_yticks(positions, labels, fontsize=7)
    axis.set_xlabel("B* (first budget where the best policy is not s1_width); rightmost = width-locked")
    axis.set_title("E2 critical budget B*")
    fig.tight_layout()
    fig.savefig(paths.figure_b_star, dpi=160)
    plt.close(fig)
    return [paths.figure_curves.as_posix(), paths.figure_b_star.as_posix()]