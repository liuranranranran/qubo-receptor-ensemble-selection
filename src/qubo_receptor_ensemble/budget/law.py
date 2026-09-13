"""E2 aggregation: budget curves, critical budget B*, paired comparisons and G2.

Statistics follow the parent plan section 4: paired per-fold deltas against the
width baseline ``s1_width``, scaffold-cluster bootstrap intervals on the
macro-average delta, an explicit MDE, per-target reporting (never only the
macro average), and a frozen policy family without family-max reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ..headroom.bootstrap import cluster_index, minimum_detectable_effect, percentile
from ..headroom.metrics_fast import metric_value

DEFAULT_BASELINE = "s1_width"


@dataclass(frozen=True)
class BudgetConfig:
    """Frozen knobs of one E2 run."""

    budgets: tuple[int, ...] = (600, 1200, 2400, 3600, 4800)
    fusions: tuple[str, ...] = ("mean", "min")
    primary_metric: str = "pr_auc"
    alpha: float = 20.0
    bootstrap_iterations: int = 2000
    bootstrap_seed: int = 0
    oracle_greedy_budgets: tuple[int, ...] = (1200, 4800)
    oracle_greedy_max_steps: int | None = None
    baseline: str = DEFAULT_BASELINE

    def as_dict(self) -> dict[str, object]:
        return {
            "budgets": list(self.budgets),
            "fusions": list(self.fusions),
            "primary_metric": self.primary_metric,
            "alpha": self.alpha,
            "bootstrap_iterations": self.bootstrap_iterations,
            "bootstrap_seed": self.bootstrap_seed,
            "oracle_greedy_budgets": list(self.oracle_greedy_budgets),
            "oracle_greedy_max_steps": self.oracle_greedy_max_steps,
            "baseline": self.baseline,
        }


@dataclass(frozen=True)
class FoldEvaluation:
    """Fixed allocations of one fold, kept for the paired bootstrap."""

    fold: int
    labels: np.ndarray
    scaffolds: tuple[str, ...]
    policy_fused: Mapping[tuple[str, str, int], np.ndarray]  # (policy, phi, budget) -> fused scores


def curve_rows(cells: Sequence[Mapping[str, object]], primary_metric: str = "pr_auc") -> list[dict[str, object]]:
    """Aggregate per-fold cell rows into ``target x phi x policy x budget`` curves."""
    grouped: dict[tuple[str, str, str, int], list[Mapping[str, object]]] = {}
    for row in cells:
        key = (str(row["target_id"]), str(row["phi"]), str(row["policy"]), int(row["budget"]))
        grouped.setdefault(key, []).append(row)
    curves: list[dict[str, object]] = []
    for (target_id, phi, policy, budget), entries in sorted(grouped.items()):
        values = np.asarray([float(entry[primary_metric]) for entry in entries], dtype=np.float64)
        refs = np.asarray(
            [
                float(entry[f"ref_{primary_metric}"])
                if entry.get(f"ref_{primary_metric}") is not None
                else np.nan
                for entry in entries
            ],
            dtype=np.float64,
        )
        deltas = values - refs
        finite_deltas = deltas[np.isfinite(deltas)]
        curves.append(
            {
                "target_id": target_id,
                "role": entries[0].get("role"),
                "phi": phi,
                "policy": policy,
                "budget": budget,
                "n_folds": len(entries),
                f"{primary_metric}_mean": float(values.mean()),
                f"{primary_metric}_std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "delta_mean": float(finite_deltas.mean()) if finite_deltas.size else float("nan"),
                "delta_worst_fold": float(finite_deltas.min()) if finite_deltas.size else float("nan"),
                "positive_folds": int((finite_deltas > 0).sum()),
                "bedroc20_mean": float(np.mean([float(entry["bedroc20"]) for entry in entries])),
                "recall5_mean": float(np.mean([float(entry["recall5"]) for entry in entries])),
                "coverage_mean": float(np.mean([float(entry["coverage"]) for entry in entries])),
                "eval_coverage_mean": float(np.mean([float(entry["eval_coverage"]) for entry in entries])),
                "eval_mean_depth_mean": float(np.mean([float(entry["eval_mean_depth"]) for entry in entries])),
                "jobs_used_mean": float(np.mean([float(entry["jobs_used"]) for entry in entries])),
            }
        )
    return curves


def macro_bootstrap_delta(
    evaluations: Sequence[FoldEvaluation],
    policy: str,
    phi: str,
    budget: int,
    baseline: str,
    *,
    primary_metric: str,
    alpha: float,
    iterations: int,
    seed: int,
) -> dict[str, object]:
    """Scaffold-cluster bootstrap of the macro-average paired delta.

    For every bootstrap replicate each fold resamples whole scaffolds with
    replacement (keeping within-fold correlation) and the metric difference is
    averaged over folds.
    """
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    per_fold_se: list[float] = []
    for evaluation in evaluations:
        policy_scores = evaluation.policy_fused.get((policy, phi, budget))
        baseline_scores = evaluation.policy_fused.get((baseline, phi, budget))
        if policy_scores is None or baseline_scores is None:
            continue
        names, cluster_rows = cluster_index(list(evaluation.scaffolds))
        draws = [rng.integers(0, len(names), size=len(names)) for _ in range(iterations)]
        values = np.empty(iterations, dtype=np.float64)
        for index, draw in enumerate(draws):
            rows = np.concatenate([cluster_rows[position] for position in draw])
            labels = evaluation.labels[rows]
            values[index] = metric_value(
                policy_scores[rows], labels, primary_metric, alpha
            ) - metric_value(baseline_scores[rows], labels, primary_metric, alpha)
        per_fold_se.append(float(np.nanstd(values, ddof=1)))
        samples.append(values)
    if not samples:
        return {"ci95_low": None, "ci95_high": None, "bootstrap_mean": None, "mde": None, "se": None}
    stacked = np.vstack(samples)
    macro = np.nanmean(stacked, axis=0)
    standard_error = float(np.nanstd(macro, ddof=1))
    return {
        "bootstrap_mean": float(np.nanmean(macro)),
        "se": standard_error,
        "ci95_low": percentile(macro, 0.025),
        "ci95_high": percentile(macro, 0.975),
        "mde": minimum_detectable_effect(standard_error, 20),
        "per_fold_se": [float(value) for value in per_fold_se],
    }


def paired_fold_deltas(
    entries: Sequence[Mapping[str, object]],
    baseline_entries: Sequence[Mapping[str, object]],
    metric: str,
) -> tuple[list[int], list[float]]:
    """Fold-aligned paired deltas ``entry - baseline`` (never positional)."""
    policy_by_fold = {int(entry["fold"]): entry for entry in entries}
    baseline_by_fold = {int(entry["fold"]): entry for entry in baseline_entries}
    folds = sorted(set(policy_by_fold) & set(baseline_by_fold))
    deltas = [
        float(policy_by_fold[fold][metric]) - float(baseline_by_fold[fold][metric])
        for fold in folds
    ]
    return folds, deltas


def detect_b_star(
    curves: Sequence[Mapping[str, object]],
    *,
    primary_metric: str = "pr_auc",
    baseline: str = DEFAULT_BASELINE,
) -> dict[str, object]:
    """Smallest budget where the best policy is not the width baseline.

    Tie-break: the baseline wins ties (conservative for the "non-trivial"
    claim), then the frozen policy order.
    """
    order = {name: index for index, name in enumerate(("s1_width", "s2_uniform", "s3_top10", "s3_top25", "s3_top50", "s4_scaffold25", "s4_scaffold50"))}
    targets = sorted({str(row["target_id"]) for row in curves})
    output: dict[str, object] = {}
    for target in targets:
        block = [row for row in curves if str(row["target_id"]) == target]
        for phi in sorted({str(row["phi"]) for row in block}):
            phi_block = [row for row in block if str(row["phi"]) == phi]
            budgets = sorted({int(row["budget"]) for row in phi_block})
            best_by_budget: dict[str, str] = {}
            for budget in budgets:
                candidates = [row for row in phi_block if int(row["budget"]) == budget]
                best = max(
                    candidates,
                    key=lambda row: (
                        float(row[f"{primary_metric}_mean"]),
                        str(row["policy"]) == baseline,
                        -order.get(str(row["policy"]), len(order)),
                    ),
                )
                best_by_budget[str(budget)] = str(best["policy"])
            b_star = None
            for budget in budgets:
                if best_by_budget[str(budget)] != baseline:
                    b_star = int(budget)
                    break
            output[f"{target}|{phi}"] = {
                "target_id": target,
                "phi": phi,
                "best_policy_by_budget": best_by_budget,
                "b_star": b_star,
                "width_locked": b_star is None,
            }
    return output


def evaluate_gate_g2(
    comparisons: Sequence[Mapping[str, object]],
    b_star: Mapping[str, Mapping[str, object]],
    prereg: Mapping[str, object],
) -> dict[str, object]:
    """Pre-registered G2 decision: width-locked vs non-trivial allocation region."""
    gate = prereg.get("gate_g2") if isinstance(prereg.get("gate_g2"), Mapping) else {}
    depth_budgets = tuple(int(value) for value in gate.get("depth_budgets", (2400, 3600, 4800)))
    min_targets = int(gate.get("min_targets", 4))
    primary_targets = [str(value) for value in prereg.get("targets", ())]
    primary_metric = str(prereg.get("primary_metric", "pr_auc"))
    baseline = str(prereg.get("baseline", DEFAULT_BASELINE))

    width_cells = 0
    width_locked_cells = 0
    for block in b_star.values():
        if str(block.get("target_id")) not in primary_targets:
            continue
        for budget, policy in (block.get("best_policy_by_budget") or {}).items():
            if int(budget) < min(depth_budgets):
                continue
            width_cells += 1
            if str(policy) == baseline:
                width_locked_cells += 1

    # Frozen rule: a non-baseline policy must beat s1_width with the same phi
    # and the same budget in >= min_targets *primary* targets.  Keeping phi
    # fixed avoids selecting the fusion family's maximum row by row.
    strict_keys: dict[str, set[str]] = {}
    loose_keys: dict[str, set[str]] = {}
    nontrivial: list[dict[str, object]] = []
    for row in comparisons:
        policy = str(row.get("policy"))
        if policy == baseline:
            continue
        if str(row.get("target_id")) not in primary_targets:
            continue
        delta = row.get("mean_delta")
        ci_low = row.get("ci95_low")
        if delta is None or float(delta) <= 0:
            continue
        if ci_low is None or float(ci_low) <= 0:
            continue
        nontrivial.append(dict(row))
        strict_key = f"{policy}@{int(row['budget'])}@{row.get('phi')}"
        strict_keys.setdefault(strict_key, set()).add(str(row.get("target_id")))
        loose_keys.setdefault(f"{policy}@{int(row['budget'])}", set()).add(str(row.get("target_id")))
    stable = {
        key: len(targets)
        for key, targets in sorted(strict_keys.items())
        if len(targets) >= min_targets
    }
    stable_any_phi = {
        key: len(targets)
        for key, targets in sorted(loose_keys.items())
        if len(targets) >= min_targets
    }

    fraction_width = width_locked_cells / width_cells if width_cells else float("nan")
    if stable:
        decision = "NONTRIVIAL_REGION"
    elif width_cells and fraction_width >= 1.0:
        decision = "WIDTH_LOCKED"
    else:
        decision = "GREY_ZONE"
    return {
        "schema": "e2_gate_g2_v1",
        "decision": decision,
        "primary_metric": primary_metric,
        "baseline": baseline,
        "depth_budgets": list(depth_budgets),
        "width_locked_cells": width_locked_cells,
        "width_cells": width_cells,
        "fraction_width_locked": fraction_width,
        "stable_policy_budget_phi": stable,
        "stable_policy_budget_any_phi": stable_any_phi,
        "nontrivial_comparisons": nontrivial,
        "targets_with_nonzero_b_star": [
            key
            for key, block in b_star.items()
            if block.get("b_star") is not None and str(block.get("target_id")) in primary_targets
        ],
    }