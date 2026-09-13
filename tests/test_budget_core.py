"""E2 budget allocation: fusion parity, policy contracts, law and runner."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from qubo_receptor_ensemble.budget.features import build_law_summary, fold_train_features
from qubo_receptor_ensemble.budget.fusion_ragged import build_ragged_fusion, fuse_ragged
from qubo_receptor_ensemble.budget.law import (
    BudgetConfig,
    curve_rows,
    detect_b_star,
    evaluate_gate_g2,
    paired_fold_deltas,
)
from qubo_receptor_ensemble.budget.metrics import budget_metric_values, recall_at_fraction
from qubo_receptor_ensemble.budget.policies import (
    BudgetError,
    BudgetSimulator,
    POLICY_NAMES,
    PolicySpec,
    policy_family,
    receptor_order,
    run_policy,
)
from qubo_receptor_ensemble.headroom.assets import LigandPanel
from qubo_receptor_ensemble.headroom.fusion import FUSION_NAMES, FusionScorer, FusionSpec, fit_fusion


def synthetic_library(seed: int = 0, ligands: int = 120, receptors: int = 6, scaffolds: int = 20):
    rng = np.random.default_rng(seed)
    scores = rng.normal(-8.0, 1.2, size=(ligands, receptors))
    labels = np.zeros(ligands)
    labels[rng.choice(ligands, ligands // 5, replace=False)] = 1.0
    scores[labels > 0.5, 0] -= 1.0
    folds = np.asarray([index % 4 for index in range(ligands)], dtype=np.int64)
    scaffold_ids = tuple(f"scaffold_{index % scaffolds:02d}" for index in range(ligands))
    return scores, labels, folds, scaffold_ids


def synthetic_panel() -> LigandPanel:
    scores, labels, folds, scaffolds = synthetic_library()
    return LigandPanel(
        target_id="SYN",
        receptor_ids=tuple(f"R{index}" for index in range(scores.shape[1])),
        ligand_ids=tuple(f"L{index:04d}" for index in range(scores.shape[0])),
        labels=labels,
        scaffolds=scaffolds,
        folds=folds,
        scores=scores,
    )


def test_ragged_fusion_matches_the_materialized_scorer_for_every_rule() -> None:
    scores, _, _, _ = synthetic_library(seed=3)
    mask = np.zeros_like(scores, dtype=bool)
    columns = (1, 3, 4)
    mask[:, columns] = True
    for name in FUSION_NAMES:
        frozen = fit_fusion(FusionSpec(name=name), scores)
        ragged = build_ragged_fusion(frozen, scores)
        expected = FusionScorer(frozen, scores).score_columns(columns)
        got = fuse_ragged(ragged, mask)
        assert np.allclose(got, expected, rtol=1e-12, atol=1e-12), name


def test_ragged_fusion_marks_undocked_ligands_as_minus_inf() -> None:
    scores, _, _, _ = synthetic_library(seed=4)
    ragged = build_ragged_fusion(fit_fusion(FusionSpec(name="mean"), scores), scores)
    mask = np.zeros_like(scores, dtype=bool)
    mask[0, 2] = True
    fused = fuse_ragged(ragged, mask)
    assert np.isneginf(fused[1:]).all()
    assert np.isfinite(fused[0])


def test_receptor_order_is_train_determined_and_permutation_free() -> None:
    scores, labels, _, _ = synthetic_library(seed=5)
    frozen = fit_fusion(FusionSpec(name="mean"), scores)
    scorer = FusionScorer(frozen, scores)
    first = receptor_order(scorer, labels)
    second = receptor_order(scorer, labels)
    assert first == second
    assert sorted(first) == list(range(scores.shape[1]))


def test_policy_budget_invariants_for_the_non_oracle_family() -> None:
    scores, labels, _, scaffolds = synthetic_library(seed=6)
    frozen = fit_fusion(FusionSpec(name="mean"), scores)
    ragged = build_ragged_fusion(frozen, scores)
    order = receptor_order(FusionScorer(frozen, scores), labels)
    for policy in policy_family([name for name in POLICY_NAMES if not name.startswith("s5_metric")]):
        for budget in (30, 120, 481):
            simulator = BudgetSimulator(scores, ragged, budget)
            run_policy(policy, simulator, order=order, scaffolds=scaffolds)
            assert simulator.jobs_used <= budget
            assert len(set(simulator.cells)) == len(simulator.cells)
            assert simulator.mask.sum() == simulator.jobs_used


def test_s1_width_is_the_width_baseline() -> None:
    scores, labels, _, scaffolds = synthetic_library(seed=7)
    frozen = fit_fusion(FusionSpec(name="mean"), scores)
    ragged = build_ragged_fusion(frozen, scores)
    order = receptor_order(FusionScorer(frozen, scores), labels)
    simulator = BudgetSimulator(scores, ragged, 40)
    run_policy(policy_family(["s1_width"])[0], simulator, order=order, scaffolds=scaffolds)
    assert simulator.jobs_used == 40
    depth = simulator.mask.sum(axis=1)
    assert set(depth.tolist()) == {0, 1}  # the budget only covers 40 of the 120 ligands
    assert int((depth == 1).sum()) == 40


def test_s5_metric_oracle_requires_labels_and_respects_the_step_cap() -> None:
    scores, labels, _, scaffolds = synthetic_library(seed=8)
    frozen = fit_fusion(FusionSpec(name="mean"), scores)
    ragged = build_ragged_fusion(frozen, scores)
    order = receptor_order(FusionScorer(frozen, scores), labels)
    spec = PolicySpec(name="s5_metric_oracle", oracle="metric")
    simulator = BudgetSimulator(scores, ragged, 20, eval_index=np.arange(60), eval_labels=labels[:60])
    with pytest.raises(BudgetError):
        run_policy(spec, simulator, order=order, scaffolds=scaffolds, labels=None)
    run_policy(
        spec,
        simulator,
        order=order,
        scaffolds=scaffolds,
        labels=labels,
        greedy_max_steps=5,
    )
    assert simulator.jobs_used == 5


def test_score_oracle_dominates_uniform_depth_per_ligand() -> None:
    """s5_score_oracle is a per-ligand score upper bound, not a metric bound."""

    scores, labels, _, scaffolds = synthetic_library(seed=9)
    frozen = fit_fusion(FusionSpec(name="mean"), scores)
    ragged = build_ragged_fusion(frozen, scores)
    order = receptor_order(FusionScorer(frozen, scores), labels)
    budget = 240
    uniform = BudgetSimulator(scores, ragged, budget)
    run_policy(policy_family(["s2_uniform"])[0], uniform, order=order, scaffolds=scaffolds)
    oracle = BudgetSimulator(scores, ragged, budget)
    run_policy(policy_family(["s5_score_oracle"])[0], oracle, order=order, scaffolds=scaffolds)
    docked = np.isfinite(uniform.fused()) & np.isfinite(oracle.fused())
    assert docked.any()
    assert np.all(oracle.fused()[docked] >= uniform.fused()[docked] - 1e-12)
    assert uniform.jobs_used == oracle.jobs_used == budget


def test_recall_at_fraction_counts_undocked_actives_as_missed() -> None:
    scores = np.asarray([5.0, 4.0, -np.inf, -np.inf])
    labels = np.asarray([1.0, 0.0, 1.0, 0.0])
    assert recall_at_fraction(scores, labels, 0.25) == pytest.approx(0.5)
    # the single best-ranked ligand is a decoy here, so no active is recalled
    scores_decoy_first = np.asarray([-np.inf, -np.inf, 4.0, 5.0])
    assert recall_at_fraction(scores_decoy_first, labels, 0.25) == pytest.approx(0.0)


def test_curve_and_b_star_and_gate() -> None:
    cells = []
    for fold in range(1, 6):
        for policy, value in (("s1_width", 0.40), ("s2_uniform", 0.46), ("s3_top25", 0.35)):
            for budget in (600, 2400):
                cells.append(
                    {
                        "target_id": "T1",
                        "role": "primary",
                        "fold": fold,
                        "phi": "mean",
                        "policy": policy,
                        "budget": budget,
                        "pr_auc": value + 0.001 * fold,
                        "ref_pr_auc": 0.40 + 0.001 * fold,
                        "bedroc20": 0.5,
                        "recall5": 0.3,
                        "coverage": 1.0,
                        "eval_coverage": 1.0,
                        "eval_mean_depth": 1.0,
                        "jobs_used": budget,
                    }
                )
    curves = curve_rows(cells)
    assert len(curves) == 6
    b_star = detect_b_star(curves)
    assert b_star["T1|mean"]["b_star"] == 600
    expected = b_star["T1|mean"]["best_policy_by_budget"]
    assert expected["600"] == "s2_uniform"
    comparisons = [
        {
            "target_id": "T1",
            "phi": "mean",
            "policy": "s2_uniform",
            "budget": 2400,
            "mean_delta": 0.06,
            "ci95_low": 0.01,
        }
    ]
    prereg = {"targets": ["T1"], "primary_metric": "pr_auc", "baseline": "s1_width", "gate_g2": {"depth_budgets": [2400], "min_targets": 1}}
    gate = evaluate_gate_g2(comparisons, b_star, prereg)
    assert gate["decision"] == "NONTRIVIAL_REGION"


def test_runner_end_to_end_on_a_synthetic_panel(headroom_workspace: Path) -> None:
    from test_headroom_runner import write_assets_config, write_dataset, write_prereg

    from qubo_receptor_ensemble.budget import runner as budget_runner

    matrix, manifest = write_dataset(headroom_workspace, ligands=120, receptors=6, folds=3)
    assets = write_assets_config(headroom_workspace, matrix, manifest)
    prereg = headroom_workspace / "e2_prereg.json"
    prereg.write_text(
        (
            '{"schema": "e2_budget_v1", "targets": ["SYN"], "budgets": [120, 240],'
            ' "fusion_family": ["mean"], "policies": ["s1_width", "s2_uniform"],'
            ' "primary_metric": "pr_auc", "baseline": "s1_width",'
            ' "bootstrap": {"iterations": 20, "seed": 0},'
            ' "gate_g2": {"depth_budgets": [120], "min_targets": 1}}'
        ),
        encoding="utf-8",
    )
    kwargs = dict(
        prereg_path=prereg,
        assets_path=assets,
        output_dir=headroom_workspace / "results" / "budget" / "e2_synth",
        jobs=1,
    )
    first = budget_runner.run_e2(**kwargs, resume=False, verbose=False)
    run_dir = Path(kwargs["output_dir"])
    before = {path.name: path.stat().st_mtime_ns for path in sorted((run_dir / "cells").glob("*.json"))}
    assert before
    second = budget_runner.run_e2(**kwargs, resume=True, verbose=False)
    assert second["shard_count"] == first["shard_count"]
    after = {path.name: path.stat().st_mtime_ns for path in sorted((run_dir / "cells").glob("*.json"))}
    assert before == after
    assert (run_dir / "gate_g2.json").is_file()
    rows = list(csv.DictReader((run_dir / "budget_cells.csv").open(encoding="utf-8")))
    assert rows and {row["policy"] for row in rows} == {"s1_width", "s2_uniform"}
    assert (run_dir / "figures").is_dir()

def _comparison(target_id: str, policy: str, budget: int, phi: str, delta: float, ci_low: float):
    return {
        "target_id": target_id,
        "phi": phi,
        "policy": policy,
        "budget": budget,
        "mean_delta": delta,
        "ci95_low": ci_low,
    }


def test_gate_counts_only_primary_targets() -> None:
    prereg = {
        "targets": ["T1"],
        "primary_metric": "pr_auc",
        "baseline": "s1_width",
        "gate_g2": {"depth_budgets": [2400], "min_targets": 1},
    }
    b_star = {
        "T1|mean": {
            "target_id": "T1",
            "phi": "mean",
            "b_star": 2400,
            "width_locked": False,
            "best_policy_by_budget": {"2400": "s2_uniform"},
        }
    }
    secondary_only = [_comparison("FA10", "s2_uniform", 2400, "mean", 0.05, 0.01)]
    gate = evaluate_gate_g2(secondary_only, b_star, prereg)
    assert gate["decision"] != "NONTRIVIAL_REGION"
    assert gate["stable_policy_budget_phi"] == {}
    primary = [_comparison("T1", "s2_uniform", 2400, "mean", 0.05, 0.01)]
    assert evaluate_gate_g2(primary, b_star, prereg)["decision"] == "NONTRIVIAL_REGION"


def test_gate_requires_the_same_phi_across_targets() -> None:
    prereg = {
        "targets": ["T1", "T2"],
        "primary_metric": "pr_auc",
        "baseline": "s1_width",
        "gate_g2": {"depth_budgets": [2400], "min_targets": 2},
    }
    b_star: dict[str, object] = {}
    mixed = [
        _comparison("T1", "s2_uniform", 2400, "mean", 0.05, 0.01),
        _comparison("T2", "s2_uniform", 2400, "min", 0.05, 0.01),
    ]
    gate = evaluate_gate_g2(mixed, b_star, prereg)
    assert gate["decision"] == "GREY_ZONE"
    assert gate["stable_policy_budget_phi"] == {}
    assert gate["stable_policy_budget_any_phi"] == {"s2_uniform@2400": 2}
    same_phi = [
        _comparison("T1", "s2_uniform", 2400, "mean", 0.05, 0.01),
        _comparison("T2", "s2_uniform", 2400, "mean", 0.05, 0.01),
    ]
    assert evaluate_gate_g2(same_phi, b_star, prereg)["decision"] == "NONTRIVIAL_REGION"


def test_paired_fold_deltas_align_by_fold_not_position() -> None:
    entries = [
        {"fold": 3, "pr_auc": 0.60},
        {"fold": 1, "pr_auc": 0.40},
        {"fold": 2, "pr_auc": 0.50},
    ]
    baseline = [
        {"fold": 1, "pr_auc": 0.30},
        {"fold": 2, "pr_auc": 0.45},
        {"fold": 3, "pr_auc": 0.55},
    ]
    folds, deltas = paired_fold_deltas(entries, baseline, "pr_auc")
    assert folds == [1, 2, 3]
    assert deltas == pytest.approx([0.10, 0.05, 0.05])


def test_train_features_detect_signal_and_report_complementarity() -> None:
    scores, labels, _, _ = synthetic_library(seed=11, ligands=200, receptors=5, scaffolds=40)
    scores[labels > 0.5, 0] -= 2.0
    frozen = fit_fusion(FusionSpec(name="mean"), scores)
    features = fold_train_features(frozen, scores, labels)
    assert np.isfinite(features["best_single_train_pr_auc"])
    assert features["best_single_train_pr_auc"] > 0.3
    assert np.isfinite(features["best_pair_train_pr_auc"])
    assert np.isfinite(features["pair_gain_train_pr_auc"])
    assert -1.0 <= features["receptor_diversity_spearman"] <= 1.0


def test_law_summary_reports_correlations_and_groups() -> None:
    rows = [
        {"target_id": "A", "phi": "mean", "b_star": 1200, "best_single_train_pr_auc": 0.3},
        {"target_id": "B", "phi": "mean", "b_star": 2400, "best_single_train_pr_auc": 0.5},
        {"target_id": "C", "phi": "mean", "b_star": None, "best_single_train_pr_auc": 0.7},
    ]
    summary = build_law_summary(rows, (600, 1200, 2400))
    assert summary["n_rows"] == 3
    assert summary["n_width_locked"] == 1
    assert np.isfinite(summary["spearman_feature_vs_b_star_rank"]["best_single_train_pr_auc"])

def test_matched_budget_gains_pick_the_best_non_baseline_policy() -> None:
    from qubo_receptor_ensemble.budget.runner import _matched_budget_gains

    comparisons = [
        {"target_id": "T1", "phi": "mean", "policy": "s1_width", "budget": 600, "mean_delta": 0.0},
        {"target_id": "T1", "phi": "mean", "policy": "s2_uniform", "budget": 600, "mean_delta": 0.01},
        {"target_id": "T1", "phi": "mean", "policy": "s4_scaffold25", "budget": 600, "mean_delta": 0.05},
        {"target_id": "T1", "phi": "mean", "policy": "s2_uniform", "budget": 2400, "mean_delta": 0.02},
        {"target_id": "T1", "phi": "mean", "policy": "s4_scaffold25", "budget": 2400, "mean_delta": 0.03},
        {"target_id": "T2", "phi": "mean", "policy": "s2_uniform", "budget": 600, "mean_delta": 0.5},
    ]
    gains = _matched_budget_gains(comparisons, "T1", "mean", (600, 1200, 2400), "s1_width")
    assert gains["best_gain_first_budget"] == pytest.approx(0.05)
    assert gains["best_gain_first_budget_policy"] == "s4_scaffold25"
    assert gains["best_gain_max_budget"] == pytest.approx(0.03)
    assert gains["best_gain_max_budget_policy"] == "s4_scaffold25"


def test_law_summary_reports_gain_correlations() -> None:
    rows = [
        {
            "target_id": f"T{index}",
            "phi": "mean",
            "b_star": 600,
            "best_single_train_pr_auc": 0.30 + 0.10 * index,
            "best_gain_first_budget": 0.05 + 0.02 * index,
        }
        for index in range(4)
    ]
    summary = build_law_summary(rows, (600, 1200))
    assert summary["n_distinct_b_star"] == 1
    assert np.isfinite(summary["spearman_gain_vs_feature"]["best_gain_first_budget~best_single_train_pr_auc"])
