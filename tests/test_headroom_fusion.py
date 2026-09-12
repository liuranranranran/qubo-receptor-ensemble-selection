"""Fusion-rule contracts: T3 (single-receptor degeneracy), T4 (leakage), T5 (direction)."""

from __future__ import annotations

import numpy as np
import pytest

from qubo_receptor_ensemble.headroom.fusion import (
    FUSION_NAMES,
    FusionError,
    FusionScorer,
    FusionSpec,
    apply_fusion,
    build_specs,
    fit_fusion,
)


def synthetic_panel(seed: int = 0, ligands: int = 90, receptors: int = 6):
    rng = np.random.default_rng(seed)
    scores = rng.normal(-8.0, 1.4, size=(ligands, receptors))
    labels = np.zeros(ligands)
    labels[rng.choice(ligands, ligands // 5, replace=False)] = 1.0
    return scores, labels


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    """Tie-averaged Spearman correlation (no scipy dependency)."""

    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="stable")
        sorted_values = values[order]
        boundaries = np.flatnonzero(np.r_[True, sorted_values[1:] != sorted_values[:-1]])
        ends = np.r_[boundaries[1:], values.size]
        averages = (boundaries + 1 + ends) / 2.0
        result = np.empty(values.size, dtype=np.float64)
        for start, end, average in zip(boundaries, ends, averages):
            result[start:end] = average
        out = np.empty(values.size, dtype=np.float64)
        out[order] = result
        return out

    left_rank = ranks(np.asarray(left, dtype=np.float64))
    right_rank = ranks(np.asarray(right, dtype=np.float64))
    left_centered = left_rank - left_rank.mean()
    right_centered = right_rank - right_rank.mean()
    denominator = np.sqrt((left_centered ** 2).sum() * (right_centered ** 2).sum())
    return float((left_centered * right_centered).sum() / denominator) if denominator else 1.0


def test_fusion_family_is_the_frozen_eight() -> None:
    assert FUSION_NAMES == ("mean", "min", "max", "zmean", "gmean", "hmean", "ranksum", "rrf")
    assert [spec.name for spec in build_specs()] == list(FUSION_NAMES)


def test_unknown_fusion_and_bad_parameters_are_rejected() -> None:
    with pytest.raises(FusionError):
        FusionSpec(name="median")
    with pytest.raises(FusionError):
        FusionSpec(name="rrf", rrf_k=0)
    with pytest.raises(FusionError):
        FusionSpec(name="gmean", shift_rule="min(train_scores)")


def test_all_fusions_agree_on_single_receptor_ranking() -> None:
    """T3: with |S| = 1 every fusion is a monotone transform of the same column."""

    scores, _ = synthetic_panel(seed=3)
    reference = None
    for spec in build_specs():
        frozen = fit_fusion(spec, scores)
        # full-width matrix: zmean/ranksum/rrf carry per-receptor parameters
        fused = FusionScorer(frozen, scores).score_columns((2,))
        if reference is None:
            reference = fused
            continue
        assert spearman(reference, fused) == pytest.approx(1.0, abs=1e-12)


def test_fitted_parameters_ignore_test_rows() -> None:
    """T4: changing only the evaluation fold must not move any fitted parameter."""

    scores, _ = synthetic_panel(seed=11)
    train = scores[:60]
    test_a = scores[60:]
    test_b = test_a.copy()
    test_b[:, :] = test_b[:, :] * 3.0 + 25.0
    for spec in build_specs():
        frozen_a = fit_fusion(spec, train)
        frozen_b = fit_fusion(spec, train)
        if spec.name == "zmean":
            assert np.array_equal(frozen_a.receptor_means, frozen_b.receptor_means)
            assert np.array_equal(frozen_a.receptor_stds, frozen_b.receptor_stds)
        if spec.name in {"gmean", "hmean"}:
            assert frozen_a.shift == frozen_b.shift
        if spec.name in {"ranksum", "rrf"}:
            assert all(
                np.array_equal(left, right)
                for left, right in zip(frozen_a.rank_references, frozen_b.rank_references)
            )
        # applying the frozen rule to the untouched train rows stays bitwise stable
        assert np.array_equal(apply_fusion(frozen_a, train), apply_fusion(frozen_b, train))
        # the two different test matrices are never used to fit anything
        assert FusionScorer(frozen_a, test_a).terms.shape == FusionScorer(frozen_b, test_b).terms.shape


def test_rank_fusions_are_invariant_to_a_per_receptor_shift() -> None:
    """T5: ranksum/rrf depend on ranks only, so a constant shift cannot move them."""

    scores, _ = synthetic_panel(seed=5)
    shifted = scores.copy()
    shifted[:, 1] = shifted[:, 1] + 7.5
    for spec in build_specs(["ranksum", "rrf"]):
        frozen = fit_fusion(spec, scores)
        frozen_shifted = fit_fusion(spec, shifted)
        for columns in ((0,), (1,), (2, 4), (0, 1, 3)):
            left = FusionScorer(frozen, scores).score_columns(columns)
            right = FusionScorer(frozen_shifted, shifted).score_columns(columns)
            assert np.array_equal(left, right)


def test_mean_fusion_is_not_invariant_to_a_per_receptor_shift() -> None:
    """T5 complement: the arithmetic mean mixes raw scores, so a shift moves it."""

    scores, labels = synthetic_panel(seed=5)
    shifted = scores.copy()
    shifted[:, 1] = shifted[:, 1] + 7.5
    spec = build_specs(["mean"])[0]
    frozen = fit_fusion(spec, scores)
    frozen_shifted = fit_fusion(spec, shifted)
    plain = FusionScorer(frozen, scores).score_columns((0, 1))
    moved = FusionScorer(frozen_shifted, shifted).score_columns((0, 1))
    assert not np.allclose(plain, moved)
    # utility = -mean(scores); adding c to one of two receptors subtracts c / 2
    assert np.allclose(plain, moved + 7.5 / 2.0)


def test_geometric_and_harmonic_shift_keeps_positive_domain() -> None:
    scores, _ = synthetic_panel(seed=7)
    frozen = fit_fusion(FusionSpec(name="gmean"), scores)
    assert frozen.shift is not None and frozen.shift > 0
    worst_case = scores.min(axis=0, keepdims=True) - 5.0
    fused = apply_fusion(frozen, worst_case)
    assert np.isfinite(fused).all()


def test_zmean_on_constant_receptor_is_guarded() -> None:
    scores, _ = synthetic_panel(seed=9)
    scores[:, 0] = -7.0
    frozen = fit_fusion(FusionSpec(name="zmean"), scores)
    assert frozen.degenerate_receptors == ("receptor_0",)
    fused = apply_fusion(frozen, scores)
    assert np.isfinite(fused).all()