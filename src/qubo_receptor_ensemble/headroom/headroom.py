"""Three headroom definitions per (target, fold, phi, k) cell.

Definitions follow the pre-registered E1 protocol:

``H_raw``
    ``max_{|S|=k} U_test(S) - U_test(S_ref)`` -- the "kill switch" upper bound
    that peeks at the test labels.  If it cannot beat the noise floor nothing
    can.
``H_nested``
    the outer-test fold is split by scaffold into halves A/B; ``S*_A`` is
    selected on A and scored on B (and mirrored), ``S_ref`` is chosen on the
    training fold; the two signed transfers are averaged.  This is the
    claimable statistic used by gate G1.
``H_perm``
    label-permutation inflation correction:
    ``H_perm = H_raw - q95(max_{S in TopM} U_test(S^perm))``.  The raw
    components are reported alongside so the correction stays auditable.

Every subset is enumerated exactly (no approximation, no beam search).  The
evaluation unit is the outer test fold as in the golden V5 protocol
(``scripts/nested_outer_k_evaluation.py``); rows must be pre-sorted by
``ligand_id`` so that the stable ranking tie-break matches
``screening.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from heapq import heappush, heappushpop
from typing import Sequence

import numpy as np

from .bootstrap import cluster_bootstrap, cluster_bootstrap_delta, percentile
from .fusion import FusionScorer, FusionSpec, fit_fusion
from .metrics_fast import METRIC_NAMES, metric_values, positions, ranking_order
from .subsets import columns_mask, count_subsets, iter_combinations, utility

DEFAULT_K_RANGE: tuple[int, ...] = (1, 2, 3, 4, 5, 6)

#: Reference selector chosen on the training fold ("single" wins ties).
REFERENCE_SOURCES: tuple[str, ...] = ("greedy", "single")


class HeadroomError(ValueError):
    """Raised when a headroom request violates its contract."""


@dataclass(frozen=True)
class HeadroomConfig:
    """Frozen pre-registration knobs used by one scan shard."""

    k_list: tuple[int, ...] = DEFAULT_K_RANGE
    metric: str = "pr_auc"
    alpha: float = 20.0
    top_m: int = 2000
    permutations: int = 200
    perm_ks: tuple[int, ...] = (2, 3)
    bootstrap_iterations: int = 2000
    bootstrap_seed: int = 0
    with_train_oracle: bool = True
    max_subsets_per_k: int | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "k_list": list(self.k_list),
            "metric": self.metric,
            "alpha": self.alpha,
            "top_m": self.top_m,
            "permutations": self.permutations,
            "perm_ks": list(self.perm_ks),
            "bootstrap_iterations": self.bootstrap_iterations,
            "bootstrap_seed": self.bootstrap_seed,
            "with_train_oracle": self.with_train_oracle,
            "max_subsets_per_k": self.max_subsets_per_k,
        }


def scaffold_allocation(
    scaffolds: Sequence[str],
    n_buckets: int,
) -> np.ndarray:
    """Deterministic size-balanced scaffold allocation into ``n_buckets``.

    Scaffolds are processed largest-first (ties broken by scaffold string) and
    each is assigned to the currently lightest bucket; ties go to the lowest
    bucket index.  Used both by the A/B nested split and by the inner-CV folds
    of the train-only phi selection.
    """
    if n_buckets < 2:
        raise HeadroomError("n_buckets must be at least 2")
    keys = [str(scaffold) for scaffold in scaffolds]
    if not keys:
        raise HeadroomError("cannot allocate an empty panel")
    groups: dict[str, list[int]] = {}
    for index, key in enumerate(keys):
        groups.setdefault(key, []).append(index)
    counts = [0] * n_buckets
    assignment: dict[str, int] = {}
    for key, rows in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        bucket = min(range(n_buckets), key=lambda index: (counts[index], index))
        assignment[key] = bucket
        counts[bucket] += len(rows)
    return np.asarray([assignment[key] for key in keys], dtype=np.int64)


def scaffold_ab_split(
    scaffolds: Sequence[str],
    labels: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic, load-balanced scaffold split of one ligand panel.

    Scaffolds are processed largest-first (ties broken by scaffold string) and
    assigned to the currently lighter half; ties go to A.  If one half ends up
    without a label class the split is repaired by moving the smallest
    informative scaffold from the richer side.  Returns local row indices.
    """
    keys = [str(scaffold) for scaffold in scaffolds]
    if not keys:
        raise HeadroomError("cannot split an empty panel")
    groups: dict[str, list[int]] = {}
    for index, key in enumerate(keys):
        groups.setdefault(key, []).append(index)
    ordered = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    active = None if labels is None else np.asarray(labels) > 0.5
    assignment: dict[str, int] = {}
    counts = [0, 0]
    active_counts = [0, 0]
    for key, rows in ordered:
        side = 0 if counts[0] <= counts[1] else 1
        assignment[key] = side
        counts[side] += len(rows)
        if active is not None:
            active_counts[side] += int(active[rows].sum())
    if active is not None:
        for _ in range(4):
            changed = False
            for side in (0, 1):
                if active_counts[side] > 0:
                    continue
                donor = 1 - side
                candidates = [
                    (len(rows), key)
                    for key, rows in ordered
                    if assignment[key] == donor and int(active[rows].sum()) > 0
                ]
                if not candidates:
                    break
                _, key = min(candidates)
                rows = groups[key]
                assignment[key] = side
                counts[donor] -= len(rows)
                counts[side] += len(rows)
                active_counts[donor] -= int(active[rows].sum())
                active_counts[side] += int(active[rows].sum())
                changed = True
            if not changed:
                break
    mask_a = np.asarray([assignment[key] == 0 for key in keys], dtype=bool)
    index = np.arange(len(keys), dtype=np.int64)
    return index[mask_a], index[~mask_a]


def greedy_columns(
    scorer: FusionScorer,
    labels: np.ndarray,
    k: int,
    metric: str,
    alpha: float,
    ligand_rank: np.ndarray | None = None,
) -> tuple[int, ...]:
    """Forward selection maximizing the training-fold fusion utility (V5 greedy)."""
    selected: tuple[int, ...] = ()
    while len(selected) < k:
        best_value = float("-inf")
        best_subset: tuple[int, ...] = ()
        for column in range(scorer.n_receptors):
            if column in selected:
                continue
            candidate = tuple(sorted((*selected, column)))
            value = utility(scorer.score_columns(candidate), labels, metric, alpha, ligand_rank)
            if value > best_value or (value == best_value and candidate < best_subset):
                best_value = value
                best_subset = candidate
        if not best_subset:
            raise HeadroomError("greedy selection ran out of receptor candidates")
        selected = best_subset
    return selected


def best_single_column(
    scorer: FusionScorer,
    labels: np.ndarray,
    metric: str,
    alpha: float,
    ligand_rank: np.ndarray | None = None,
) -> tuple[int, ...]:
    """Best single receptor on the given label set (train-fold reference)."""
    best_value = float("-inf")
    best_column = 0
    for column in range(scorer.n_receptors):
        value = utility(scorer.score_columns((column,)), labels, metric, alpha, ligand_rank)
        if value > best_value:
            best_value = value
            best_column = column
    return (best_column,)


def select_reference(
    scorer_train: FusionScorer,
    train_labels: np.ndarray,
    k: int,
    config: HeadroomConfig,
    ligand_rank_train: np.ndarray | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...], str]:
    """``S_ref`` = the better of {greedy, single} on the training fold."""
    single = best_single_column(scorer_train, train_labels, config.metric, config.alpha, ligand_rank_train)
    greedy = greedy_columns(scorer_train, train_labels, k, config.metric, config.alpha, ligand_rank_train)
    single_value = utility(
        scorer_train.score_columns(single), train_labels, config.metric, config.alpha, ligand_rank_train
    )
    greedy_value = utility(
        scorer_train.score_columns(greedy), train_labels, config.metric, config.alpha, ligand_rank_train
    )
    if single_value >= greedy_value:
        return single, greedy, "single"
    return single, greedy, "greedy"


def _permutation_max_metric(
    labels_matrix: np.ndarray,
    orders: np.ndarray,
    metric: str,
    alpha: float,
    active_total: int,
    chunk: int = 16,
) -> np.ndarray:
    """``max_S U(S^perm)`` for a batch of permuted label vectors.

    ``labels_matrix`` is ``(n_perm, n_ligands)`` boolean; ``orders`` is
    ``(n_subsets, n_ligands)`` ranking orders of the fixed fused scores.
    """
    n_perm, n_ligands = labels_matrix.shape
    values = np.full(n_perm, -np.inf, dtype=np.float64)
    if metric == "pr_auc":
        rank_positions = positions(n_ligands)
        for start in range(0, orders.shape[0], chunk):
            block = orders[start:start + chunk]
            ordered = labels_matrix[:, block]
            cumulative = np.cumsum(ordered, axis=2)
            scores = (ordered * cumulative / rank_positions).sum(axis=2) / active_total
            values = np.maximum(values, scores.max(axis=1))
        return values
    if metric == "bedroc20":
        weights = np.exp(-alpha * positions(n_ligands) / n_ligands)
        random_expected = active_total * float(weights.mean())
        best = float(weights[:active_total].sum()) / random_expected
        worst = float(weights[n_ligands - active_total:].sum()) / random_expected
        span = best - worst
        if span == 0:
            return np.full(n_perm, float("nan"))
        for start in range(0, orders.shape[0], chunk):
            block = orders[start:start + chunk]
            rie = (labels_matrix[:, block] * weights).sum(axis=2) / random_expected
            values = np.maximum(values, ((rie - worst) / span).max(axis=1))
        return values
    raise HeadroomError(f"permutation null is not implemented for metric {metric}")


def permutation_null(
    scorer: FusionScorer,
    top_masks: Sequence[int],
    labels: np.ndarray,
    config: HeadroomConfig,
    seed: int,
    ligand_rank: np.ndarray | None = None,
) -> dict[str, object]:
    """Quantiles of ``max_S U_test(S^perm)`` over the stored top-M subsets."""
    labels_array = np.asarray(labels)
    active_total = int((labels_array > 0.5).sum())
    if not top_masks or active_total in (0, labels_array.size):
        return {
            "q95": float("nan"),
            "q50": float("nan"),
            "mean": float("nan"),
            "max": float("nan"),
            "n_permutations": 0,
            "n_subsets": int(len(top_masks)),
        }
    orders = np.empty((len(top_masks), labels_array.size), dtype=np.int64)
    for index, mask in enumerate(top_masks):
        columns = tuple(column for column in range(scorer.n_receptors) if mask >> column & 1)
        orders[index] = ranking_order(scorer.score_columns(columns), ligand_rank)
    rng = np.random.default_rng(seed)
    n_perm = max(1, int(config.permutations))
    labels_matrix = np.stack(
        [labels_array[rng.permutation(labels_array.size)] > 0.5 for _ in range(n_perm)]
    )
    maxima = _permutation_max_metric(
        labels_matrix, orders, config.metric, config.alpha, active_total
    )
    finite = maxima[np.isfinite(maxima)]
    if finite.size == 0:
        return {
            "q95": float("nan"),
            "q50": float("nan"),
            "mean": float("nan"),
            "max": float("nan"),
            "n_permutations": n_perm,
            "n_subsets": int(len(top_masks)),
        }
    return {
        "q95": percentile(finite, 0.95),
        "q50": percentile(finite, 0.50),
        "mean": float(finite.mean()),
        "max": float(finite.max()),
        "n_permutations": int(finite.size),
        "n_subsets": int(len(top_masks)),
    }


def _has_both_classes(labels: np.ndarray) -> bool:
    """A metric needs at least one active and one non-active row."""
    values = np.asarray(labels)
    active = int((values > 0.5).sum())
    return 0 < active < values.size


def scan_shard(
    *,
    target_id: str,
    fold: int,
    spec: FusionSpec,
    train_scores: np.ndarray,
    train_labels: np.ndarray,
    train_scaffolds: Sequence[str],
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    test_scaffolds: Sequence[str],
    config: HeadroomConfig,
    ligand_rank_test: np.ndarray | None = None,
    ligand_rank_train: np.ndarray | None = None,
    seed: int = 0,
) -> dict[str, object]:
    """Full battery of one (target, fold, phi) shard over ``config.k_list``."""
    train_scores = np.asarray(train_scores, dtype=np.float64)
    test_scores = np.asarray(test_scores, dtype=np.float64)
    train_labels = np.asarray(train_labels)
    test_labels = np.asarray(test_labels)
    frozen = fit_fusion(spec, train_scores)
    scorer_train = FusionScorer(frozen, train_scores)
    scorer_test = FusionScorer(frozen, test_scores)
    a_index, b_index = scaffold_ab_split(test_scaffolds, test_labels)
    test_clusters = [str(value) for value in test_scaffolds]
    cells: list[dict[str, object]] = []

    for k in config.k_list:
        single, greedy, ref_source = select_reference(
            scorer_train, train_labels, k, config, ligand_rank_train
        )
        ref_columns = single if ref_source == "single" else greedy
        fused_ref = scorer_test.score_columns(ref_columns)
        fused_greedy = scorer_test.score_columns(greedy)
        fused_single = scorer_test.score_columns(single)
        test_scorable = _has_both_classes(test_labels)
        a_scorable = _has_both_classes(test_labels[a_index])
        b_scorable = _has_both_classes(test_labels[b_index])
        u_test_ref = (
            utility(fused_ref, test_labels, config.metric, config.alpha, ligand_rank_test)
            if test_scorable
            else float("nan")
        )
        u_a_ref = (
            utility(
                fused_ref[a_index], test_labels[a_index], config.metric, config.alpha,
                None if ligand_rank_test is None else ligand_rank_test[a_index],
            )
            if a_scorable
            else float("nan")
        )
        u_b_ref = (
            utility(
                fused_ref[b_index], test_labels[b_index], config.metric, config.alpha,
                None if ligand_rank_test is None else ligand_rank_test[b_index],
            )
            if b_scorable
            else float("nan")
        )

        best_test = (float("-inf"), ())
        best_a = (float("-inf"), ())
        best_b = (float("-inf"), ())
        best_train = (float("-inf"), ())
        n_subsets = int(math.comb(scorer_test.n_receptors, k))
        if config.max_subsets_per_k is not None and n_subsets > config.max_subsets_per_k:
            raise HeadroomError(
                f"C({scorer_test.n_receptors},{k})={n_subsets} exceeds "
                f"max_subsets_per_k={config.max_subsets_per_k}"
            )
        top_heap: list[tuple[float, tuple[int, ...]]] = []
        collect_top = k in set(config.perm_ks)
        for columns in iter_combinations(scorer_test.n_receptors, k):
            fused = scorer_test.score_columns(columns)
            value_test = (
                utility(fused, test_labels, config.metric, config.alpha, ligand_rank_test)
                if test_scorable
                else float("nan")
            )
            if value_test > best_test[0]:
                best_test = (value_test, columns)
            if a_scorable:
                value_a = utility(
                    fused[a_index], test_labels[a_index], config.metric, config.alpha,
                    None if ligand_rank_test is None else ligand_rank_test[a_index],
                )
                if value_a > best_a[0]:
                    best_a = (value_a, columns)
            if b_scorable:
                value_b = utility(
                    fused[b_index], test_labels[b_index], config.metric, config.alpha,
                    None if ligand_rank_test is None else ligand_rank_test[b_index],
                )
                if value_b > best_b[0]:
                    best_b = (value_b, columns)
            if config.with_train_oracle:
                value_train = utility(
                    scorer_train.score_columns(columns), train_labels,
                    config.metric, config.alpha, ligand_rank_train,
                )
                if value_train > best_train[0]:
                    best_train = (value_train, columns)
            if collect_top:
                if len(top_heap) < config.top_m:
                    heappush(top_heap, (value_test, columns))
                elif value_test > top_heap[0][0]:
                    heappushpop(top_heap, (value_test, columns))

        columns_a = best_a[1]
        columns_b = best_b[1]
        fused_a = scorer_test.score_columns(columns_a) if columns_a else None
        fused_b = scorer_test.score_columns(columns_b) if columns_b else None
        u_b_of_a = (
            utility(
                fused_a[b_index], test_labels[b_index], config.metric, config.alpha,
                None if ligand_rank_test is None else ligand_rank_test[b_index],
            )
            if fused_a is not None and b_scorable
            else float("nan")
        )
        u_a_of_b = (
            utility(
                fused_b[a_index], test_labels[a_index], config.metric, config.alpha,
                None if ligand_rank_test is None else ligand_rank_test[a_index],
            )
            if fused_b is not None and a_scorable
            else float("nan")
        )
        h_raw = best_test[0] - u_test_ref if test_scorable else float("nan")
        h_ab = u_b_of_a - u_b_ref if np.isfinite(u_b_of_a) and np.isfinite(u_b_ref) else float("nan")
        h_ba = u_a_of_b - u_a_ref if np.isfinite(u_a_of_b) and np.isfinite(u_a_ref) else float("nan")
        finite_transfers = [value for value in (h_ab, h_ba) if np.isfinite(value)]
        h_nested = float(np.mean(finite_transfers)) if finite_transfers else float("nan")

        noise = cluster_bootstrap_delta(
            scores_a=fused_greedy,
            scores_b=fused_single,
            labels=test_labels,
            clusters=test_clusters,
            metric=config.metric,
            alpha=config.alpha,
            ligand_rank=ligand_rank_test,
            iterations=config.bootstrap_iterations,
            seed=config.bootstrap_seed,
        )

        is_a = np.zeros(test_labels.size, dtype=bool)
        is_a[a_index] = True

        def nested_statistic(rows: np.ndarray, _fused_a=fused_a, _fused_b=fused_b,
                             _fused_ref=fused_ref, _is_a=is_a,
                             _labels=test_labels) -> float:
            a_rows = rows[_is_a[rows]]
            b_rows = rows[~_is_a[rows]]
            if a_rows.size < 2 or b_rows.size < 2:
                return float("nan")
            if not _has_both_classes(_labels[a_rows]) or not _has_both_classes(_labels[b_rows]):
                return float("nan")
            rank_a = None if ligand_rank_test is None else ligand_rank_test[a_rows]
            rank_b = None if ligand_rank_test is None else ligand_rank_test[b_rows]
            delta_b = utility(_fused_a[b_rows], _labels[b_rows], config.metric, config.alpha, rank_b) - utility(
                _fused_ref[b_rows], _labels[b_rows], config.metric, config.alpha, rank_b
            )
            delta_a = utility(_fused_b[a_rows], _labels[a_rows], config.metric, config.alpha, rank_a) - utility(
                _fused_ref[a_rows], _labels[a_rows], config.metric, config.alpha, rank_a
            )
            return 0.5 * (delta_a + delta_b)

        nested_boot = cluster_bootstrap(
            nested_statistic,
            test_clusters,
            iterations=config.bootstrap_iterations,
            seed=config.bootstrap_seed + 1,
        )

        if a_scorable and b_scorable:
            oracle_columns = columns_a if best_a[0] >= best_b[0] else columns_b
        elif a_scorable:
            oracle_columns = columns_a
        elif b_scorable:
            oracle_columns = columns_b
        else:
            oracle_columns = best_test[1]
        cell = {
            "k": k,
            "n_subsets": n_subsets,
            "u_test_ref": u_test_ref,
            "ref_source": ref_source,
            "ref_columns": list(ref_columns),
            "u_test_greedy": utility(fused_greedy, test_labels, config.metric, config.alpha, ligand_rank_test),
            "greedy_columns": list(greedy),
            "u_test_single": utility(fused_single, test_labels, config.metric, config.alpha, ligand_rank_test),
            "single_columns": list(single),
            "u_test_oracle": best_test[0],
            "oracle_columns": list(best_test[1]),
            "h_raw": h_raw,
            "u_a_ref": u_a_ref,
            "u_b_ref": u_b_ref,
            "u_a_oracle": best_a[0],
            "columns_a": list(columns_a),
            "u_b_of_a": u_b_of_a,
            "u_b_oracle": best_b[0],
            "columns_b": list(columns_b),
            "u_a_of_b": u_a_of_b,
            "h_ab": h_ab,
            "h_ba": h_ba,
            "h_nested": h_nested,
            "u_train_oracle": best_train[0] if config.with_train_oracle else None,
            "columns_train_oracle": list(best_train[1]) if config.with_train_oracle else None,
            "u_test_of_train_selected": (
                utility(scorer_test.score_columns(best_train[1]), test_labels,
                        config.metric, config.alpha, ligand_rank_test)
                if config.with_train_oracle and best_train[1]
                else None
            ),
            "metric_train_selected": (
                metric_values(
                    scorer_test.score_columns(best_train[1]), test_labels,
                    METRIC_NAMES, config.alpha, ligand_rank_test,
                )
                if config.with_train_oracle and best_train[1]
                else None
            ),
            "metric_oracle": metric_values(
                scorer_test.score_columns(oracle_columns if oracle_columns else best_test[1]),
                test_labels,
                METRIC_NAMES,
                config.alpha,
                ligand_rank_test,
            ),
            "metric_ref": metric_values(fused_ref, test_labels, METRIC_NAMES, config.alpha, ligand_rank_test),
            "metric_greedy": metric_values(fused_greedy, test_labels, METRIC_NAMES, config.alpha, ligand_rank_test),
            "metric_single": metric_values(fused_single, test_labels, METRIC_NAMES, config.alpha, ligand_rank_test),
            "noise_floor": noise,
            "nested_bootstrap": nested_boot,
            "h_nested_lower95": nested_boot["ci95_low"],
            "top_masks": (
                [columns_mask(entry[1], scorer_test.n_receptors) for entry in sorted(top_heap, reverse=True)]
                if collect_top
                else None
            ),
        }
        cell["ratio"] = (
            h_nested / noise["se"] if np.isfinite(noise["se"]) and noise["se"] > 0 else float("nan")
        )
        cells.append(cell)

    return {
        "schema": "e1_shard_v1",
        "target_id": target_id,
        "fold": int(fold),
        "phi": spec.name,
        "fusion": frozen.as_dict(),
        "config": config.as_dict(),
        "a_size": int(a_index.size),
        "b_size": int(b_index.size),
        "a_active": int((test_labels[a_index] > 0.5).sum()),
        "b_active": int((test_labels[b_index] > 0.5).sum()),
        "test_ligand_count": int(test_labels.size),
        "train_ligand_count": int(train_labels.size),
        "receptor_count": int(scorer_test.n_receptors),
        "test_scaffold_count": int(len(set(test_clusters))),
        "cells": cells,
    }


def _fold_battery(
    scores: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    scaffolds: Sequence[str],
    spec: FusionSpec,
    config: HeadroomConfig,
    seed: int = 0,
) -> dict[int, dict[str, object]]:
    results: dict[int, dict[str, object]] = {}
    for fold in sorted({int(value) for value in folds}):
        train_index = np.flatnonzero(folds != fold)
        test_index = np.flatnonzero(folds == fold)
        results[fold] = scan_shard(
            target_id="fixture",
            fold=fold,
            spec=spec,
            train_scores=scores[train_index],
            train_labels=labels[train_index],
            train_scaffolds=[scaffolds[index] for index in train_index],
            test_scores=scores[test_index],
            test_labels=labels[test_index],
            test_scaffolds=[scaffolds[index] for index in test_index],
            config=config,
            seed=seed,
        )
    return results


def headroom_raw(
    scores: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    fusion: FusionSpec,
    k_range: Sequence[int] | None = None,
    scaffolds: Sequence[str] | None = None,
    config: HeadroomConfig | None = None,
) -> dict[int, dict[int, dict[str, object]]]:
    """Per-fold ``H_raw`` (and its inputs) for every k in ``k_range``."""
    settings = config or HeadroomConfig(k_list=tuple(k_range or DEFAULT_K_RANGE))
    if k_range is not None:
        settings = HeadroomConfig(**{**settings.as_dict(), "k_list": tuple(k_range)})
    if scaffolds is None:
        scaffolds = ["ligand" for _ in range(len(scores))]
    battery = _fold_battery(scores, labels, folds, scaffolds, fusion, settings)
    return {
        fold: {int(cell["k"]): cell for cell in payload["cells"]}
        for fold, payload in battery.items()
    }


def headroom_nested(
    scores: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    fusion: FusionSpec,
    k_range: Sequence[int] | None = None,
    split_key: str = "scaffold",
    scaffolds: Sequence[str] | None = None,
    config: HeadroomConfig | None = None,
) -> dict[int, dict[int, dict[str, object]]]:
    """Per-fold ``H_nested`` for every k in ``k_range``."""
    if split_key != "scaffold":
        raise HeadroomError("E1 only pre-registers the scaffold A/B split")
    return headroom_raw(scores, labels, folds, fusion, k_range, scaffolds=scaffolds, config=config)


def headroom_perm(
    scores: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    fusion: FusionSpec,
    k_range: Sequence[int] | None = None,
    top_m: int = 2000,
    n_perm: int = 200,
    scaffolds: Sequence[str] | None = None,
    config: HeadroomConfig | None = None,
) -> dict[int, dict[int, dict[str, object]]]:
    """Per-fold ``H_perm`` components for every k in ``k_range``."""
    settings = config or HeadroomConfig(k_list=tuple(k_range or DEFAULT_K_RANGE))
    settings = HeadroomConfig(
        **{
            **settings.as_dict(),
            "k_list": tuple(k_range or settings.k_list),
            "top_m": int(top_m),
            "permutations": int(n_perm),
        }
    )
    if scaffolds is None:
        scaffolds = ["ligand" for _ in range(len(scores))]
    battery = _fold_battery(scores, labels, folds, scaffolds, fusion, settings)
    output: dict[int, dict[int, dict[str, object]]] = {}
    for fold, payload in battery.items():
        fold_result: dict[int, dict[str, object]] = {}
        for cell in payload["cells"]:
            k = int(cell["k"])
            top_masks = cell.get("top_masks") or []
            if not top_masks:
                fold_result[k] = {"h_perm": None, "null": None}
                continue
            test_index = np.flatnonzero(folds == fold)
            frozen = fit_fusion(fusion, scores[np.flatnonzero(folds != fold)])
            scorer = FusionScorer(frozen, scores[test_index])
            null = permutation_null(
                scorer, list(top_masks), labels[test_index], settings, seed=fold
            )
            fold_result[k] = {
                "h_perm": float(cell["h_raw"]) - float(null["q95"]),
                "h_raw": float(cell["h_raw"]),
                "null": null,
            }
        output[fold] = fold_result
    return output