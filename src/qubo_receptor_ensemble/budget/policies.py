"""E2 allocation policies and the budget simulator.

The budget is counted in *docking jobs* = (ligand, receptor) cells.  Every
policy is a function of information that is legal at decision time:

- the **training fold** (labels allowed) fixes the receptor order and any
  label-dependent parameter;
- the **revealed docking scores** of the library (labels never);
- ligand metadata (scaffold).

The full/``hidden`` matrix is used only by the oracle policies, which are
explicitly upper bounds (``s5_*``), and by the evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from ..headroom.fusion import FusionScorer
from ..headroom.metrics_fast import metric_value
from ..headroom.subsets import utility
from .fusion_ragged import RaggedFusion, fuse_ragged


class BudgetError(ValueError):
    """Raised when an allocation violates the budget contract."""


@dataclass(frozen=True)
class PolicySpec:
    """One frozen allocation policy."""

    name: str
    top_fraction: float | None = None
    oracle: str | None = None  # "score" (no labels) | "metric" (upper bound)

    def as_dict(self) -> dict[str, object]:
        return {"name": self.name, "top_fraction": self.top_fraction, "oracle": self.oracle}


POLICY_NAMES: tuple[str, ...] = (
    "s1_width",
    "s2_uniform",
    "s3_top10",
    "s3_top25",
    "s3_top50",
    "s4_scaffold25",
    "s4_scaffold50",
    "s5_score_oracle",
    "s5_metric_oracle",
)


def policy_family(names: Sequence[str] | None = None) -> tuple[PolicySpec, ...]:
    """Frozen policy family in the pre-registered order."""
    selected = tuple(POLICY_NAMES if names is None else names)
    unknown = [name for name in selected if name not in POLICY_NAMES]
    if unknown:
        raise BudgetError(f"unknown policies: {unknown}")
    specs: list[PolicySpec] = []
    for name in selected:
        if name.startswith("s3_top"):
            specs.append(PolicySpec(name=name, top_fraction=int(name.replace("s3_top", "")) / 100.0))
        elif name.startswith("s4_scaffold"):
            specs.append(PolicySpec(name=name, top_fraction=int(name.replace("s4_scaffold", "")) / 100.0))
        elif name == "s5_score_oracle":
            specs.append(PolicySpec(name=name, oracle="score"))
        elif name == "s5_metric_oracle":
            specs.append(PolicySpec(name=name, oracle="metric"))
        else:
            specs.append(PolicySpec(name=name))
    return tuple(specs)


class BudgetSimulator:
    """Reveal docking cells one stage at a time and fuse each ligand's own cells."""

    def __init__(
        self,
        hidden_scores: np.ndarray,
        ragged: RaggedFusion,
        budget: int,
        *,
        metric: str = "pr_auc",
        alpha: float = 20.0,
        eval_index: np.ndarray | None = None,
        eval_labels: np.ndarray | None = None,
    ) -> None:
        self.hidden = np.asarray(hidden_scores, dtype=np.float64)
        if self.hidden.ndim != 2:
            raise BudgetError("hidden_scores must be 2-D")
        self.ragged = ragged
        self.budget = int(budget)
        if self.budget < 0:
            raise BudgetError("budget must be non-negative")
        self.metric = metric
        self.alpha = alpha
        self.mask = np.zeros(self.hidden.shape, dtype=bool)
        self.cells: list[tuple[int, int]] = []
        self.eval_index = (
            np.arange(self.hidden.shape[0], dtype=np.int64) if eval_index is None else np.asarray(eval_index, dtype=np.int64)
        )
        self.eval_labels = eval_labels

    def evaluation_vector(self, mask: np.ndarray | None = None) -> np.ndarray:
        """Fused scores restricted to the evaluation (held-out) ligands."""
        source = self.mask if mask is None else mask
        return fuse_ragged(self.ragged, source)[self.eval_index]

    # -- state -----------------------------------------------------------
    @property
    def jobs_used(self) -> int:
        return len(self.cells)

    @property
    def remaining(self) -> int:
        return self.budget - self.jobs_used

    @property
    def n_ligands(self) -> int:
        return int(self.hidden.shape[0])

    @property
    def n_receptors(self) -> int:
        return int(self.hidden.shape[1])

    def observed_scores(self) -> np.ndarray:
        """Revealed scores with NaN for unobserved cells (policy-legal view)."""
        return np.where(self.mask, self.hidden, np.nan)

    def fused(self) -> np.ndarray:
        return fuse_ragged(self.ragged, self.mask)

    def reveal(self, cells: Iterable[tuple[int, int]]) -> int:
        """Reveal new cells, respecting the remaining budget; returns jobs spent."""
        spent = 0
        for ligand, receptor in cells:
            if self.remaining <= 0:
                break
            if not (0 <= ligand < self.n_ligands and 0 <= receptor < self.n_receptors):
                raise BudgetError(f"cell ({ligand}, {receptor}) outside the matrix")
            if self.mask[ligand, receptor]:
                continue
            self.mask[ligand, receptor] = True
            self.cells.append((int(ligand), int(receptor)))
            spent += 1
        return spent

    def per_ligand_depth(self) -> np.ndarray:
        return self.mask.sum(axis=1)

    def summary(self) -> dict[str, object]:
        depth = self.per_ligand_depth()
        docked = depth > 0
        eval_depth = depth[self.eval_index]
        eval_docked = eval_depth > 0
        return {
            "jobs_used": self.jobs_used,
            "n_docked": int(docked.sum()),
            "coverage": float(docked.mean()) if depth.size else 0.0,
            "mean_depth": float(depth[docked].mean()) if docked.any() else 0.0,
            "max_depth": int(depth.max()) if depth.size else 0,
            "receptor_usage": int((self.mask.sum(axis=0) > 0).sum()),
            "eval_n": int(eval_depth.size),
            "eval_coverage": float(eval_docked.mean()) if eval_depth.size else 0.0,
            "eval_mean_depth": float(eval_depth[eval_docked].mean()) if eval_docked.any() else 0.0,
        }


def receptor_order(
    scorer: FusionScorer,
    train_labels: np.ndarray,
    metric: str = "pr_auc",
    alpha: float = 20.0,
) -> tuple[int, ...]:
    """Rank receptors by single-receptor train utility (best first)."""
    values = [
        (
            utility(scorer.score_columns((column,)), train_labels, metric, alpha),
            column,
        )
        for column in range(scorer.n_receptors)
    ]
    return tuple(column for _, column in sorted(values, key=lambda item: (-item[0], item[1])))


def _next_receptors(order: Sequence[int], observed: np.ndarray) -> list[int]:
    row = np.asarray(observed)
    return [int(column) for column in order if not row[column]]


def run_policy(
    spec: PolicySpec,
    simulator: BudgetSimulator,
    *,
    order: Sequence[int],
    scaffolds: Sequence[str],
    labels: np.ndarray | None = None,
    greedy_max_steps: int | None = None,
) -> dict[str, object]:
    """Execute one policy against the simulator; returns allocation diagnostics."""
    if not order:
        raise BudgetError("receptor order must not be empty")
    if spec.oracle == "metric":
        return _run_metric_oracle(spec, simulator, labels=labels, greedy_max_steps=greedy_max_steps)
    if spec.oracle == "score":
        return _run_score_oracle(spec, simulator, order=order)
    if spec.name == "s1_width":
        return _run_s1(simulator, order=order)
    if spec.name == "s2_uniform":
        return _run_s2(simulator, order=order)
    if spec.name.startswith("s3_top"):
        return _run_s3(spec, simulator, order=order)
    if spec.name.startswith("s4_scaffold"):
        return _run_s4(spec, simulator, order=order, scaffolds=scaffolds)
    raise BudgetError(f"no runner for policy {spec.name}")


# -- policies ------------------------------------------------------------
def _run_s1(simulator: BudgetSimulator, *, order: Sequence[int]) -> dict[str, object]:
    """Width limit: one receptor (the best) for as many ligands as the budget allows."""
    n = min(simulator.n_ligands, simulator.remaining)
    primary = int(order[0])
    simulator.reveal((ligand, primary) for ligand in range(n))
    return {"stages": ["width"]}


def _run_s2(simulator: BudgetSimulator, *, order: Sequence[int]) -> dict[str, object]:
    """Uniform depth: floor(B / N) receptors for every ligand; slack stays unspent."""
    depth = simulator.remaining // simulator.n_ligands
    if depth <= 0:
        return _run_s1(simulator, order=order)
    columns = list(order[: min(depth, simulator.n_receptors)])
    simulator.reveal((ligand, column) for ligand in range(simulator.n_ligands) for column in columns)
    return {"stages": ["uniform"], "planned_depth": depth}


def _run_s3(spec: PolicySpec, simulator: BudgetSimulator, *, order: Sequence[int]) -> dict[str, object]:
    """Two stage: 1 receptor for all ligands, then extra receptors on the top x%."""
    primary = int(order[0])
    simulator.reveal((ligand, primary) for ligand in range(simulator.n_ligands))
    stage_one = simulator.fused()
    fraction = float(spec.top_fraction or 0.25)
    n_top = max(1, int(np.ceil(simulator.n_ligands * fraction)))
    ranked = np.argsort(-stage_one, kind="stable")[:n_top]  # stage-1 score only (labels never used)
    rounds = 0
    while simulator.remaining > 0:
        extra = _next_receptors(order, simulator.mask[ranked[0]] if ranked.size else np.zeros(simulator.n_receptors, dtype=bool))
        if not extra:
            break
        column = int(extra[0])
        simulator.reveal((int(ligand), column) for ligand in ranked)
        rounds += 1
    return {"stages": ["primary_all", "top_up"], "top_fraction": fraction, "top_up_rounds": rounds}


def _run_s4(
    spec: PolicySpec,
    simulator: BudgetSimulator,
    *,
    order: Sequence[int],
    scaffolds: Sequence[str],
) -> dict[str, object]:
    """Scaffold two stage: representatives first, then expand the best groups."""
    primary = int(order[0])
    groups: dict[str, list[int]] = {}
    for index, scaffold in enumerate(scaffolds):
        groups.setdefault(str(scaffold), []).append(index)
    representatives = [members[0] for members in groups.values()]
    simulator.reveal((int(ligand), primary) for ligand in representatives)
    stage_one = simulator.fused()
    fraction = float(spec.top_fraction or 0.25)
    n_top = max(1, int(np.ceil(len(groups) * fraction)))
    scored_groups = sorted(
        groups.values(),
        key=lambda members: (float(stage_one[members[0]]), str(members[0])),
        reverse=True,
    )[:n_top]
    members = [ligand for group in scored_groups for ligand in group]
    rounds = 0
    while simulator.remaining > 0 and members:
        observed_any = simulator.mask[members].any(axis=0)
        extra = _next_receptors(order, observed_any)
        if not extra:
            break
        column = int(extra[0])
        simulator.reveal((int(ligand), column) for ligand in members)
        rounds += 1
    return {
        "stages": ["scaffold_representatives", "expand_top_groups"],
        "top_fraction": fraction,
        "scaffold_count": len(groups),
        "expanded_scaffolds": len(scored_groups),
        "top_up_rounds": rounds,
    }


def _run_score_oracle(spec: PolicySpec, simulator: BudgetSimulator, *, order: Sequence[int]) -> dict[str, object]:
    """Per-ligand score oracle: give every ligand its own best k receptors (no labels)."""
    depth = simulator.remaining // simulator.n_ligands
    if depth <= 0:
        return _run_s1(simulator, order=order)
    columns = max(1, min(depth, simulator.n_receptors))
    # per-ligand oracle: every ligand keeps the `columns` receptors with the best
    # per-receptor term *for that ligand* (uses the hidden matrix, no labels).
    terms = simulator.ragged.terms
    ascending = simulator.ragged.name != "rrf"  # rrf rewards larger terms
    index = np.argsort(terms, axis=1, kind="stable")
    picked = index[:, :columns] if ascending else index[:, -columns:]
    cells = [(ligand, int(column)) for ligand in range(simulator.n_ligands) for column in picked[ligand]]
    simulator.reveal(cells)
    return {"stages": ["score_oracle"], "planned_depth": columns}


def _run_metric_oracle(
    spec: PolicySpec,
    simulator: BudgetSimulator,
    *,
    labels: np.ndarray | None,
    greedy_max_steps: int | None,
) -> dict[str, object]:
    """Cell-greedy oracle that maximises the *evaluation* metric (upper bound)."""
    if labels is None or simulator.eval_labels is None:
        raise BudgetError("s5_metric_oracle needs the evaluation subset and its labels")
    steps = 0
    limit = simulator.budget if greedy_max_steps is None else min(simulator.budget, int(greedy_max_steps))
    while simulator.jobs_used < limit:
        best: tuple[float, int, int] | None = None
        for ligand in range(simulator.n_ligands):
            row = simulator.mask[ligand]
            if row.all():
                continue
            for receptor in range(simulator.n_receptors):
                if row[receptor]:
                    continue
                probe = simulator.mask.copy()
                probe[ligand, receptor] = True
                value = metric_value(
                    simulator.evaluation_vector(probe),
                    simulator.eval_labels,
                    simulator.metric,
                    simulator.alpha,
                )
                if not np.isfinite(value):
                    continue
                key = (float(value), -ligand, -receptor)
                if best is None or key > (best[0], -best[1], -best[2]):
                    best = (float(value), ligand, receptor)
        if best is None:
            break
        simulator.reveal([(best[1], best[2])])
        steps += 1
    return {"stages": ["metric_oracle"], "greedy_steps": steps}