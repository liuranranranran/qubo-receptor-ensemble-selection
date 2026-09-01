"""Offline masked active-docking replay with an external score oracle."""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ..screening import scalar_metrics
from .acquisition import AcquisitionConfig, PosteriorAcquisitionEvaluator
from .config import validate_active_docking_config
from .predictor import (
    BayesianResidualPredictor,
    NearestReceptorPredictor,
    ObservedScoreMeanPredictor,
    PredictorConfig,
    ScorePredictor,
)
from .qubo import BatchConstraints, build_batch_qubo
from .solvers import solve_batch_qubo
from .state import PartialObservationState, Task
from .warm_start import WarmStartConfig, plan_warm_start


@dataclass(frozen=True)
class ReplayStrategyResult:
    name: str
    task_sequence: tuple[Task, ...]
    rounds: tuple[dict[str, object], ...]
    evaluation: dict[str, object]
    final_state: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "task_sequence": [list(task) for task in self.task_sequence],
            "rounds": list(self.rounds),
            "evaluation": self.evaluation,
            "final_state": self.final_state,
        }


@dataclass(frozen=True)
class ReplayResult:
    strategies: tuple[ReplayStrategyResult, ...]
    metadata: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "metadata": self.metadata,
            "strategies": [strategy.to_dict() for strategy in self.strategies],
        }


def _stable_int(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{seed}|{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _task_costs(tasks: Sequence[Task], config: Mapping[str, object]) -> dict[Task, float]:
    raw = config.get("task_cost", 1.0)
    costs: dict[Task, float] = {}
    if isinstance(raw, Mapping):
        for task in tasks:
            key = f"{task[0]}||{task[1]}"
            costs[task] = float(raw.get(key, raw.get(task[1], 1.0)))
    else:
        costs = {task: float(raw) for task in tasks}
    return costs


def _runtime_config(config: Mapping[str, object]) -> dict[str, object]:
    value = copy.deepcopy(dict(config))
    value.setdefault("schema_version", "1.0")
    value.setdefault("workflow", "masked_active_docking_replay")
    value.setdefault("warm_start", {})
    value.setdefault("predictor", {})
    value.setdefault("acquisition", {})
    value.setdefault("budget", {"total_cost": 10.0, "batch_cost": 2.0})
    value.setdefault("constraints", {})
    value.setdefault("candidate_cap", 64)
    value.setdefault("evaluation", {"metrics": ["bedroc20", "pr_auc", "ef1"]})
    value.setdefault("strategies", ["value_greedy"])
    value["warm_start"].setdefault("baseline_receptor", value["predictor"].get("baseline_receptor", ""))
    value["predictor"].setdefault("baseline_receptor", value["warm_start"].get("baseline_receptor", ""))
    value["predictor"].setdefault("posterior_samples", value["acquisition"].get("monte_carlo_samples", 64))
    value["acquisition"].setdefault("top_q", 10)
    value["acquisition"].setdefault("monte_carlo_samples", value["predictor"].get("posterior_samples", 64))
    value["acquisition"].setdefault("risk_lambda", 0.0)
    value["budget"].setdefault("total_cost", 10.0)
    value["budget"].setdefault("batch_cost", value["budget"]["total_cost"])
    value["constraints"].setdefault("max_per_ligand", 1)
    value["constraints"].setdefault("max_per_receptor", None)
    value["constraints"].setdefault("max_per_scaffold", None)
    validate_active_docking_config(value)
    return value


def _make_predictor(config: Mapping[str, object], state: PartialObservationState) -> ScorePredictor:
    predictor_config = config.get("predictor", {})
    assert isinstance(predictor_config, Mapping)
    model = str(predictor_config.get("model", "bayesian_residual"))
    common = PredictorConfig(
        baseline_receptor=str(predictor_config.get("baseline_receptor", "")),
        prior_precision=float(predictor_config.get("prior_precision", 1.0)),
        noise_variance=float(predictor_config.get("noise_variance", 1.0)),
        posterior_samples=int(predictor_config.get("posterior_samples", 64)),
        random_seed=int(config.get("random_seed", 0)),
    )
    if model == "bayesian_residual":
        predictor: ScorePredictor = BayesianResidualPredictor(common)
    elif model == "nearest_receptor":
        predictor = NearestReceptorPredictor()
    elif model in {"observed_mean", "residual_mean"}:
        predictor = ObservedScoreMeanPredictor()
    else:
        raise ValueError(f"unsupported active-docking predictor: {model}")
    return predictor.fit(state, training_data=[])


def _candidate_pool(
    state: PartialObservationState,
    predictor: ScorePredictor,
    config: Mapping[str, object],
) -> tuple[Task, ...]:
    acquisition_config = config.get("acquisition", {})
    assert isinstance(acquisition_config, Mapping)
    evaluator = PosteriorAcquisitionEvaluator(
        state,
        predictor,
        AcquisitionConfig(
            top_q=int(acquisition_config.get("top_q", 10)),
            monte_carlo_samples=int(acquisition_config.get("monte_carlo_samples", 64)),
            risk_lambda=float(acquisition_config.get("risk_lambda", 0.0)),
            utility_mode=str(acquisition_config.get("utility_mode", "ranking_score")),
            random_seed=int(config.get("random_seed", 0)),
        ),
    )
    all_tasks = state.unfinished_tasks()
    values = evaluator.all_task_values(all_tasks)
    predictions = evaluator.predictions
    requested_cap = int(config.get("candidate_cap", len(all_tasks)))
    mandatory: set[Task] = set()
    scaffold_seen: set[str] = set()
    for task in sorted(all_tasks, key=lambda item: (-values[item] / state.cost_for(item), item)):
        ligand = next(row for row in state.ligand_manifest if str(row["ligand_id"]) == task[0])
        scaffold = str(ligand.get("scaffold", ligand.get("scaffold_smiles", "__unknown__")))
        if scaffold not in scaffold_seen:
            mandatory.add(task)
            scaffold_seen.add(scaffold)
    cluster_seen: set[str] = set()
    receptors = {str(row["receptor_id"]): row for row in state.receptor_manifest}
    for task in sorted(all_tasks, key=lambda item: (-predictions[item].variance, item)):
        cluster = str(receptors[task[1]].get("cluster", receptors[task[1]].get("receptor_cluster", "__unknown__")))
        if cluster not in cluster_seen:
            mandatory.add(task)
            cluster_seen.add(cluster)
    ordered = sorted(all_tasks, key=lambda item: (-values[item] / state.cost_for(item), -predictions[item].variance, item))
    pool = tuple(sorted(set(ordered[:requested_cap]) | mandatory))
    return pool


def _candidate_pruning_report(
    state: PartialObservationState, pool: Sequence[Task], config: Mapping[str, object]
) -> dict[str, object]:
    return {
        "all_unfinished_task_count": len(state.unfinished_tasks()),
        "candidate_task_count": len(pool),
        "candidate_cap": int(config.get("candidate_cap", len(pool))),
        "rule": "fixed value-per-cost plus uncertainty and scaffold/cluster representatives",
        "attribution_boundary": "candidate pruning is a classical preprocessing component, not a solver or quantum gain",
    }


def _constraints_for(
    available: Sequence[Task],
    state: PartialObservationState,
    config: Mapping[str, object],
    budget: float,
) -> BatchConstraints:
    raw = config.get("constraints", {})
    assert isinstance(raw, Mapping)
    activation_raw = raw.get("receptor_activation_cost")
    activated_receptors = {task[1] for task in state.observed_scores}
    if isinstance(activation_raw, Mapping):
        activation_cost = {
            str(receptor_id): 0.0 if str(receptor_id) in activated_receptors else float(cost)
            for receptor_id, cost in activation_raw.items()
        }
    elif activation_raw is None:
        activation_cost = None
    else:
        activation_cost = {
            str(row["receptor_id"]): 0.0
            if str(row["receptor_id"]) in activated_receptors
            else float(activation_raw)
            for row in state.receptor_manifest
        }
    return BatchConstraints(
        budget=budget,
        task_costs={task: state.cost_for(task) for task in available},
        max_per_ligand=raw.get("max_per_ligand"),
        max_per_receptor=raw.get("max_per_receptor"),
        max_per_scaffold=raw.get("max_per_scaffold"),
        ligand_scaffolds={
            str(row["ligand_id"]): str(row.get("scaffold", row.get("scaffold_smiles", "__unknown__")))
            for row in state.ligand_manifest
        },
        receptor_activation_cost=activation_cost,
        penalty=float(raw.get("penalty", 10.0)),
        cost_unit=raw.get("cost_unit"),
        equal_cost=bool(raw.get("equal_cost", False)),
        coefficient_scale=float(raw.get("coefficient_scale", 1.0)),
    )


def _random_tasks(available: Sequence[Task], constraints: BatchConstraints, seed: int) -> tuple[Task, ...]:
    rng = np.random.default_rng(seed)
    order = list(available)
    rng.shuffle(order)
    selected: list[Task] = []
    for task in order:
        proposed = tuple(sorted((*selected, task)))
        if constraints_for_tasks(proposed, constraints):
            selected.append(task)
    return tuple(sorted(selected))


def constraints_for_tasks(tasks: Sequence[Task], constraints: BatchConstraints) -> bool:
    from .qubo import BatchQUBO

    # Reuse the same validation rules through a tiny valid QUBO shell.
    return _direct_feasible(tasks, constraints)


def _direct_feasible(tasks: Sequence[Task], constraints: BatchConstraints) -> bool:
    selected = tuple(tasks)
    total = sum(float(constraints.task_costs[task]) for task in selected)
    receptors = {task[1] for task in selected}
    total += sum(constraints.activation_cost(receptor) for receptor in receptors)
    if total > constraints.budget + 1e-9:
        return False
    if constraints.max_per_ligand is not None and any(sum(task[0] == ligand for task in selected) > constraints.max_per_ligand for ligand in {task[0] for task in selected}):
        return False
    if constraints.max_per_receptor is not None and any(sum(task[1] == receptor for task in selected) > constraints.max_per_receptor for receptor in {task[1] for task in selected}):
        return False
    if constraints.max_per_scaffold is not None and any(sum(constraints.ligand_scaffolds.get(task[0], "__unknown__") == scaffold for task in selected) > constraints.max_per_scaffold for scaffold in {constraints.ligand_scaffolds.get(task[0], "__unknown__") for task in selected}):
        return False
    return True


def _round_robin_tasks(available: Sequence[Task], state: PartialObservationState, constraints: BatchConstraints) -> tuple[Task, ...]:
    receptor_order = {str(row["receptor_id"]): index for index, row in enumerate(state.receptor_manifest)}
    order = sorted(available, key=lambda task: (receptor_order[task[1]], task[0], task[1]))
    selected: list[Task] = []
    for task in order:
        proposed = tuple(sorted((*selected, task)))
        if _direct_feasible(proposed, constraints):
            selected.append(task)
    return tuple(sorted(selected))


def _final_evaluation(state: PartialObservationState, predictor: ScorePredictor, labels: Mapping[str, str], metrics: Sequence[str]) -> dict[str, object]:
    predictions = predictor.predict(state.unfinished_tasks())
    scores_by_ligand: dict[str, list[float]] = {}
    for (ligand_id, _), score in state.observed_scores.items():
        scores_by_ligand.setdefault(ligand_id, []).append(float(score))
    for task, prediction in predictions.items():
        scores_by_ligand.setdefault(task[0], []).append(prediction.mean)
    if set(labels) != set(scores_by_ligand):
        raise ValueError("hidden_labels must cover exactly the ligand manifest")
    ranking_rows = []
    for ligand_id in sorted(scores_by_ligand):
        label = str(labels[ligand_id])
        if label not in {"active", "decoy", "inactive"}:
            raise ValueError("hidden_labels must contain active, decoy or inactive")
        ranking_rows.append({
            "ligand_id": ligand_id,
            "label": label,
            "binary_label": int(label == "active"),
            "ranking_score": -float(np.mean(scores_by_ligand[ligand_id])),
        })
    ranked = sorted(ranking_rows, key=lambda row: (-row["ranking_score"], row["ligand_id"]))
    computed = scalar_metrics(ranked, [0.01, 0.05], 20.0)
    selected_metrics: dict[str, object] = {
        "hidden_labels_used_for_evaluation": True,
        "score_fusion": "mean_visible_or_posterior_mean",
        "ligand_count": len(ranked),
    }
    for metric in metrics:
        if metric == "bedroc20":
            selected_metrics[metric] = computed["bedroc_alpha_20"]
        elif metric == "pr_auc":
            selected_metrics[metric] = computed["pr_auc_average_precision"]
        elif metric == "ef1":
            selected_metrics[metric] = computed["EF1%"]
        elif metric == "ef5":
            selected_metrics[metric] = computed["EF5%"]
        elif metric == "roc_auc":
            selected_metrics[metric] = computed["roc_auc_pairwise"]
    return selected_metrics


def _deterministic_solver_audit(value: Mapping[str, object]) -> dict[str, object]:
    """Remove wall-clock measurements from replay artifacts used for replication."""
    output = {key: item for key, item in value.items() if key != "solver_time_seconds"}
    metadata = output.get("metadata")
    if isinstance(metadata, Mapping):
        output["metadata"] = {
            key: item for key, item in metadata.items() if key != "solver_time_seconds"
        }
    return output


def _state_for_replay(
    scores: Mapping[Task, float],
    ligand_manifest: Sequence[Mapping[str, object]],
    receptor_manifest: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
) -> PartialObservationState:
    tasks = tuple(sorted(scores))
    costs = _task_costs(tasks, config)
    warm = config["warm_start"]
    assert isinstance(warm, Mapping)
    warm_config = WarmStartConfig(
        baseline_receptor=str(warm["baseline_receptor"]),
        cluster_fraction=float(warm.get("cluster_fraction", 0.1)),
        min_ligands_per_cluster=int(warm.get("min_ligands_per_cluster", 1)),
        random_seed=int(warm.get("random_seed", config.get("random_seed", 0))),
    )
    warm_tasks = plan_warm_start(ligand_manifest, receptor_manifest, warm_config)
    if not set(warm_tasks).issubset(scores):
        raise ValueError("warm-start task is absent from full score matrix")
    state = PartialObservationState(
        ligand_manifest=[dict(row) for row in ligand_manifest],
        receptor_manifest=[dict(row) for row in receptor_manifest],
        candidate_tasks=set(tasks),
        task_costs=costs,
        receptor_activation_costs={
            str(row["receptor_id"]): float(
                config.get("constraints", {}).get("receptor_activation_cost", 0.0)
                if isinstance(config.get("constraints", {}), Mapping)
                else 0.0
            )
            for row in receptor_manifest
        },
        warm_start_state={"strategy": "fixed_label_free", "tasks": [list(task) for task in warm_tasks]},
        scaffold_metadata={str(row["ligand_id"]): str(row.get("scaffold", row.get("scaffold_smiles", "__unknown__"))) for row in ligand_manifest},
        receptor_cluster_metadata={str(row["receptor_id"]): str(row.get("cluster", row.get("receptor_cluster", "__unknown__"))) for row in receptor_manifest},
    )
    state.reveal({task: float(scores[task]) for task in warm_tasks})
    return state


def build_masked_replay_state(
    score_matrix: Mapping[Task, float],
    ligand_manifest: Sequence[Mapping[str, object]],
    receptor_manifest: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
) -> PartialObservationState:
    """Construct a state with only fixed warm-start scores exposed."""
    return _state_for_replay(
        score_matrix,
        ligand_manifest,
        receptor_manifest,
        _runtime_config(config),
    )


def run_masked_prediction_gate(
    score_matrix: Mapping[Task, float],
    ligand_manifest: Sequence[Mapping[str, object]],
    receptor_manifest: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
) -> dict[str, object]:
    """Score prediction models on masked pairs after fitting only warm-start data."""
    runtime = _runtime_config(config)
    state = _state_for_replay(score_matrix, ligand_manifest, receptor_manifest, runtime)
    predictor = _make_predictor(runtime, state)
    hidden = {task: float(score) for task, score in score_matrix.items() if task not in state.observed_scores}
    primary = predictor.calibration_report(hidden)
    baselines: dict[str, dict[str, float | int]] = {}
    for name, baseline in (("nearest_receptor", NearestReceptorPredictor()), ("observed_mean", ObservedScoreMeanPredictor())):
        baselines[name] = baseline.fit(state, training_data=[]).calibration_report(hidden)
    best_baseline = min(report["rmse"] for report in baselines.values())
    return {
        "workflow": "masked_score_prediction_gate",
        "primary_model": runtime["predictor"].get("model", "bayesian_residual"),
        "primary_report": primary,
        "baseline_reports": baselines,
        "primary_model_allowed_for_replay": bool(primary["rmse"] <= best_baseline),
        "hidden_scores_used_only_after_prediction": True,
        "hidden_labels_used": False,
        "warm_start_observed_task_count": len(state.observed_scores),
    }


def run_masked_replay(
    score_matrix: Mapping[Task, float],
    ligand_manifest: Sequence[Mapping[str, object]],
    receptor_manifest: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
    *,
    hidden_labels: Mapping[str, str],
) -> ReplayResult:
    """Run deterministic strategy comparison; hidden labels are only read at evaluation."""
    runtime = _runtime_config(config)
    all_tasks = {(str(row["ligand_id"]), str(receptor["receptor_id"])) for row in ligand_manifest for receptor in receptor_manifest}
    if set(score_matrix) != all_tasks:
        missing = sorted(all_tasks - set(score_matrix))
        extra = sorted(set(score_matrix) - all_tasks)
        raise ValueError(f"score matrix task identities differ; missing={missing}, extra={extra}")
    initial_state = _state_for_replay(score_matrix, ligand_manifest, receptor_manifest, runtime)
    initial_predictor = _make_predictor(runtime, initial_state)
    common_pool = _candidate_pool(initial_state, initial_predictor, runtime)
    strategy_results: list[ReplayStrategyResult] = []
    total_budget = float(runtime["budget"]["total_cost"])
    batch_budget = float(runtime["budget"]["batch_cost"])
    for strategy in runtime["strategies"]:
        strategy_name = str(strategy)
        state = initial_state.copy()
        audits: list[dict[str, object]] = []
        sequence: list[Task] = []
        for _ in range(1000):
            if state.docking_cost >= total_budget - 1e-9:
                break
            available = tuple(task for task in common_pool if task not in state.observed_scores)
            if not available:
                break
            predictor = _make_predictor(runtime, state)
            acquisition_raw = runtime["acquisition"]
            assert isinstance(acquisition_raw, Mapping)
            evaluator = PosteriorAcquisitionEvaluator(
                state,
                predictor,
                AcquisitionConfig(
                    top_q=int(acquisition_raw.get("top_q", 10)),
                    monte_carlo_samples=int(acquisition_raw.get("monte_carlo_samples", 64)),
                    risk_lambda=float(acquisition_raw.get("risk_lambda", 0.0)),
                    utility_mode=str(acquisition_raw.get("utility_mode", "ranking_score")),
                    random_seed=int(runtime.get("random_seed", 0)) + state.current_round,
                ),
            )
            remaining_budget = min(batch_budget, total_budget - state.docking_cost)
            constraints = _constraints_for(available, state, runtime, remaining_budget)
            values = evaluator.all_task_values(available)
            interactions = evaluator.interaction_matrix(available)
            qubo = build_batch_qubo(available, values, interactions, constraints, batch_interaction_weight=float(acquisition_raw.get("batch_interaction_weight", 1.0)))
            if strategy_name == "random":
                selected = _random_tasks(available, constraints, int(runtime.get("random_seed", 0)) + state.current_round)
                solver_audit = {"backend": "random", "tasks": [list(task) for task in selected], "energy": qubo.energy_for_tasks(selected), "time_budget_seconds": runtime.get("solver", {}).get("time_budget_seconds")}
            elif strategy_name == "receptor_round_robin":
                selected = _round_robin_tasks(available, state, constraints)
                solver_audit = {"backend": "receptor_round_robin", "tasks": [list(task) for task in selected], "energy": qubo.energy_for_tasks(selected), "time_budget_seconds": runtime.get("solver", {}).get("time_budget_seconds")}
            else:
                solver_config = runtime.get("solver", {})
                assert isinstance(solver_config, Mapping)
                solver_result = solve_batch_qubo(qubo, backend=strategy_name, random_seed=int(runtime.get("random_seed", 0)) + state.current_round, time_budget_seconds=solver_config.get("time_budget_seconds"))
                selected = solver_result.tasks
                solver_audit = _deterministic_solver_audit(solver_result.as_dict())
            if not selected:
                audits.append({
                    "round": state.current_round,
                    "candidate_pool": [list(task) for task in common_pool],
                    "available_tasks": [list(task) for task in available],
                    "selected_tasks": [],
                    "revealed_tasks": [],
                    "stop_reason": "no_feasible_or_positive_batch",
                    "solver": solver_audit,
                })
                break
            # This is the only point where oracle scores enter the visible state.
            state.reveal({task: float(score_matrix[task]) for task in selected})
            sequence.extend(selected)
            audits.append({
                "round": state.current_round - 1,
                "candidate_pool": [list(task) for task in common_pool],
                "available_tasks": [list(task) for task in available],
                "selected_tasks": [list(task) for task in selected],
                "revealed_tasks": [list(task) for task in selected],
                "solver": solver_audit,
                "qubo_fingerprint": qubo.fingerprint,
                "time_budget_seconds": runtime.get("solver", {}).get("time_budget_seconds"),
                "observed_task_count_after": len(state.observed_scores),
                "docking_cost_after": state.docking_cost,
            })
        final_predictor = _make_predictor(runtime, state)
        evaluation_raw = runtime["evaluation"]
        assert isinstance(evaluation_raw, Mapping)
        evaluation = _final_evaluation(state, final_predictor, hidden_labels, [str(item) for item in evaluation_raw["metrics"]])
        strategy_results.append(ReplayStrategyResult(
            name=strategy_name,
            task_sequence=tuple(sequence),
            rounds=tuple(audits),
            evaluation=evaluation,
            final_state=state.to_dict(),
        ))
    return ReplayResult(
        strategies=tuple(strategy_results),
        metadata={
            "workflow": "masked_active_docking_replay",
            "backend_types": {strategy.name: [audit["solver"].get("backend") for audit in strategy.rounds] for strategy in strategy_results},
            "candidate_pool": [list(task) for task in common_pool],
            "candidate_pruning": _candidate_pruning_report(initial_state, common_pool, runtime),
            "replay_mask_strategy": runtime.get("replay_mask_strategy", "scaffold_cluster"),
            "candidate_pool_is_shared": True,
            "oracle_scores_used_only_after_selection": True,
            "hidden_labels_used_only_for_final_evaluation": True,
            "real_docking_executed": False,
            "quantum_hardware_used": False,
        },
    )
