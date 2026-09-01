from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from qubo_receptor_ensemble.active_docking.acquisition import (
    AcquisitionConfig,
    PosteriorAcquisitionEvaluator,
)
from qubo_receptor_ensemble.active_docking.config import (
    ActiveDockingConfigError,
    load_active_docking_config,
)
from qubo_receptor_ensemble.active_docking.predictor import (
    BayesianResidualPredictor,
    PredictorConfig,
    ScorePrediction,
)
from qubo_receptor_ensemble.active_docking.qubo import (
    BatchConstraints,
    BatchQUBO,
    build_batch_qubo,
)
from qubo_receptor_ensemble.active_docking.replay import (
    run_masked_replay,
)
from qubo_receptor_ensemble.active_docking.solvers import solve_batch_qubo
from qubo_receptor_ensemble.active_docking.state import (
    PartialObservationState,
    StateError,
)
from qubo_receptor_ensemble.active_docking.warm_start import (
    WarmStartConfig,
    plan_warm_start,
)


def _manifests() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    ligands = [
        {"ligand_id": "l1", "scaffold": "g1", "features": [1.0, 0.0]},
        {"ligand_id": "l2", "scaffold": "g1", "features": [0.9, 0.1]},
        {"ligand_id": "l3", "scaffold": "g2", "features": [0.0, 1.0]},
        {"ligand_id": "l4", "scaffold": "g3", "features": [0.2, 0.8]},
    ]
    receptors = [
        {"receptor_id": "r0", "cluster": "c0", "features": [1.0, 0.0]},
        {"receptor_id": "r1", "cluster": "c1", "features": [0.0, 1.0]},
        {"receptor_id": "r2", "cluster": "c2", "features": [0.5, 0.5]},
    ]
    return ligands, receptors


def _scores() -> dict[tuple[str, str], float]:
    return {
        (ligand, receptor): float(-10.0 - i - 0.5 * j)
        for i, ligand in enumerate(("l1", "l2", "l3", "l4"))
        for j, receptor in enumerate(("r0", "r1", "r2"))
    }


def test_partial_state_tracks_observed_and_unknown_and_serializes() -> None:
    ligands, receptors = _manifests()
    state = PartialObservationState(
        ligand_manifest=ligands,
        receptor_manifest=receptors,
        candidate_tasks={("l1", "r0"), ("l1", "r1")},
        task_costs={("l1", "r0"): 1.0, ("l1", "r1"): 2.0},
    )
    assert state.completed_tasks() == ()
    assert state.unfinished_tasks() == (("l1", "r0"), ("l1", "r1"))
    state.reveal({("l1", "r0"): -10.0})
    assert state.completed_tasks() == (("l1", "r0"),)
    assert state.unfinished_tasks() == (("l1", "r1"),)
    restored = PartialObservationState.from_dict(json.loads(json.dumps(state.to_dict())))
    assert restored.to_dict() == state.to_dict()
    assert restored.current_round == 1
    restored.assert_no_hidden_information()


def test_partial_state_reveal_is_atomic_and_rejects_duplicate_or_invalid_tasks() -> None:
    ligands, receptors = _manifests()
    state = PartialObservationState(
        ligand_manifest=ligands,
        receptor_manifest=receptors,
        candidate_tasks={("l1", "r0"), ("l1", "r1")},
    )
    with pytest.raises(StateError):
        state.reveal({("l1", "r0"): -1.0, ("l2", "r1"): -2.0})
    assert state.completed_tasks() == ()
    state.reveal({("l1", "r0"): -1.0})
    with pytest.raises(StateError):
        state.reveal({("l1", "r0"): -1.0})
    with pytest.raises(StateError):
        state.reveal({("l2", "r2"): -2.0})


def test_warm_start_is_deterministic_scaffold_stratified_and_cluster_covering() -> None:
    ligands, receptors = _manifests()
    config = WarmStartConfig(
        baseline_receptor="r0",
        cluster_fraction=0.5,
        min_ligands_per_cluster=1,
        random_seed=19,
    )
    first = plan_warm_start(ligands, receptors, config)
    second = plan_warm_start(ligands, receptors, config)
    assert first == second
    assert {ligand for ligand, receptor in first if receptor == "r0"} == {
        "l1",
        "l2",
        "l3",
        "l4",
    }
    covered_clusters = {
        next(row["cluster"] for row in receptors if row["receptor_id"] == receptor)
        for _, receptor in first
    }
    assert covered_clusters == {"c0", "c1", "c2"}
    assert all("label" not in ligand for ligand in ligands)


def test_bayesian_residual_predictor_has_posterior_schema_and_reproducible_samples() -> None:
    ligands, receptors = _manifests()
    state = PartialObservationState(
        ligand_manifest=ligands,
        receptor_manifest=receptors,
        candidate_tasks=set((ligand["ligand_id"], receptor["receptor_id"]) for ligand in ligands for receptor in receptors),
    )
    state.reveal({
        ("l1", "r0"): -10.0,
        ("l1", "r1"): -10.4,
        ("l2", "r0"): -11.0,
        ("l2", "r1"): -11.2,
        ("l3", "r0"): -12.0,
        ("l4", "r0"): -13.0,
    })
    predictor = BayesianResidualPredictor(
        PredictorConfig(baseline_receptor="r0", posterior_samples=32, random_seed=7)
    )
    predictor.fit(state, training_data=[])
    candidates = (("l3", "r1"), ("l4", "r2"))
    predictions = predictor.predict(candidates)
    assert set(predictions) == set(candidates)
    assert all(isinstance(item, ScorePrediction) for item in predictions.values())
    assert all(item.variance > 0 for item in predictions.values())
    first = predictor.sample(candidates, random_state=123)
    second = predictor.sample(candidates, random_state=123)
    assert first == second
    assert predictor.calibration_report({("l4", "r1"): -13.0})["count"] == 1


def test_acquisition_computes_unit_value_pair_gamma_and_uncertainty_effect() -> None:
    ligands, receptors = _manifests()
    state = PartialObservationState(
        ligand_manifest=ligands,
        receptor_manifest=receptors,
        candidate_tasks={("l1", "r0"), ("l2", "r0"), ("l1", "r1"), ("l2", "r1")},
        task_costs={("l1", "r0"): 1.0, ("l2", "r0"): 1.0, ("l1", "r1"): 2.0, ("l2", "r1"): 1.0},
    )
    state.reveal({("l1", "r0"): -10.0, ("l2", "r0"): -11.0})
    predictor = BayesianResidualPredictor(
        PredictorConfig(baseline_receptor="r0", posterior_samples=16, random_seed=3)
    )
    predictor.fit(state, training_data=[])
    evaluator = PosteriorAcquisitionEvaluator(
        state,
        predictor,
        AcquisitionConfig(top_q=2, monte_carlo_samples=16, random_seed=5, risk_lambda=0.2),
    )
    assert evaluator.marginal_value(()) == 0.0
    single = evaluator.marginal_value((("l1", "r1"),))
    assert evaluator.unit_value(("l1", "r1")) == pytest.approx(single / 2.0)
    gamma = evaluator.pairwise_interaction(("l1", "r1"), ("l2", "r1"))
    expected = (
        evaluator.set_value((("l1", "r1"), ("l2", "r1")))
        - evaluator.set_value((("l1", "r1"),))
        - evaluator.set_value((("l2", "r1"),))
    )
    assert gamma == pytest.approx(expected)
    assert evaluator.predictions[("l1", "r1")].variance != evaluator.predictions[("l2", "r1")].variance


def test_batch_qubo_contains_symmetric_matrix_and_constraint_diagnostics() -> None:
    tasks = (("l1", "r1"), ("l1", "r2"), ("l2", "r1"))
    values = {tasks[0]: 3.0, tasks[1]: 2.0, tasks[2]: 2.5}
    gamma = {(tasks[0], tasks[1]): 1.0, (tasks[0], tasks[2]): -0.5, (tasks[1], tasks[2]): 0.0}
    constraints = BatchConstraints(
        budget=3.0,
        task_costs={task: 1.5 if task == tasks[0] else 1.0 for task in tasks},
        max_per_ligand=1,
        max_per_receptor=2,
        max_per_scaffold=1,
        ligand_scaffolds={"l1": "g1", "l2": "g2"},
        penalty=10.0,
    )
    qubo = build_batch_qubo(tasks, values, gamma, constraints, batch_interaction_weight=0.5)
    assert qubo.matrix.shape == (qubo.variable_count, qubo.variable_count)
    assert np.allclose(qubo.matrix, qubo.matrix.T)
    assert qubo.variable_count >= len(tasks)
    assert qubo.validate_tasks((tasks[0],)).is_feasible
    assert not qubo.validate_tasks((tasks[0], tasks[1])).is_feasible
    assert qubo.task_set_from_assignment(qubo.assignment_for_tasks((tasks[0],))) == (tasks[0],)
    assert len(qubo.fingerprint) == 64


def test_batch_qubo_supports_fixed_coefficient_scaling_and_equal_cost_cardinality() -> None:
    tasks = (("l1", "r1"), ("l2", "r1"), ("l3", "r2"))
    constraints = BatchConstraints(
        budget=2.0,
        task_costs={task: 1.0 for task in tasks},
        equal_cost=True,
        coefficient_scale=3.0,
        penalty=20.0,
    )
    qubo = build_batch_qubo(tasks, {task: 1.0 for task in tasks}, {}, constraints)
    assert qubo.metadata["coefficient_scale"] == 3.0
    assert "cardinality" in {name.split("|")[0] for name, _, _ in qubo.constraint_specs}
    assert not qubo.validate_tasks(tasks).is_feasible


def test_solvers_return_task_sets_and_quantum_backend_is_explicitly_simulated() -> None:
    tasks = (("l1", "r1"), ("l2", "r1"), ("l3", "r2"))
    constraints = BatchConstraints(
        budget=2.0,
        task_costs={task: 1.0 for task in tasks},
        max_per_ligand=1,
        max_per_receptor=2,
        max_per_scaffold=2,
        ligand_scaffolds={"l1": "g1", "l2": "g2", "l3": "g3"},
        penalty=20.0,
    )
    qubo = build_batch_qubo(tasks, {task: float(i + 1) for i, task in enumerate(tasks)}, {}, constraints)
    exact = solve_batch_qubo(qubo, backend="exact", random_seed=4)
    greedy = solve_batch_qubo(qubo, backend="value_greedy", random_seed=4)
    quantum = solve_batch_qubo(qubo, backend="quantum_compatible_simulator", random_seed=4)
    assert qubo.validate_tasks(exact.tasks).is_feasible
    assert qubo.validate_tasks(greedy.tasks).is_feasible
    assert qubo.validate_tasks(quantum.tasks).is_feasible
    assert quantum.metadata["quantum_hardware_used"] is False
    assert quantum.metadata["backend_type"] == "quantum_compatible_simulation"


def test_replay_is_masked_deterministic_and_reveals_only_selected_tasks() -> None:
    ligands, receptors = _manifests()
    labels = {"l1": "active", "l2": "decoy", "l3": "active", "l4": "decoy"}
    config = {
        "random_seed": 11,
        "warm_start": {
            "baseline_receptor": "r0",
            "cluster_fraction": 0.0,
            "min_ligands_per_cluster": 0,
        },
        "predictor": {"baseline_receptor": "r0", "posterior_samples": 8},
        "acquisition": {"top_q": 2, "monte_carlo_samples": 8, "risk_lambda": 0.1},
        "candidate_cap": 4,
        "budget": {"total_cost": 5.0, "batch_cost": 1.0},
        "constraints": {"max_per_ligand": 1, "max_per_receptor": 4, "max_per_scaffold": 2},
        "task_cost": 1.0,
        "strategies": ["value_greedy", "exact", "quantum_compatible_simulator"],
        "evaluation": {"metrics": ["bedroc20", "pr_auc", "ef1"]},
    }
    first = run_masked_replay(_scores(), ligands, receptors, config, hidden_labels=labels)
    second = run_masked_replay(_scores(), ligands, receptors, config, hidden_labels=labels)
    assert first.to_dict() == second.to_dict()
    for strategy in first.strategies:
        assert strategy.task_sequence
        assert all(
            {tuple(task) for task in round_audit["revealed_tasks"]}
            <= {tuple(task) for task in round_audit["selected_tasks"]}
            for round_audit in strategy.rounds
        )
        assert all("hidden_score" not in json.dumps(round_audit) for round_audit in strategy.rounds)
        assert all("active" not in json.dumps(round_audit) and "decoy" not in json.dumps(round_audit) for round_audit in strategy.rounds)
        assert strategy.evaluation["hidden_labels_used_for_evaluation"] is True
    assert len({tuple(tuple(task) for task in round_audit["candidate_pool"]) for round_audit in first.strategies[0].rounds}) == 1


def test_config_is_independent_and_rejects_real_docking(tmp_path: Path) -> None:
    config_path = tmp_path / "active.json"
    config_path.write_text(
        json.dumps({
            "schema_version": "1.0",
            "workflow": "masked_active_docking_replay",
            "random_seed": 1,
            "warm_start": {"baseline_receptor": "r0"},
            "predictor": {"baseline_receptor": "r0", "posterior_samples": 4},
            "acquisition": {"top_q": 2, "monte_carlo_samples": 4},
            "candidate_cap": 10,
            "budget": {"total_cost": 4, "batch_cost": 2},
            "constraints": {"max_per_ligand": 1, "max_per_receptor": 2, "max_per_scaffold": 1},
            "solver": {"backend": "exact"},
            "evaluation": {"metrics": ["bedroc20"]},
            "artifact_output_directory": "results/active",
        }),
        encoding="utf-8",
    )
    loaded = load_active_docking_config(config_path)
    assert loaded.workflow == "masked_active_docking_replay"
    assert loaded.artifact_output_directory.name == "active"
    bad = json.loads(config_path.read_text(encoding="utf-8"))
    bad["workflow"] = "real_docking"
    config_path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ActiveDockingConfigError):
        load_active_docking_config(config_path)
