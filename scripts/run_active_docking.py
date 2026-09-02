"""Validate and run the offline masked active ligand-receptor docking workflow."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qubo_receptor_ensemble.active_docking.config import load_active_docking_config
from qubo_receptor_ensemble.active_docking.replay import (
    run_masked_prediction_gate,
    run_masked_replay,
)


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_manifests(ligand_path: Path, receptor_path: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    ligands = _read_json(ligand_path)
    receptors = _read_json(receptor_path)
    if not isinstance(ligands, list) or not isinstance(receptors, list):
        raise ValueError("ligand and receptor manifests must be JSON lists")
    return [dict(row) for row in ligands], [dict(row) for row in receptors]


def _read_matrix(
    path: Path,
    *,
    include_labels: bool = True,
) -> tuple[dict[tuple[str, str], float], dict[str, str]]:
    if path.suffix.lower() == ".json":
        payload = _read_json(path)
        if isinstance(payload, dict):
            rows = payload.get("scores", payload.get("matrix", payload.get("rows", [])))
            labels = (
                {str(key): str(value) for key, value in payload.get("labels", {}).items()}
                if include_labels and isinstance(payload.get("labels", {}), dict)
                else {}
            )
        else:
            rows, labels = payload, {}
        if not isinstance(rows, list):
            raise ValueError("JSON matrix must contain a rows list")
        scores: dict[tuple[str, str], float] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("matrix rows must be objects")
            if "receptor_id" in row:
                task = (str(row["ligand_id"]), str(row["receptor_id"]))
                scores[task] = float(row.get("score", row.get("docking_score")))
            else:
                ligand_id = str(row["ligand_id"])
                if include_labels and "label" in row:
                    labels[ligand_id] = str(row["label"])
                for receptor_id, value in row.items():
                    if receptor_id not in {"ligand_id", "label", "target_id"} and value not in ("", None):
                        scores[(ligand_id, str(receptor_id))] = float(value)
        return scores, labels
    scores = {}
    labels = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("matrix CSV is empty")
    for row in rows:
        ligand_id = str(row.get("ligand_id", ""))
        if not ligand_id:
            raise ValueError("matrix CSV requires ligand_id")
        if include_labels and row.get("label"):
            labels[ligand_id] = str(row["label"])
        if row.get("receptor_id"):
            scores[(ligand_id, str(row["receptor_id"]))] = float(row.get("score", row.get("docking_score", "")))
        else:
            for receptor_id, value in row.items():
                if receptor_id not in {"ligand_id", "label", "target_id"} and value not in ("", None):
                    scores[(ligand_id, receptor_id)] = float(value)
    return scores, labels


def _add_inputs(parser: argparse.ArgumentParser, *, matrix: bool = False) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--format", choices=("json", "csv", "markdown"), default="json")
    if matrix:
        parser.add_argument("--matrix", type=Path, required=True)
        parser.add_argument("--ligand-manifest", type=Path, required=True)
        parser.add_argument("--receptor-manifest", type=Path, required=True)
        parser.add_argument("--output", type=Path, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="validate an offline replay config")
    _add_inputs(validate)
    for name in ("predict", "replay", "compare"):
        command = subparsers.add_parser(name)
        _add_inputs(command, matrix=True)
        if name in {"replay", "compare"}:
            command.add_argument("--labels", type=Path, default=None, help="optional label JSON used only after replay for evaluation")
    return parser


def _render_audit(value: object, output_format: str) -> str:
    if output_format == "json":
        return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=True)
    if not isinstance(value, Mapping) or not isinstance(value.get("strategies"), list):
        if output_format == "markdown":
            return "# Active Docking Audit\n\n```json\n" + json.dumps(value, indent=2, ensure_ascii=True, allow_nan=True) + "\n```"
        return "field,value\nvalue," + json.dumps(value, ensure_ascii=True)
    rows = []
    for strategy in value["strategies"]:
        evaluation = strategy.get("evaluation", {}) if isinstance(strategy, Mapping) else {}
        rows.append({
            "strategy": strategy.get("name", "") if isinstance(strategy, Mapping) else "",
            "revealed_tasks": len(strategy.get("task_sequence", [])) if isinstance(strategy, Mapping) else 0,
            "rounds": len(strategy.get("rounds", [])) if isinstance(strategy, Mapping) else 0,
            "bedroc20": evaluation.get("bedroc20", "") if isinstance(evaluation, Mapping) else "",
            "pr_auc": evaluation.get("pr_auc", "") if isinstance(evaluation, Mapping) else "",
            "ef1": evaluation.get("ef1", "") if isinstance(evaluation, Mapping) else "",
        })
    if output_format == "csv":
        columns = ["strategy", "revealed_tasks", "rounds", "bedroc20", "pr_auc", "ef1"]
        lines = [",".join(columns)]
        for row in rows:
            lines.append(",".join(str(row[column]) for column in columns))
        return "\n".join(lines)
    lines = ["# Active Docking Audit", "", "| strategy | revealed tasks | rounds | BEDROC20 | PR-AUC | EF1% |", "|---|---:|---:|---:|---:|---:|"]
    lines.extend(f"| {row['strategy']} | {row['revealed_tasks']} | {row['rounds']} | {row['bedroc20']} | {row['pr_auc']} | {row['ef1']} |" for row in rows)
    return "\n".join(lines)


def _write_output(value: object, output: Path | None, output_format: str) -> None:
    text = _render_audit(value, output_format)
    if output is None:
        print(text)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
        print(json.dumps({"status": "written", "output": str(output)}, ensure_ascii=True))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_active_docking_config(args.config)
    if args.command == "validate":
        _write_output({"status": "valid", "workflow": config.workflow, "strategies": list(config.strategies), "real_docking_executed": False}, None, args.format)
        return 0
    ligands, receptors = _read_manifests(args.ligand_manifest, args.receptor_manifest)
    scores, labels_from_matrix = _read_matrix(
        args.matrix,
        include_labels=args.command != "predict",
    )
    if args.command == "predict":
        result = run_masked_prediction_gate(scores, ligands, receptors, config.data)
    else:
        labels = labels_from_matrix
        if args.labels is not None:
            payload = _read_json(args.labels)
            labels = {str(key): str(value) for key, value in payload.items()} if isinstance(payload, dict) else labels
        if not labels:
            raise ValueError("replay requires labels in the matrix or --labels JSON")
        result = run_masked_replay(scores, ligands, receptors, config.data, hidden_labels=labels).to_dict()
    _write_output(result, args.output, args.format)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
