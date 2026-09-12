"""Compact summary of one E1 run (used by the remote watch-and-shutdown helper).

The summary is intentionally small and text-friendly: it is what gets written
next to the archive so a run can be audited after the machine is switched off.
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Mapping, Sequence

PRIMARY_K = (2, 3)


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except Exception:
        return []


def _to_float(value: object, default: float = float("nan")) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _mean(values: Sequence[float]) -> float | None:
    finite = [value for value in values if value == value]
    return float(statistics.fmean(finite)) if finite else None


def _fraction(values: Sequence[float], predicate) -> float | None:
    finite = [value for value in values if value == value]
    if not finite:
        return None
    return sum(1 for value in finite if predicate(value)) / len(finite)


def _gate(summary: Mapping[str, object]) -> dict[str, object]:
    gate = _read_json(Path(str(summary["main_dir"])) / "gate_g1.json")
    return {
        "decision": gate.get("decision"),
        "fraction_cells_go_per_phi_cell": gate.get("fraction_cells_go_per_phi_cell"),
        "fraction_cells_go_fold_oracle_phi": gate.get("fraction_cells_go_fold_oracle_phi"),
        "fraction_cells_go_train_selected_phi": gate.get("fraction_cells_go_train_selected_phi"),
        "n_cells_considered_per_phi_cell": gate.get("n_cells_considered_per_phi_cell"),
        "k_star_displacement": (gate.get("tie_break_trace") or {}).get("k_star_displacement"),
    }


def build_summary(main_dir: Path | str, sensitivity_dir: Path | str | None = None) -> dict[str, object]:
    """Aggregate the headline numbers of a main run and (optionally) its sensitivity run."""
    main = Path(main_dir)
    summary: dict[str, object] = {"main_dir": main.as_posix()}
    summary["gate"] = _gate({"main_dir": main.as_posix()})

    manifest = _read_json(main / "run_manifest.json")
    summary["run"] = {
        "run_id": manifest.get("run_id", main.name),
        "git_commit": manifest.get("git_commit"),
        "config": manifest.get("config"),
        "k_coverage": manifest.get("k_coverage"),
        "shard_count": manifest.get("shard_count"),
        "targets": manifest.get("targets"),
    }

    rows = [row for row in _read_csv(main / "cell_metrics.csv") if row.get("method") == "oracle"]
    if rows:
        h_raw = [_to_float(row.get("h_raw")) for row in rows]
        h_nested = [_to_float(row.get("h_nested")) for row in rows]
        noise = [_to_float(row.get("noise_floor")) for row in rows]
        summary["overall"] = {
            "cells": len(rows),
            "h_raw_mean": _mean(h_raw),
            "h_raw_positive_fraction": _fraction(h_raw, lambda value: value > 0),
            "h_nested_mean": _mean(h_nested),
            "h_nested_positive_fraction": _fraction(h_nested, lambda value: value > 0),
            "noise_floor_mean": _mean(noise),
            "cells_passing": sum(
                1
                for row in rows
                if _to_float(row.get("h_nested_lower95")) > _to_float(row.get("noise_floor"))
                and _to_float(row.get("ratio")) > 1.0
            ),
        }
        primary = [row for row in rows if str(row.get("role")) == "primary" and int(_to_float(row.get("k"), -1)) in PRIMARY_K]
        by_phi: dict[str, list[float]] = {}
        by_target: dict[str, list[float]] = {}
        for row in (primary or rows):
            by_phi.setdefault(str(row.get("phi")), []).append(_to_float(row.get("h_nested")))
            by_target.setdefault(str(row.get("target_id")), []).append(_to_float(row.get("h_nested")))
        summary["h_nested_mean_by_phi"] = {key: _mean(value) for key, value in sorted(by_phi.items())}
        summary["h_nested_mean_by_target"] = {key: _mean(value) for key, value in sorted(by_target.items())}

    map_rows = _read_csv(main / "headroom_map.csv")
    if map_rows:
        verdicts: dict[str, int] = {}
        for row in map_rows:
            verdict = str(row.get("verdict", ""))
            verdicts[verdict] = verdicts.get(verdict, 0) + 1
        top = sorted(
            map_rows,
            key=lambda row: -_to_float(row.get("ratio"), float("-inf")),
        )[:10]
        summary["map"] = {
            "rows": len(map_rows),
            "verdict_counts": verdicts,
            "top_ratio_rows": [
                {
                    "target_id": row.get("target_id"),
                    "role": row.get("role"),
                    "phi": row.get("phi"),
                    "k": row.get("k"),
                    "h_nested": _to_float(row.get("h_nested")),
                    "noise_floor": _to_float(row.get("noise_floor")),
                    "ratio": _to_float(row.get("ratio")),
                    "go_folds": row.get("go_folds"),
                }
                for row in top
            ],
        }

    bootstrap = _read_json(main / "bootstrap_report.json")
    per_cell = bootstrap.get("per_cell")
    summary["bootstrap"] = {
        "cells": len(per_cell) if isinstance(per_cell, dict) else 0,
        "unit": bootstrap.get("unit"),
        "iterations": bootstrap.get("iterations"),
    }

    if sensitivity_dir is not None:
        sensitivity = Path(sensitivity_dir)
        sens_rows = [
            row for row in _read_csv(sensitivity / "cell_metrics.csv") if row.get("method") == "oracle"
        ]
        targets: dict[str, list[float]] = {}
        passing: dict[str, int] = {}
        for row in sens_rows:
            if int(_to_float(row.get("k"), -1)) not in PRIMARY_K:
                continue
            target_id = str(row.get("target_id"))
            targets.setdefault(target_id, []).append(_to_float(row.get("h_nested")))
            if (
                _to_float(row.get("h_nested_lower95")) > _to_float(row.get("noise_floor"))
                and _to_float(row.get("ratio")) > 1.0
            ):
                passing[target_id] = passing.get(target_id, 0) + 1
        summary["sensitivity"] = {
            "dir": sensitivity.as_posix(),
            "shards": len(list((sensitivity / "cells").glob("*.json"))) if (sensitivity / "cells").is_dir() else 0,
            "h_nested_mean_by_target": {key: _mean(value) for key, value in sorted(targets.items())},
            "cells_passing_by_target": dict(sorted(passing.items())),
        }
    return summary


def format_summary_text(summary: Mapping[str, object]) -> str:
    """Human-readable rendering of :func:`build_summary`."""
    lines = [f"E1 summary | run={summary.get('run', {}).get('run_id') if isinstance(summary.get('run'), Mapping) else ''}"]
    gate = summary.get("gate") if isinstance(summary.get("gate"), Mapping) else {}
    lines.append(
        "gate: {decision} | per-cell={per:.3f} | fold-oracle-phi={oracle} | train-selected-phi={train} | grid={grid}".format(
            decision=gate.get("decision"),
            per=_to_float(gate.get("fraction_cells_go_per_phi_cell")),
            oracle=gate.get("fraction_cells_go_fold_oracle_phi"),
            train=gate.get("fraction_cells_go_train_selected_phi"),
            grid=gate.get("n_cells_considered_per_phi_cell"),
        )
    )
    overall = summary.get("overall") if isinstance(summary.get("overall"), Mapping) else {}
    if overall:
        lines.append(
            "overall: cells={cells} h_raw_mean={raw:.4f} (positive {rawpos:.2f}) h_nested_mean={nested:.4f} "
            "(positive {nestedpos:.2f}) noise_floor_mean={noise:.4f} passing={passing}".format(
                cells=overall.get("cells"),
                raw=_to_float(overall.get("h_raw_mean")),
                rawpos=_to_float(overall.get("h_raw_positive_fraction")),
                nested=_to_float(overall.get("h_nested_mean")),
                nestedpos=_to_float(overall.get("h_nested_positive_fraction")),
                noise=_to_float(overall.get("noise_floor_mean")),
                passing=overall.get("cells_passing"),
            )
        )
    by_phi = summary.get("h_nested_mean_by_phi")
    if isinstance(by_phi, Mapping) and by_phi:
        ranked = sorted(by_phi.items(), key=lambda item: -_to_float(item[1], float("-inf")))
        lines.append("phi ranking (mean H_nested, k in {2,3}): " + " | ".join(f"{phi}={_to_float(value):.4f}" for phi, value in ranked))
    by_target = summary.get("h_nested_mean_by_target")
    if isinstance(by_target, Mapping) and by_target:
        lines.append("targets: " + " | ".join(f"{target}={_to_float(value):.4f}" for target, value in by_target.items()))
    map_block = summary.get("map") if isinstance(summary.get("map"), Mapping) else {}
    if map_block:
        lines.append(f"map: rows={map_block.get('rows')} verdicts={map_block.get('verdict_counts')}")
    sensitivity = summary.get("sensitivity") if isinstance(summary.get("sensitivity"), Mapping) else {}
    if sensitivity:
        lines.append(f"sensitivity: shards={sensitivity.get('shards')}")
        by_variant = sensitivity.get("h_nested_mean_by_target")
        if isinstance(by_variant, Mapping):
            lines.append(
                "sensitivity H_nested (k in {2,3}): "
                + " | ".join(f"{key}={_to_float(value):.4f}" for key, value in by_variant.items())
            )
        passing = sensitivity.get("cells_passing_by_target")
        if isinstance(passing, Mapping) and passing:
            lines.append("sensitivity passing cells: " + " | ".join(f"{key}={value}" for key, value in passing.items()))
    return "\n".join(lines) + "\n"