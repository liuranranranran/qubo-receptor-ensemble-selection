"""Run the production active ligand-receptor docking workflow.

The offline complete-matrix replay remains in ``run_active_docking.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qubo_receptor_ensemble.active_docking.production import ActiveProductionRunner
from qubo_receptor_ensemble.active_docking.production_config import (
    load_active_production_config,
)


def _non_empty_path(value: str) -> Path:
    if not value.strip():
        raise argparse.ArgumentTypeError("path must be non-empty")
    return Path(value)


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--prepared-run-directory", type=_non_empty_path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=("json", "csv", "markdown"), default="json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "prepare", "run", "resume", "finalize"):
        command = subparsers.add_parser(name)
        _add_common_arguments(command)
        if name in {"prepare", "run"}:
            command.add_argument("--overwrite", action="store_true")
        if name in {"run", "resume"}:
            command.add_argument("--max-rounds", type=int, default=None)
    return parser


def _load_config(args: argparse.Namespace):
    return load_active_production_config(
        args.config,
        data_root=args.data_root,
        prepared_run_directory=args.prepared_run_directory,
    )


def _render(value: object, output_format: str) -> str:
    if output_format == "json":
        return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
    if output_format == "csv":
        if isinstance(value, Mapping):
            lines = ["field,value"]
            for key in sorted(value):
                item = json.dumps(value[key], ensure_ascii=True, allow_nan=False)
                lines.append(f"{key},{item}")
            return "\n".join(lines)
        return "field,value\nvalue," + json.dumps(value, ensure_ascii=True, allow_nan=False)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
    return "# Active Docking Production Audit\n\n```json\n" + payload + "\n```"


def _write(value: object, args: argparse.Namespace) -> None:
    rendered = _render(value, args.format)
    if args.output is None:
        print(rendered)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(json.dumps({"status": "written", "output": str(args.output)}, ensure_ascii=True))


def _progress(message: str) -> None:
    """Keep live progress on stderr so stdout remains machine-readable."""
    print(message, file=sys.stderr, flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _load_config(args)
    if args.command == "validate":
        _write(
            {
                "status": "valid",
                "workflow": config.workflow,
                "config_fingerprint": config.fingerprint,
                "prepared_run_directory": str(config.prepared_run_directory),
                "active_run_directory": str(config.active_run_directory),
                "real_docking_executed": False,
                "quantum_hardware_used": False,
            },
            args,
        )
        return 0

    runner = ActiveProductionRunner(config, progress=_progress)
    if args.command == "prepare":
        result = runner.prepare(overwrite=args.overwrite)
        result["real_docking_executed"] = False
    elif args.command == "run":
        result = runner.run(
            resume=False,
            overwrite=args.overwrite,
            max_rounds=args.max_rounds,
        )
    elif args.command == "resume":
        result = runner.run(resume=True, overwrite=False, max_rounds=args.max_rounds)
    else:
        result = runner.finalize()
    _write(result, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
