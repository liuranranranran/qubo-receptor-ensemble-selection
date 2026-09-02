"""CLI for creating anonymized offline active-docking replay inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from qubo_receptor_ensemble.active_docking.replay_inputs import (
    prepare_anonymous_replay_inputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare anonymized JSON inputs from active prepare outputs and a complete score matrix."
    )
    parser.add_argument("--active-manifest", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = prepare_anonymous_replay_inputs(args.active_manifest, args.matrix, args.output)
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
