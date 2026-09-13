"""E2 budget-allocation scan CLI (zero docking).

Examples::

    python scripts/budget_scan.py run \
      --prereg configs/experiments/e2_budget_preregistration.json \
      --assets  configs/e1_assets_remote.json \
      --output-dir "$DATA_ROOT/results/budget/e2_20260915" \
      --jobs 32 --resume

    python scripts/budget_scan.py report \
      --prereg configs/experiments/e2_budget_preregistration.json \
      --assets  configs/e1_assets_remote.json \
      --output-dir "$DATA_ROOT/results/budget/e2_20260915"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qubo_receptor_ensemble.budget import runner  # noqa: E402


def _ints(value: str | None) -> tuple[int, ...] | None:
    if not value:
        return None
    return tuple(int(item) for item in value.split(",") if item.strip())


def _strings(value: str | None) -> tuple[str, ...] | None:
    if not value:
        return None
    return tuple(item.strip() for item in value.split(",") if item.strip())


def command_run(args: argparse.Namespace) -> int:
    result = runner.run_e2(
        prereg_path=Path(args.prereg),
        assets_path=Path(args.assets),
        output_dir=Path(args.output_dir),
        jobs=int(args.jobs),
        resume=bool(args.resume),
        targets=_strings(args.targets),
        budgets=_ints(args.budgets),
        fusions=_strings(args.fusions),
        policies=_strings(args.policies),
        folds=_ints(args.folds),
    )
    gate = result["gate_g2"]
    print(
        "G2: {decision} | width-locked fraction={fraction} | shards={shards} rows={rows}".format(
            decision=gate.get("decision"),
            fraction=gate.get("fraction_width_locked"),
            shards=result["shard_count"],
            rows=result["cell_rows"],
        )
    )
    if gate.get("stable_policy_budget_phi"):
        print("stable policy@budget@phi:", gate["stable_policy_budget_phi"])
    print(f"products -> {args.output_dir}")
    return 0


def command_report(args: argparse.Namespace) -> int:
    paths = runner.build_paths(Path(args.output_dir))
    prereg = runner.load_preregistration(Path(args.prereg))
    shards = []
    for path in sorted(paths.cells.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and payload.get("schema") == "e2_shard_v1":
            shards.append(payload)
    if not shards:
        raise SystemExit(f"no E2 shard checkpoints under {paths.cells}")
    try:
        config = runner.config_from_shards(shards)
    except runner.BudgetRunnerError:
        config = runner.config_from_prereg(prereg)
        print("[warn] shard configs unavailable; falling back to the pre-registration config")
    # Preserve the provenance recorded by the run phase (targets, shard wall time).
    verification: tuple[dict[str, object], ...] = ()
    shard_elapsed: float | None = None
    if paths.input_manifest.is_file():
        payload = json.loads(paths.input_manifest.read_text(encoding="utf-8"))
        verification = tuple(payload.get("targets", ()))
    if paths.run_manifest.is_file():
        try:
            prior = json.loads(paths.run_manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            prior = {}
        if prior.get("shard_elapsed_seconds") is not None:
            shard_elapsed = float(prior["shard_elapsed_seconds"])
    result = runner.assemble_products(
        paths=paths,
        shards=shards,
        prereg=prereg,
        config=config,
        run_id=paths.root.name,
        panels={},
        verification=verification,
        elapsed=shard_elapsed,
        assets_path=Path(args.assets),
    )
    gate = result["gate_g2"]
    stable = gate.get("stable_policy_budget_phi") or {}
    print(
        f"rebuilt {result['shard_count']} shards | G2={gate.get('decision')} | "
        f"width-locked={gate.get('fraction_width_locked')} | stable={stable}"
    )
    print(f"products -> {args.output_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    for name, handler in (("run", command_run), ("report", command_report)):
        node = sub.add_parser(name)
        node.add_argument("--prereg", required=True)
        node.add_argument("--assets", required=True)
        node.add_argument("--output-dir", required=True)
        node.add_argument("--jobs", type=int, default=1)
        if name == "run":
            node.add_argument("--resume", action="store_true")
            node.add_argument("--targets", default=None)
            node.add_argument("--budgets", default=None)
            node.add_argument("--fusions", default=None)
            node.add_argument("--policies", default=None)
            node.add_argument("--folds", default=None)
        node.set_defaults(handler=handler)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())