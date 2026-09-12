"""E1 headroom scan CLI.

Commands:

``verify``  D1 input verification only (no scan): writes ``input_manifest.json``
            with SHA-256, receptor order, folds and scaffold fill rate.
``run``     full pre-registered scan + gate + figures.
``report``  rebuild products from existing shard checkpoints.

Examples::

    python scripts/headroom_scan.py verify \
      --prereg configs/experiments/e1_headroom_preregistration.json \
      --assets configs/e1_assets.json --output-dir results/headroom/e1_20260911

    python scripts/headroom_scan.py run \
      --prereg configs/experiments/e1_headroom_preregistration.json \
      --assets configs/e1_assets.json --output-dir results/headroom/e1_20260911 \
      --jobs 24 --resume
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qubo_receptor_ensemble.headroom import runner  # noqa: E402


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prereg", required=True, help="frozen pre-registration JSON")
    parser.add_argument("--assets", required=True, help="E1 asset table JSON")
    parser.add_argument("--output-dir", required=True, help="run directory under results/headroom/")
    parser.add_argument(
        "--root",
        action="append",
        default=None,
        metavar="NAME=PATH",
        help="override an asset root, e.g. --root run_root=$DATA_ROOT/results/headroom/e1_x",
    )


def _add_scan_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--jobs", type=int, default=1, help="parallel shards (processes)")
    parser.add_argument("--resume", action="store_true", help="skip finished shard checkpoints")
    parser.add_argument("--targets", default=None, help="comma-separated subset of asset target ids")
    parser.add_argument("--k-min", type=int, default=None)
    parser.add_argument("--k-max", type=int, default=None)
    parser.add_argument("--bootstrap-iterations", type=int, default=None)
    parser.add_argument("--top-m", type=int, default=None)
    parser.add_argument("--permutations", type=int, default=None)
    parser.add_argument("--perm-ks", default=None, help="comma-separated k values for H_perm")
    parser.add_argument("--inner-fold-count", type=int, default=3)
    parser.add_argument("--skip-perm", action="store_true")
    parser.add_argument("--skip-phi-selection", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--no-train-oracle", action="store_true")
    parser.add_argument("--max-subsets-per-k", type=int, default=None)
    parser.add_argument("--allow-missing-primary", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="debug profile: k<=4, 200 bootstrap iterations, 50 permutations",
    )


def _parse_roots(values: list[str] | None) -> dict[str, str] | None:
    if not values:
        return None
    roots: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise SystemExit(f"--root expects NAME=PATH, got: {item}")
        name, path = item.split("=", 1)
        name = name.strip()
        if not name or not path.strip():
            raise SystemExit(f"--root expects NAME=PATH, got: {item}")
        roots[name] = path.strip()
    return roots


def _parse_targets(value: str | None) -> tuple[str, ...] | None:
    if not value:
        return None
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_ints(value: str | None) -> tuple[int, ...] | None:
    if not value:
        return None
    return tuple(int(item) for item in value.split(",") if item.strip())


def command_verify(args: argparse.Namespace) -> int:
    from qubo_receptor_ensemble.headroom import assets as asset_module
    from qubo_receptor_ensemble.headroom.runner import (
        build_run_paths,
        load_preregistration,
        _load_panels,
        _config_hash,
    )
    from qubo_receptor_ensemble.io import write_json

    prereg = load_preregistration(Path(args.prereg))
    roots, specs = asset_module.load_asset_specs(Path(args.assets), _parse_roots(args.root))
    targets = _parse_targets(args.targets)
    if targets:
        specs = [spec for spec in specs if spec.target_id in set(targets)]
    panels, verification = _load_panels(specs)
    paths = build_run_paths(Path(args.output_dir))
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.cells.mkdir(parents=True, exist_ok=True)
    input_paths = [Path(args.prereg), Path(args.assets)]
    for spec in specs:
        input_paths.extend(
            path for path in (spec.matrix, spec.manifest, spec.problem_json, spec.backfill_problem_json) if path
        )
    write_json(
        paths.input_manifest,
        {
            "schema": "e1_input_manifest_v1",
            "run_id": paths.root.name,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "preregistration": {"path": prereg["_path"], "sha256": prereg["_sha256"]},
            "assets_config": Path(args.assets).as_posix(),
            "roots": roots,
            "targets": verification,
            "primary_targets_available": sorted(panels),
            "files_sha256": asset_module.sha256_records(input_paths),
        },
    )
    for report in verification:
        if report.get("status") != "ok":
            print(f"[verify] {report['target_id']}: {report['status']} {report.get('error', '')}")
            continue
        print(
            f"[verify] {report['target_id']} ({report['role']}): "
            f"{report['ligand_count']} ligands x {report['receptor_count']} receptors | "
            f"folds={report['fold_count']} | scaffolds={report['scaffold_count']} | "
            f"source={report['source']} | backfilled_scaffold={report['backfilled_scaffold_rows']}"
        )
        if report.get("problems"):
            print(f"          problems: {report['problems']}")
    print(f"[verify] manifest -> {paths.input_manifest}")
    return 0


def command_run(args: argparse.Namespace) -> int:
    k_list = None
    bootstrap_iterations = args.bootstrap_iterations
    permutations = args.permutations
    top_m = args.top_m
    if args.quick:
        k_min = args.k_min if args.k_min is not None else 1
        k_max = args.k_max if args.k_max is not None else 4
        k_list = tuple(range(k_min, k_max + 1))
        bootstrap_iterations = bootstrap_iterations if bootstrap_iterations is not None else 200
        permutations = permutations if permutations is not None else 50
    elif args.k_min is not None or args.k_max is not None:
        prereg = runner.load_preregistration(Path(args.prereg))
        k_min = args.k_min if args.k_min is not None else int(prereg["k_range"][0])
        k_max = args.k_max if args.k_max is not None else int(prereg["k_range"][1])
        k_list = tuple(range(k_min, k_max + 1))
    result = runner.run_e1(
        prereg_path=Path(args.prereg),
        assets_path=Path(args.assets),
        output_dir=Path(args.output_dir),
        jobs=args.jobs,
        resume=args.resume,
        root_overrides=_parse_roots(args.root),
        targets=_parse_targets(args.targets),
        k_list=k_list,
        bootstrap_iterations=bootstrap_iterations,
        top_m=top_m,
        permutations=permutations,
        perm_ks=_parse_ints(args.perm_ks),
        inner_fold_count=args.inner_fold_count,
        skip_perm=args.skip_perm,
        skip_phi_selection=args.skip_phi_selection,
        skip_figures=args.skip_figures,
        with_train_oracle=not args.no_train_oracle,
        max_subsets_per_k=args.max_subsets_per_k,
        allow_missing_primary=args.allow_missing_primary,
    )
    gate = result["gate"]
    print(
        "gate: {decision} | per-cell GO={per:.3f} | fold-oracle GO={oracle} | "
        "train-selected GO={train}".format(
            decision=gate.get("decision"),
            per=float(gate.get("fraction_cells_go_per_phi_cell") or 0.0),
            oracle=gate.get("fraction_cells_go_fold_oracle_phi"),
            train=gate.get("fraction_cells_go_train_selected_phi"),
        )
    )
    print(f"products -> {result['paths']['root']}")
    return 0


def command_report(args: argparse.Namespace) -> int:
    result = runner.assemble_products(
        prereg_path=Path(args.prereg),
        assets_path=Path(args.assets),
        output_dir=Path(args.output_dir),
        jobs=int(args.jobs),
        root_overrides=_parse_roots(args.root),
        inner_fold_count=args.inner_fold_count,
        skip_figures=args.skip_figures,
        targets=_parse_targets(args.targets),
    )
    gate = result["gate"]
    print(f"rebuilt {result['shard_count']} shards | gate={gate.get('decision')}")
    print(f"products -> {args.output_dir}")
    return 0


def command_extract_seeds(args: argparse.Namespace) -> int:
    """Rebuild per-seed / min-aggregated matrices from a run's score_tables."""
    from qubo_receptor_ensemble.headroom.seed_matrices import (
        aggregate_seed_matrices,
        discover_seeds,
        extract_seed_matrix,
        reference_receptor_order,
        write_seed_matrix,
    )
    from qubo_receptor_ensemble.io import write_json

    run_dir = Path(args.run_dir)
    score_tables = run_dir / "score_tables"
    if not score_tables.is_dir():
        raise SystemExit(f"score_tables directory not found: {score_tables}")
    reference = (
        Path(args.reference_matrix)
        if args.reference_matrix
        else run_dir / "matrices" / "primary_median_matrix.csv"
    )
    receptor_order = (
        reference_receptor_order(reference) if reference.is_file() else None
    )
    seeds = _parse_ints(args.seeds) or discover_seeds(score_tables)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "matrices" / "seed_matrices"
    output_dir.mkdir(parents=True, exist_ok=True)
    audit: dict[str, object] = {
        "schema": "e1_seed_extraction_v1",
        "run_dir": run_dir.as_posix(),
        "score_tables": score_tables.as_posix(),
        "reference_matrix": reference.as_posix() if reference.is_file() else None,
        "receptor_order": list(receptor_order) if receptor_order else None,
        "seeds": list(seeds),
        "outputs": {},
    }
    matrices = {}
    for seed in seeds:
        seed_matrix = extract_seed_matrix(
            score_tables,
            seed,
            target_id=args.target_id,
            receptor_order=receptor_order,
        )
        record = write_seed_matrix(seed_matrix, output_dir / f"seed_{seed}_matrix.csv")
        record["ligand_count"] = seed_matrix.n_ligands
        record["receptor_count"] = seed_matrix.n_receptors
        audit["outputs"][str(seed)] = record  # type: ignore[index]
        matrices[seed] = seed_matrix
        print(
            f"[seed] {seed}: {seed_matrix.n_ligands} ligands x {seed_matrix.n_receptors} receptors "
            f"-> {record['path']}"
        )
    if len(matrices) > 1:
        aggregations = ("min", "median") if args.aggregation == "both" else (args.aggregation,)
        for aggregation in aggregations:
            aggregated = aggregate_seed_matrices(matrices, aggregation=aggregation)
            record = write_seed_matrix(aggregated, output_dir / f"seed_{aggregation}_matrix.csv")
            record["ligand_count"] = aggregated.n_ligands
            record["receptor_count"] = aggregated.n_receptors
            audit[f"aggregated_{aggregation}"] = record
            print(f"[seed] {aggregation}-aggregated -> {record['path']}")
    write_json(output_dir / "seed_extraction_audit.json", audit)
    print(f"[seed] audit -> {(output_dir / 'seed_extraction_audit.json').as_posix()}")
    return 0


def command_summarize(args: argparse.Namespace) -> int:
    """Compact summary of a finished run (gate + headline numbers + optional sensitivity)."""
    from qubo_receptor_ensemble.headroom.summary import build_summary, format_summary_text
    from qubo_receptor_ensemble.io import write_json

    summary = build_summary(
        Path(args.run_dir),
        Path(args.sensitivity_dir) if args.sensitivity_dir else None,
    )
    text = format_summary_text(summary)
    print(text, end="")
    if args.output_json:
        write_json(Path(args.output_json), summary)
        print(f"[summarize] json -> {args.output_json}")
    if args.output_text:
        Path(args.output_text).write_text(text, encoding="utf-8")
        print(f"[summarize] text -> {args.output_text}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser("verify", help="D1 input verification only")
    _add_common(verify)
    verify.add_argument("--targets", default=None)
    verify.set_defaults(handler=command_verify)

    run = subparsers.add_parser("run", help="run the pre-registered E1 scan")
    _add_common(run)
    _add_scan_options(run)
    run.set_defaults(handler=command_run)

    seeds = subparsers.add_parser(
        "extract-seeds",
        help="rebuild per-seed and min-aggregated matrices from a run's score_tables",
    )
    seeds.add_argument("--run-dir", required=True, help="canonical run directory containing score_tables/")
    seeds.add_argument("--output-dir", default=None, help="defaults to <run-dir>/matrices/seed_matrices")
    seeds.add_argument("--seeds", default=None, help="comma-separated seeds; defaults to auto-discovery")
    seeds.add_argument("--target-id", default=None, help="override the target id written into the matrix")
    seeds.add_argument(
        "--aggregation",
        choices=("min", "median", "both"),
        default="both",
        help="aggregated matrix over the seeds (default: both)",
    )
    seeds.add_argument(
        "--reference-matrix",
        default=None,
        help="canonical primary matrix used to freeze the receptor column order",
    )
    seeds.set_defaults(handler=command_extract_seeds)

    summarize = subparsers.add_parser(
        "summarize", help="compact summary of a finished run (for watch/archive)"
    )
    summarize.add_argument("--run-dir", required=True)
    summarize.add_argument("--sensitivity-dir", default=None)
    summarize.add_argument("--output-json", default=None)
    summarize.add_argument("--output-text", default=None)
    summarize.set_defaults(handler=command_summarize)

    report = subparsers.add_parser("report", help="rebuild products from checkpoints")
    _add_common(report)
    report.add_argument("--jobs", type=int, default=1)
    report.add_argument("--targets", default=None)
    report.add_argument("--inner-fold-count", type=int, default=None)
    report.add_argument("--skip-figures", action="store_true")
    report.set_defaults(handler=command_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())