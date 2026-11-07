"""Command-line entry point."""

import argparse

from .analysis import analyse as run_analysis
from .config import DATASETS
from .interpretability import interpret
from .pipeline import run_experiments
from .sensitivity import pnfsens
from .validation import validate_results


def main(argv=None):
    parser = argparse.ArgumentParser(prog="gmdh-conformal")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run the experiment protocol")
    run.add_argument("--datasets", default=",".join(DATASETS))
    run.add_argument("--repeats", type=int, default=5)
    run.add_argument("--output-dir", default="outputs")
    analysis_cmd = commands.add_parser("analyse", help="run detailed analysis on saved fold CSV files")
    analysis_cmd.add_argument("--output-dir", default="outputs")
    analysis_cmd.add_argument("--datasets", default=",".join(DATASETS))
    analysis_cmd.add_argument("--repeats", type=int, default=5)
    analysis_cmd.add_argument("--fig-dir", default="figs")
    commands.add_parser("interpret", help="export Yacht GMDH equations and EBM importances")
    sensitivity = commands.add_parser("pnfsens", help="run the PNF threshold sensitivity analysis")
    sensitivity.add_argument("--datasets", default=",".join(DATASETS))
    sensitivity.add_argument("--repeats", type=int, default=5)
    sensitivity.add_argument("--output-dir", default="outputs")
    validate = commands.add_parser("validate", help="check the archived revision result bundle")
    validate.add_argument("--results-dir", default="provenance/results/supplement_20261107/gmdh_revision/rev_out")
    args = parser.parse_args(argv)
    if args.command == "run":
        if args.repeats < 1:
            parser.error("--repeats must be positive")
        names = tuple(name.strip() for name in args.datasets.split(",") if name.strip())
        unknown = set(names).difference(DATASETS)
        if unknown:
            parser.error(f"unknown dataset(s): {', '.join(sorted(unknown))}")
        run_experiments(names, args.repeats, args.output_dir)
    elif args.command == "analyse":
        analysis_args = argparse.Namespace(
            results_dir=args.output_dir,
            datasets=args.datasets,
            repeats=args.repeats,
            fig_dir=args.fig_dir,
        )
        run_analysis(analysis_args)
    elif args.command == "interpret":
        interpret()
    elif args.command == "pnfsens":
        names = tuple(name.strip() for name in args.datasets.split(",") if name.strip())
        unknown = set(names).difference(DATASETS)
        if unknown:
            parser.error(f"unknown dataset(s): {', '.join(sorted(unknown))}")
        args.datasets = ",".join(names)
        pnfsens(args)
    else:
        try:
            validate_results(args.results_dir)
        except (ValueError, FileNotFoundError) as exc:
            parser.exit(1, f"Result validation failed:\n{exc}\n")


if __name__ == "__main__":
    main()
