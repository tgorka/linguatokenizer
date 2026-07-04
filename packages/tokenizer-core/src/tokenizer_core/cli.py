"""Command-line entry point for tokenizer-core.

Subcommands are added as BMAD stories land (``train``, ``benchmark``,
``adapt``, ...). ``benchmark`` (Story 1.4) runs the baseline comparison table
end to end; ``adapt``/``train`` remain to be added.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from tokenizer_core import __version__
from tokenizer_core.benchmark import (
    compare_baselines,
    write_comparison_markdown,
)
from tokenizer_core.benchmark.fertility import write_report
from tokenizer_core.train.config import Recipe


def _run_benchmark(args: argparse.Namespace) -> int:
    """Handler for the ``benchmark`` subcommand.

    Loads ``args.recipe``, runs :func:`compare_baselines`, writes both the
    JSON and Markdown comparison artifacts under ``args.out_dir``, and prints
    one summary line per baseline.
    """
    recipe = Recipe.load(args.recipe)
    report = compare_baselines(recipe, data_dir=args.data_dir)

    out_dir = Path(args.out_dir)
    language_code = report.language_code
    json_path = write_report(
        report, out_dir / f"baselines-{language_code}-en.json"
    )
    md_path = write_comparison_markdown(
        report, out_dir / f"baselines-{language_code}-en.md"
    )

    for outcome in report.outcomes:
        if outcome.report is not None:
            print(
                f"{outcome.name}: fertility_vs_english="
                f"{outcome.report.fertility_vs_english:.3f}"
            )
        else:
            print(f"{outcome.name}: skipped ({outcome.skip_reason})")

    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="tokenizer",
        description=(
            "Token-efficient tokenizers for under-served languages "
            "and adapters onto existing model tokenizers."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"tokenizer-core {__version__}",
    )
    parser.set_defaults(func=None)
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Score baseline tokenizers on aligned FLORES pl/en fertility.",
    )
    benchmark_parser.add_argument(
        "--recipe",
        default="languages/pl/recipe.yaml",
        help="Path to the language recipe YAML file.",
    )
    benchmark_parser.add_argument(
        "--out-dir",
        default="results",
        help="Directory to write the comparison JSON/Markdown artifacts to.",
    )
    benchmark_parser.add_argument(
        "--data-dir",
        default=None,
        help="Corpus data-plane root (defaults to $TOKENIZER_DATA_DIR).",
    )
    benchmark_parser.set_defaults(func=_run_benchmark)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.func is None:
        parser.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
