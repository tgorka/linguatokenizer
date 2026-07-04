"""Command-line entry point for tokenizer-core.

Subcommands are added as BMAD stories land (``train``, ``benchmark``,
``adapt``, ...). ``benchmark`` (Story 1.4) runs the baseline comparison table
end to end; ``train`` (Story 2.1) trains each recipe candidate; ``adapt``
remains to be added.
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
from tokenizer_core.corpora.sources import _resolve_data_dir
from tokenizer_core.economics import PRICING_PRESETS, estimate_savings
from tokenizer_core.train.artifact import save_artifact
from tokenizer_core.train.config import Recipe
from tokenizer_core.train.trainers import TrainerError, train_candidate


def _run_economics(args: argparse.Namespace) -> int:
    """Handler for the ``economics`` subcommand.

    Turns a baseline vs optimized fertility (tokens/word) plus a monthly
    workload and a pricing preset into cost/throughput/context savings.
    """
    pricing = PRICING_PRESETS.get(args.pricing)
    if pricing is None:
        print(
            f"unknown pricing preset {args.pricing!r}; choose from "
            f"{sorted(PRICING_PRESETS)}"
        )
        return 2

    report = estimate_savings(
        args.baseline_fertility,
        args.optimized_fertility,
        pricing=pricing,
        input_words_per_month=args.input_words,
        output_words_per_month=args.output_words,
    )
    print(
        f"fertility {report.baseline_fertility:.3f} -> "
        f"{report.optimized_fertility:.3f} tok/word "
        f"({report.token_reduction_pct:.0f}% fewer tokens)"
    )
    print(
        f"cost ({pricing.name}): ${report.monthly_cost_before:,.0f}/mo -> "
        f"${report.monthly_cost_after:,.0f}/mo "
        f"(save ${report.monthly_savings:,.0f}/mo, "
        f"{report.savings_pct:.0f}%)"
    )
    print(
        f"throughput x{report.throughput_gain_x:.2f}  "
        f"effective-context x{report.effective_context_gain_x:.2f}  "
        f"KV-memory -{report.kv_memory_reduction_pct:.0f}%"
    )
    return 0


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


def _run_train(args: argparse.Namespace) -> int:
    """Handler for the ``train`` subcommand.

    Loads ``args.recipe``, trains every ``bpe``/``unigram`` candidate,
    saves each trained artifact under
    ``<data_dir>/tokenizers/<language_code>/<candidate.name>/``, skips any
    candidate whose algorithm isn't trainable here (e.g. ``superbpe``, Story
    2.2's scope) with a printed message, and prints one summary line per
    candidate that trains successfully.
    """
    recipe = Recipe.load(args.recipe)
    data_dir = _resolve_data_dir(args.data_dir)

    for candidate in recipe.candidates:
        try:
            tokenizer, metadata = train_candidate(
                recipe,
                candidate,
                n_docs=args.n_docs,
                seed=args.seed,
                data_dir=data_dir,
            )
        except TrainerError as err:
            print(f"{candidate.name}: skipped ({err})")
            continue

        artifact_dir = (
            data_dir / "tokenizers" / recipe.language.code / candidate.name
        )
        save_artifact(tokenizer, metadata, artifact_dir)

        print(
            f"{candidate.name}: trained (vocab_size={metadata.vocab_size}, "
            f"corpus_hash={metadata.corpus_hash})"
        )

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

    train_parser = subparsers.add_parser(
        "train",
        help="Train each bpe/unigram candidate in a recipe.",
    )
    train_parser.add_argument(
        "--recipe",
        default="languages/pl/recipe.yaml",
        help="Path to the language recipe YAML file.",
    )
    train_parser.add_argument(
        "--data-dir",
        default=None,
        help="Data-plane root (defaults to $TOKENIZER_DATA_DIR).",
    )
    train_parser.add_argument(
        "--n-docs",
        type=int,
        default=10_000,
        help=(
            "Total documents to sample across recipe.train_corpora. Kept "
            "modest by default: byte-level BPE (and especially Unigram) "
            "training memory scales with corpus size, and a 32k vocab needs "
            "only tens of MB. Raise it if you have the RAM headroom."
        ),
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for corpus sampling.",
    )
    train_parser.set_defaults(func=_run_train)

    economics_parser = subparsers.add_parser(
        "economics",
        help="Estimate inference cost/throughput/context savings.",
    )
    economics_parser.add_argument(
        "--baseline-fertility",
        type=float,
        required=True,
        help="Current tokenizer tokens/word on the target language.",
    )
    economics_parser.add_argument(
        "--optimized-fertility",
        type=float,
        required=True,
        help="Optimized tokenizer tokens/word.",
    )
    economics_parser.add_argument(
        "--pricing",
        default="claude-sonnet",
        help=f"Pricing preset ({', '.join(sorted(PRICING_PRESETS))}).",
    )
    economics_parser.add_argument(
        "--input-words",
        type=float,
        default=1_000_000,
        help="Prompt words per month (target language).",
    )
    economics_parser.add_argument(
        "--output-words",
        type=float,
        default=300_000,
        help="Generated words per month (target language).",
    )
    economics_parser.set_defaults(func=_run_economics)

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
