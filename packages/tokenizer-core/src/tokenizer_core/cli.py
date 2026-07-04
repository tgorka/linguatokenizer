"""Command-line entry point for tokenizer-core.

Subcommands are added as BMAD stories land (``train``, ``benchmark``,
``adapt``, ...). For now the CLI exposes ``version`` so the console script and
packaging wiring can be smoke-tested end to end.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from tokenizer_core import __version__


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
    parser.add_subparsers(dest="command", metavar="<command>")
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
