"""Tests for the CLI's argument-parsing/wiring, incl. the ``benchmark``
subcommand (Story 1.4).

The ``benchmark`` handler's scoring pipeline (``compare_baselines``) is
monkeypatched here so this module tests only the CLI wiring -- argument
parsing, defaults, and that the handler writes both output artifacts -- in
isolation from real HF network access, matching the story's I/O Matrix.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tokenizer_core import cli
from tokenizer_core.benchmark.compare import BaselineOutcome, ComparisonReport
from tokenizer_core.benchmark.fertility import (
    AlignedFertilityReport,
    FertilityReport,
)


def test_build_parser_has_version() -> None:
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--version"])


def test_main_with_no_command_prints_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = cli.main([])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "usage" in captured.out.lower()


def test_benchmark_subcommand_defaults() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["benchmark"])

    assert args.command == "benchmark"
    assert args.recipe == "languages/pl/recipe.yaml"
    assert args.out_dir == "results"
    assert args.data_dir is None
    assert args.func is cli._run_benchmark


def test_benchmark_subcommand_accepts_overrides() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "benchmark",
            "--recipe",
            "languages/xx/recipe.yaml",
            "--out-dir",
            "out",
            "--data-dir",
            "/tmp/data",
        ]
    )

    assert args.recipe == "languages/xx/recipe.yaml"
    assert args.out_dir == "out"
    assert args.data_dir == "/tmp/data"


def _fake_report() -> ComparisonReport:
    return ComparisonReport(
        language_code="pl",
        outcomes=(
            BaselineOutcome(
                name="fake/baseline-a",
                report=AlignedFertilityReport(
                    target=FertilityReport(
                        n_docs=1,
                        n_words=3,
                        n_chars=11,
                        n_tokens=6,
                        fertility=2.0,
                        chars_per_token=11 / 6,
                        bytes_per_token=11 / 6,
                    ),
                    english=FertilityReport(
                        n_docs=1,
                        n_words=4,
                        n_chars=11,
                        n_tokens=4,
                        fertility=1.0,
                        chars_per_token=11 / 4,
                        bytes_per_token=11 / 4,
                    ),
                    fertility_vs_english=2.0,
                ),
                skip_reason=None,
            ),
            BaselineOutcome(
                name="fake/baseline-b",
                report=None,
                skip_reason="failed to load",
            ),
        ),
    )


def test_main_benchmark_writes_json_and_markdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    recipe_path = tmp_path / "recipe.yaml"
    recipe_path.write_text("placeholder: true", encoding="utf-8")
    out_dir = tmp_path / "results"

    fake_recipe = object()
    monkeypatch.setattr(
        cli.Recipe, "load", staticmethod(lambda path: fake_recipe)
    )

    captured_args: dict[str, object] = {}

    def fake_compare_baselines(recipe: object, *, data_dir: object = None):
        captured_args["recipe"] = recipe
        captured_args["data_dir"] = data_dir
        return _fake_report()

    monkeypatch.setattr(cli, "compare_baselines", fake_compare_baselines)

    exit_code = cli.main(
        [
            "benchmark",
            "--recipe",
            str(recipe_path),
            "--out-dir",
            str(out_dir),
        ]
    )

    assert exit_code == 0
    assert captured_args["recipe"] is fake_recipe
    assert captured_args["data_dir"] is None

    json_path = out_dir / "baselines-pl-en.json"
    md_path = out_dir / "baselines-pl-en.md"
    assert json_path.exists()
    assert md_path.exists()
    assert json.loads(json_path.read_text()) == _fake_report().as_dict()

    captured = capsys.readouterr()
    assert "fake/baseline-a" in captured.out
    assert "fertility_vs_english=2.000" in captured.out
    assert "fake/baseline-b" in captured.out
    assert "skipped" in captured.out
