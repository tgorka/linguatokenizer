"""Tests for the CLI's argument-parsing/wiring, incl. the ``benchmark``
(Story 1.4) and ``train`` (Story 2.1) subcommands.

The ``benchmark``/``train`` handlers' underlying pipelines
(``compare_baselines``/``train_candidate``/``save_artifact``) are
monkeypatched here so this module tests only the CLI wiring -- argument
parsing, defaults, and skip-and-continue behavior -- in isolation from real
HF network access or the native ``tokenizers`` dependency, matching each
story's I/O Matrix.
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
from tokenizer_core.train.artifact import TrainerMetadata
from tokenizer_core.train.config import (
    Candidate,
    CorpusSpec,
    Language,
    Recipe,
    Targets,
)
from tokenizer_core.train.trainers import TrainerError


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


# --- train subcommand (Story 2.1) ------------------------------------------


def test_train_subcommand_defaults() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["train"])

    assert args.command == "train"
    assert args.recipe == "languages/pl/recipe.yaml"
    assert args.data_dir is None
    assert args.n_docs == 10_000
    assert args.seed == 42
    assert args.func is cli._run_train


def test_train_subcommand_accepts_overrides() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "train",
            "--recipe",
            "languages/xx/recipe.yaml",
            "--data-dir",
            "/tmp/data",
            "--n-docs",
            "1000",
            "--seed",
            "1",
        ]
    )

    assert args.recipe == "languages/xx/recipe.yaml"
    assert args.data_dir == "/tmp/data"
    assert args.n_docs == 1000
    assert args.seed == 1


def _fake_recipe() -> Recipe:
    corpus = CorpusSpec(source="local", id="unused.txt", weight=1.0)
    return Recipe(
        language=Language(code="pl", name="Polish", scripts=["Latn"]),
        train_corpora=[corpus],
        eval_corpora=[corpus],
        candidates=[
            Candidate(name="pl-bpe-32k", algorithm="bpe", vocab_size=100),
            Candidate(
                name="pl-unigram-32k", algorithm="unigram", vocab_size=100
            ),
            Candidate(
                name="pl-superbpe-32k", algorithm="superbpe", vocab_size=100
            ),
        ],
        targets=Targets(max_fertility_vs_english=1.0, max_quality_drop_pct=1.0),
        baselines=["gpt2"],
    )


def test_main_train_trains_saves_and_skips_unsupported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    recipe_path = tmp_path / "recipe.yaml"
    recipe_path.write_text("placeholder: true", encoding="utf-8")
    data_dir = tmp_path / "data"

    fake_recipe = _fake_recipe()
    monkeypatch.setattr(
        cli.Recipe, "load", staticmethod(lambda path: fake_recipe)
    )

    trained_calls: list[str] = []

    def fake_train_candidate(recipe, candidate, *, n_docs, seed, data_dir):
        if candidate.algorithm not in ("bpe", "unigram"):
            raise TrainerError(
                f"candidate {candidate.name!r} has unsupported algorithm "
                f"{candidate.algorithm!r}"
            )
        trained_calls.append(candidate.name)
        metadata = TrainerMetadata(
            name=candidate.name,
            algorithm=candidate.algorithm,
            vocab_size=candidate.vocab_size,
            byte_fallback=candidate.byte_fallback,
            normalization=candidate.normalization,
            corpus_hash=f"hash-{candidate.name}",
            seed=seed,
            language_code=recipe.language.code,
        )
        return object(), metadata

    saved_dirs: list[Path] = []

    def fake_save_artifact(tokenizer, metadata, out_dir):
        saved_dirs.append(Path(out_dir))
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        return out_path / "tokenizer.json", out_path / "metadata.json"

    monkeypatch.setattr(cli, "train_candidate", fake_train_candidate)
    monkeypatch.setattr(cli, "save_artifact", fake_save_artifact)

    exit_code = cli.main(
        [
            "train",
            "--recipe",
            str(recipe_path),
            "--data-dir",
            str(data_dir),
        ]
    )

    assert exit_code == 0
    assert trained_calls == ["pl-bpe-32k", "pl-unigram-32k"]
    assert saved_dirs == [
        data_dir / "tokenizers" / "pl" / "pl-bpe-32k",
        data_dir / "tokenizers" / "pl" / "pl-unigram-32k",
    ]

    captured = capsys.readouterr()
    assert "pl-bpe-32k: trained" in captured.out
    assert "hash-pl-bpe-32k" in captured.out
    assert "pl-unigram-32k: trained" in captured.out
    assert "pl-superbpe-32k: skipped" in captured.out
    assert "superbpe" in captured.out
