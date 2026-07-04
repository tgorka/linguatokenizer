"""Tests for the baseline comparison table (``benchmark.compare``).

The ``transformers``-backed baseline loading is exercised via a fake module
injected into ``sys.modules`` (mirroring ``test_sources.py``'s pattern for
``datasets``), so these tests never need the real ``adapters``/``corpora``
extras or network access, per the repo's CI (``uv sync --all-packages``, no
extras). FLORES resolution goes through the real ``resolve_source``, backed
by a fake ``datasets`` module the same way ``test_sources.py`` does it.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from tokenizer_core.benchmark.compare import (
    BaselineOutcome,
    CompareError,
    ComparisonReport,
    _find_aligned_pair,
    compare_baselines,
    write_comparison_markdown,
)
from tokenizer_core.benchmark.fertility import (
    AlignedFertilityReport,
    FertilityReport,
    write_report,
)
from tokenizer_core.train.config import Recipe


def _recipe(
    *,
    baselines: list[str] | None = None,
    eval_corpora: list[dict] | None = None,
) -> Recipe:
    data = {
        "language": {"code": "pl", "name": "Polish", "scripts": ["Latn"]},
        "train_corpora": [
            {
                "source": "hf",
                "id": "some/train",
                "config": "pl",
                "split": "train",
            }
        ],
        "eval_corpora": eval_corpora
        if eval_corpora is not None
        else [
            {
                "source": "hf",
                "id": "facebook/flores",
                "config": "pol_Latn",
                "split": "devtest",
            }
        ],
        "candidates": [
            {"name": "pl-bpe-32k", "algorithm": "bpe", "vocab_size": 32000}
        ],
        "targets": {
            "max_fertility_vs_english": 1.0,
            "max_quality_drop_pct": 1.0,
        },
        "baselines": baselines
        if baselines is not None
        else ["fake/baseline-a"],
    }
    return Recipe.model_validate(data)


def _fake_datasets_module(by_config: dict[str, list[dict[str, object]]]):
    """Fake ``datasets`` module whose ``load_dataset`` dispatches on config."""

    def load_dataset(
        path: str,
        name: str | None = None,
        split: str | None = None,
        streaming: bool = True,
    ) -> list[dict[str, object]]:
        return list(by_config.get(name, []))

    return types.SimpleNamespace(load_dataset=load_dataset)


class _WordCountTokenizer:
    """Fake HF fast tokenizer: one token per whitespace-separated word."""

    def encode(self, text: str) -> list[str]:
        return text.split()


def _fake_transformers_module(
    tokenizers_by_name: dict[str, object] | None = None,
    error_names: set[str] | None = None,
):
    """Fake ``transformers`` module for ``sys.modules`` injection."""
    tokenizers_by_name = tokenizers_by_name or {}
    error_names = error_names or set()

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(name: str, token: str | None = None) -> object:
            if name in error_names:
                raise RuntimeError(f"gated repo {name!r}, terms not accepted")
            return tokenizers_by_name.get(name, _WordCountTokenizer())

    return types.SimpleNamespace(AutoTokenizer=_FakeAutoTokenizer)


# --- compare_baselines: all load fine --------------------------------------


def test_compare_baselines_all_load_scores_every_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        _fake_datasets_module(
            {
                "pol_Latn": [{"text": "ala ma kota"}, {"text": "kot ma ale"}],
                "eng_Latn": [
                    {"text": "a has a cat"},
                    {"text": "the cat has a"},
                ],
            }
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        _fake_transformers_module(),
    )

    recipe = _recipe(baselines=["fake/baseline-a", "fake/baseline-b"])
    report = compare_baselines(recipe)

    assert isinstance(report, ComparisonReport)
    assert report.language_code == "pl"
    assert len(report.outcomes) == 2
    for outcome in report.outcomes:
        assert isinstance(outcome, BaselineOutcome)
        assert outcome.skip_reason is None
        assert isinstance(outcome.report, AlignedFertilityReport)


# --- compare_baselines: one baseline fails to load -------------------------


def test_compare_baselines_one_failure_still_scores_the_rest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        _fake_datasets_module(
            {
                "pol_Latn": [{"text": "ala ma kota"}],
                "eng_Latn": [{"text": "a has a cat"}],
            }
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        _fake_transformers_module(error_names={"fake/gated-baseline"}),
    )

    recipe = _recipe(baselines=["fake/gated-baseline", "fake/baseline-b"])
    report = compare_baselines(recipe)

    assert len(report.outcomes) == 2
    failed, ok = report.outcomes
    assert failed.name == "fake/gated-baseline"
    assert failed.report is None
    assert failed.skip_reason is not None
    assert "fake/gated-baseline" in failed.skip_reason

    assert ok.name == "fake/baseline-b"
    assert ok.skip_reason is None
    assert isinstance(ok.report, AlignedFertilityReport)


# --- compare_baselines: missing FLORES eval corpus -------------------------


def test_compare_baselines_missing_flores_corpus_raises_compare_error() -> None:
    recipe = _recipe(
        eval_corpora=[
            {
                "source": "hf",
                "id": "wikimedia/wikipedia",
                "config": "20231101.pl",
                "split": "test",
            }
        ]
    )

    with pytest.raises(CompareError, match="facebook/flores"):
        compare_baselines(recipe)


# --- compare_baselines: FLORES pair length mismatch ------------------------


def test_compare_baselines_length_mismatch_propagates_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        _fake_datasets_module(
            {
                "pol_Latn": [
                    {"text": "a"},
                    {"text": "b"},
                    {"text": "c"},
                ],
                "eng_Latn": [{"text": "x"}, {"text": "y"}],
            }
        ),
    )
    monkeypatch.setitem(
        sys.modules, "transformers", _fake_transformers_module()
    )

    recipe = _recipe(baselines=["fake/baseline-a"])

    with pytest.raises(ValueError, match=r"target=3.*english=2"):
        compare_baselines(recipe)


# --- report round-trip ------------------------------------------------------


def test_comparison_report_write_report_round_trips(tmp_path: Path) -> None:
    report = ComparisonReport(
        language_code="pl",
        outcomes=(
            BaselineOutcome(
                name="fake/baseline-a",
                report=AlignedFertilityReport(
                    target=FertilityReport(
                        n_docs=1,
                        n_words=3,
                        n_chars=11,
                        n_tokens=3,
                        fertility=1.0,
                        chars_per_token=11 / 3,
                        bytes_per_token=11 / 3,
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
                    fertility_vs_english=1.0,
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
    dest = tmp_path / "baselines-pl-en.json"

    write_report(report, dest)

    assert json.loads(dest.read_text()) == report.as_dict()


# --- markdown summary --------------------------------------------------------


def test_write_comparison_markdown_shows_scores_and_skip_reason(
    tmp_path: Path,
) -> None:
    report = ComparisonReport(
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
                skip_reason="failed to load baseline 'fake/baseline-b': boom",
            ),
        ),
    )
    dest = tmp_path / "baselines-pl-en.md"

    returned = write_comparison_markdown(report, dest)

    assert returned == dest
    text = dest.read_text()
    assert "fake/baseline-a" in text
    assert "2.000" in text  # fertility_vs_english for the scored baseline
    assert "fake/baseline-b" in text
    assert "failed to load baseline 'fake/baseline-b': boom" in text


def test_write_comparison_markdown_creates_nested_directories(
    tmp_path: Path,
) -> None:
    report = ComparisonReport(language_code="pl", outcomes=())
    dest = tmp_path / "nested" / "dir" / "baselines-pl-en.md"

    write_comparison_markdown(report, dest)

    assert dest.exists()


# --- compare_baselines: ungated local aligned FLORES pair ------------------


def test_compare_baselines_uses_local_aligned_pair(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The Polish recipe points at ungated FLORES-200 files on the data plane
    # (config names the FLORES side); compare must pair pol_Latn<->eng_Latn
    # by config without any hf/facebook/flores entry.
    pol = tmp_path / "pol_Latn.devtest"
    eng = tmp_path / "eng_Latn.devtest"
    pol.write_text("ala ma kota\nkot ma ale\n", encoding="utf-8")
    eng.write_text("a has a cat\nthe cat has a\n", encoding="utf-8")

    monkeypatch.setitem(
        sys.modules, "transformers", _fake_transformers_module()
    )
    recipe = _recipe(
        baselines=["fake/baseline-a"],
        eval_corpora=[
            {
                "source": "local",
                "id": str(pol),
                "config": "pol_Latn",
                "split": "devtest",
            },
            {
                "source": "local",
                "id": str(eng),
                "config": "eng_Latn",
                "split": "devtest",
            },
        ],
    )

    report = compare_baselines(recipe)

    assert len(report.outcomes) == 1
    outcome = report.outcomes[0]
    assert outcome.report is not None
    # pol: 6 tokens / 6 words = 1.0 ; eng: 7 tokens / 7 words = 1.0 -> ratio 1.0
    assert outcome.report.fertility_vs_english == 1.0


def test_compare_baselines_no_aligned_pair_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules, "transformers", _fake_transformers_module()
    )
    recipe = _recipe(
        eval_corpora=[
            {
                "source": "hf",
                "id": "wikimedia/wikipedia",
                "config": "20231101.pl",
                "split": "test",
            }
        ]
    )
    with pytest.raises(CompareError, match="no aligned eval pair"):
        compare_baselines(recipe)


# --- aligned-pair detection across scripts (multi-language) -----------------


def test_find_aligned_pair_non_latin_script() -> None:
    # Russian (rus_Cyrl) target + English must be found — not just *_Latn.
    recipe = _recipe(
        eval_corpora=[
            {
                "source": "local",
                "id": "/data/rus.devtest",
                "config": "rus_Cyrl",
                "split": "devtest",
            },
            {
                "source": "local",
                "id": "/data/eng.devtest",
                "config": "eng_Latn",
                "split": "devtest",
            },
        ]
    )
    target, english = _find_aligned_pair(recipe)
    assert target.config == "rus_Cyrl"
    assert english.config == "eng_Latn"


def test_find_aligned_pair_ignores_non_flores_eval_entries() -> None:
    # A wikipedia eval entry (config 20231101.de) must not be mistaken for the
    # FLORES target; the deu_Latn FLORES entry is picked instead.
    recipe = _recipe(
        eval_corpora=[
            {
                "source": "local",
                "id": "/data/deu.devtest",
                "config": "deu_Latn",
                "split": "devtest",
            },
            {
                "source": "local",
                "id": "/data/eng.devtest",
                "config": "eng_Latn",
                "split": "devtest",
            },
            {
                "source": "hf",
                "id": "wikimedia/wikipedia",
                "config": "20231101.de",
                "split": "train",
            },
        ]
    )
    target, english = _find_aligned_pair(recipe)
    assert target.config == "deu_Latn"
    assert english.config == "eng_Latn"
