"""Tests for encode-speed timing."""

from __future__ import annotations

import pytest

from tokenizer_core.benchmark.timing import encode_timing


class _WhitespaceEncoder:
    def encode(self, text: str) -> list[str]:
        return text.split()


def test_timing_counts_docs_and_tokens() -> None:
    report = encode_timing(_WhitespaceEncoder(), ["a b c", "d e"], repeats=2)
    assert report.n_docs == 2
    assert report.n_tokens == 5
    assert report.seconds >= 0.0
    # tokens_per_sec is consistent with n_tokens / seconds.
    if report.seconds > 0:
        assert report.tokens_per_sec == pytest.approx(
            report.n_tokens / report.seconds
        )
        assert report.ms_per_doc == pytest.approx(
            report.seconds / report.n_docs * 1000
        )


def test_timing_empty_corpus_is_zero() -> None:
    report = encode_timing(_WhitespaceEncoder(), [])
    assert report.n_docs == 0
    assert report.tokens_per_sec == 0.0
    assert report.ms_per_doc == 0.0


def test_timing_accepts_callable() -> None:
    report = encode_timing(lambda t: t.split(), ["one two three"])
    assert report.n_tokens == 3


def test_timing_rejects_zero_repeats() -> None:
    with pytest.raises(ValueError, match="repeats"):
        encode_timing(_WhitespaceEncoder(), ["a"], repeats=0)
