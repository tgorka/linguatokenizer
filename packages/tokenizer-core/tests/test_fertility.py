"""Tests for the fertility / compression metrics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tokenizer_core.benchmark import (
    aligned_fertility,
    count_words,
    fertility,
    write_report,
)


class _WhitespaceEncoder:
    """Trivial encoder: one token per whitespace-separated chunk."""

    def encode(self, text: str) -> list[str]:
        return text.split()


class _CharEncoder:
    """Trivial encoder: one token per character (worst-case fertility)."""

    def encode(self, text: str) -> list[str]:
        return list(text)


def test_count_words_handles_polish_diacritics() -> None:
    # Five words, Polish diacritics must not split words.
    assert count_words("Zażółć gęślą jaźń na wiosnę") == 5


def test_count_words_keeps_internal_apostrophes_and_hyphens() -> None:
    assert count_words("d'Artagnan czarno-biały") == 2


def test_fertility_whitespace_encoder_is_one_token_per_word() -> None:
    corpus = ["ala ma kota", "kot ma ale"]
    report = fertility(_WhitespaceEncoder(), corpus)
    assert report.n_docs == 2
    assert report.n_words == 6
    assert report.n_tokens == 6
    assert report.fertility == 1.0


def test_fertility_char_encoder_matches_char_count() -> None:
    corpus = ["abc def"]
    report = fertility(_CharEncoder(), corpus)
    # 7 characters including the space -> 7 tokens, 2 words.
    assert report.n_tokens == 7
    assert report.fertility == 3.5


def test_fertility_accepts_plain_callable() -> None:
    report = fertility(lambda t: t.split(), ["one two three"])
    assert report.fertility == 1.0


def test_fertility_empty_corpus_is_zero_not_error() -> None:
    report = fertility(_WhitespaceEncoder(), [])
    assert report.fertility == 0.0
    assert report.chars_per_token == 0.0
    assert report.bytes_per_token == 0.0


def test_bytes_per_token_reflects_utf8_width() -> None:
    # "ą" is 2 UTF-8 bytes; one char-token -> 2 bytes/token.
    report = fertility(_CharEncoder(), ["ą"])
    assert report.n_tokens == 1
    assert report.bytes_per_token == 2.0


def test_report_as_dict_roundtrips_keys() -> None:
    report = fertility(_WhitespaceEncoder(), ["ala ma kota"])
    d = report.as_dict()
    assert d["fertility"] == 1.0
    assert set(d) == {
        "n_docs",
        "n_words",
        "n_chars",
        "n_tokens",
        "fertility",
        "chars_per_token",
        "bytes_per_token",
    }


def test_aligned_fertility_equal_length_computes_ratio() -> None:
    target = ["ala ma kota", "kot ma ale"]
    english = ["a has a cat", "the cat has an a"]
    report = aligned_fertility(_WhitespaceEncoder(), target, english)

    assert report.target == fertility(_WhitespaceEncoder(), target)
    assert report.english == fertility(_WhitespaceEncoder(), english)
    assert report.fertility_vs_english == (
        report.target.fertility / report.english.fertility
    )


def test_aligned_fertility_mismatched_length_raises_naming_both_counts() -> (
    None
):
    target = ["a", "b", "c"]
    english = ["x", "y"]

    with pytest.raises(ValueError, match=r"target=3.*english=2"):
        aligned_fertility(_WhitespaceEncoder(), target, english)


def test_aligned_fertility_empty_english_corpus_ratio_is_zero() -> None:
    report = aligned_fertility(_WhitespaceEncoder(), [], [])
    assert report.fertility_vs_english == 0.0


def test_aligned_fertility_does_not_encode_before_length_check() -> None:
    class _ExplodingEncoder:
        def encode(self, text: str) -> list[str]:
            raise AssertionError("encoder must not be called on mismatch")

    with pytest.raises(ValueError, match=r"target=1.*english=0"):
        aligned_fertility(_ExplodingEncoder(), ["a"], [])


def test_aligned_fertility_does_not_swap_target_and_english() -> None:
    # Asymmetric fertility per side (char- vs whitespace-tokenized) so a
    # target/english swap inside aligned_fertility would flip the ratio and
    # fail this assertion, unlike same-shape fixtures.
    target = ["abc"]  # char encoder -> 3 tokens, 1 word -> fertility 3.0
    english = ["a b c d"]  # whitespace encoder -> 4 tokens, 4 words -> 1.0

    def encoder(text: str) -> list[str]:
        return list(text) if text == "abc" else text.split()

    report = aligned_fertility(encoder, target, english)

    assert report.target.fertility == 3.0
    assert report.english.fertility == 1.0
    assert report.fertility_vs_english == 3.0


def test_aligned_fertility_passes_normalize_through_to_both_sides() -> None:
    # NFD decomposes "ą" into "a" + combining ogonek (2 chars) instead of 1.
    target = ["ą"]
    english = ["a"]

    report = aligned_fertility(_CharEncoder(), target, english, normalize="NFD")

    assert report.target.n_chars == 2
    assert report.english.n_chars == 1


def test_write_report_creates_nested_directories(tmp_path) -> None:
    report = fertility(_WhitespaceEncoder(), ["ala ma kota"])
    dest = tmp_path / "results" / "tmp" / "x" / "report.json"

    returned = write_report(report, dest)

    assert returned == dest
    assert dest.exists()


def test_write_report_roundtrips_fertility_report(tmp_path) -> None:
    report = fertility(_WhitespaceEncoder(), ["ala ma kota"])
    dest = tmp_path / "report.json"

    write_report(report, dest)

    assert json.loads(dest.read_text()) == report.as_dict()


def test_write_report_roundtrips_aligned_fertility_report(tmp_path) -> None:
    report = aligned_fertility(
        _WhitespaceEncoder(), ["ala ma kota"], ["a has a cat"]
    )
    dest = tmp_path / "aligned.json"

    write_report(report, dest)

    assert json.loads(dest.read_text()) == report.as_dict()


def test_write_report_accepts_str_path(tmp_path) -> None:
    report = fertility(_WhitespaceEncoder(), ["ala ma kota"])
    dest = str(tmp_path / "report.json")

    returned = write_report(report, dest)

    assert returned == Path(dest)
    assert json.loads(Path(dest).read_text()) == report.as_dict()
