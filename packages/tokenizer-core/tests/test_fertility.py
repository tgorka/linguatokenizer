"""Tests for the fertility / compression metrics."""

from __future__ import annotations

from tokenizer_core.benchmark import count_words, fertility


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
