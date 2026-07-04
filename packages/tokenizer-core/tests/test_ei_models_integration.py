"""Tests for the ei-models penalty integration."""

from __future__ import annotations

from tokenizer_core.integrations.ei_models import compare, measure_penalty


class _CharEncoder:
    """One token per character (worst-case fertility)."""

    def encode(self, text: str) -> list[str]:
        return list(text)


class _WordEncoder:
    """One token per whitespace word (fertility ~1.0)."""

    def encode(self, text: str) -> list[str]:
        return text.split()


def test_measure_penalty_char_encoder() -> None:
    # Polish words longer than English -> higher char-fertility -> penalty > 1.
    pol = ["kotki pieski"]  # 12 chars, 2 words -> 6 tok/word
    eng = ["cat dog"]  # 7 chars, 2 words -> 3.5 tok/word
    p = measure_penalty(_CharEncoder(), pol, eng)
    assert p.target_fertility == 6.0
    assert p.english_fertility == 3.5
    assert p.penalty_x == 6.0 / 3.5


def test_compare_reports_penalty_removed() -> None:
    pol = ["kotki pieski", "domy koty"]
    eng = ["cat dog", "home cats"]
    # baseline over-segments (char), optimized is word-level (penalty 1.0).
    result = compare(_CharEncoder(), _WordEncoder(), pol, eng)
    assert result["baseline_penalty_x"] > 1.0
    assert result["optimized_penalty_x"] == 1.0
    # optimized has penalty 1.0 (parity) -> 100% of excess removed.
    assert result["penalty_removed_pct"] == 100.0


def test_compare_no_baseline_penalty_is_zero_removed() -> None:
    pol = ["a b"]
    eng = ["a b"]
    result = compare(_WordEncoder(), _WordEncoder(), pol, eng)
    assert result["penalty_removed_pct"] == 0.0
