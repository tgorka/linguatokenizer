"""Tests for the quality gates (``benchmark.quality``)."""

from __future__ import annotations

import math

from tokenizer_core.benchmark.quality import (
    base_language_preservation,
    bits_per_byte,
    quality_gate,
)


def test_bits_per_byte_basic() -> None:
    # nll of ln(2) nats over 1 byte == exactly 1 bit/byte.
    assert bits_per_byte(math.log(2), 1) == 1.0
    assert bits_per_byte(0.0, 100) == 0.0


def test_bits_per_byte_zero_bytes_is_zero_not_error() -> None:
    assert bits_per_byte(5.0, 0) == 0.0


def test_quality_gate_passes_within_budget() -> None:
    report = quality_gate(1.000, 1.005, budget_pct=1.0)  # +0.5%
    assert report.passed is True
    assert round(report.delta_pct, 6) == 0.5


def test_quality_gate_fails_over_budget() -> None:
    report = quality_gate(1.000, 1.020, budget_pct=1.0)  # +2%
    assert report.passed is False
    assert round(report.delta_pct, 3) == 2.0


def test_quality_gate_improvement_passes() -> None:
    # Quality improved (bpb dropped) -> negative delta, always passes.
    report = quality_gate(1.000, 0.950, budget_pct=1.0)
    assert report.passed is True
    assert report.delta_pct < 0


def test_base_language_preservation_all_unchanged() -> None:
    base = [[1, 2, 3], [4, 5]]
    adapted = [[1, 2, 3], [4, 5]]  # identical, all ids < 100
    assert base_language_preservation(base, adapted, base_vocab_size=100) == 1.0


def test_base_language_preservation_counts_shifted_tokens() -> None:
    # Second doc diverges at position 1 (a new id >= base_vocab_size).
    base = [[1, 2, 3], [4, 5, 6]]
    adapted = [[1, 2, 3], [4, 999, 6]]
    # doc1: 3/3 preserved; doc2: prefix match stops at index 1 -> 1 preserved.
    # total base tokens = 6, preserved = 4.
    frac = base_language_preservation(base, adapted, base_vocab_size=100)
    assert round(frac, 4) == round(4 / 6, 4)


def test_base_language_preservation_empty_is_one() -> None:
    assert base_language_preservation([], [], base_vocab_size=100) == 1.0
