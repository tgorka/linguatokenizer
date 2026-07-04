"""Tests for the inference-cost economics calculator."""

from __future__ import annotations

import pytest

from tokenizer_core.economics import (
    PRICING_PRESETS,
    Pricing,
    estimate_savings,
)


def test_cost_scales_linearly_with_fertility() -> None:
    # Halving fertility halves cost, doubles throughput and context.
    pricing = Pricing("t", input_usd_per_1m=2.0, output_usd_per_1m=10.0)
    r = estimate_savings(
        2.0,
        1.0,
        pricing=pricing,
        input_words_per_month=1_000_000,
        output_words_per_month=200_000,
    )
    assert r.token_reduction_pct == 50.0
    assert r.savings_pct == 50.0
    assert r.monthly_cost_after == pytest.approx(r.monthly_cost_before / 2)
    assert r.throughput_gain_x == pytest.approx(2.0)
    assert r.effective_context_gain_x == pytest.approx(2.0)
    assert r.kv_memory_reduction_pct == 50.0


def test_absolute_cost_uses_both_legs() -> None:
    pricing = Pricing("t", input_usd_per_1m=3.0, output_usd_per_1m=15.0)
    r = estimate_savings(
        2.0,
        2.0,  # no improvement -> before == after, savings 0
        pricing=pricing,
        input_words_per_month=1_000_000,
        output_words_per_month=1_000_000,
    )
    # input: 1e6 words * 2 tok/word = 2e6 tokens -> 2 * $3 = $6
    # output: 1e6 words * 2 tok/word = 2e6 tokens -> 2 * $15 = $30
    assert r.monthly_cost_before == pytest.approx(36.0)
    assert r.monthly_savings == 0.0


def test_polish_realistic_case() -> None:
    # ~2.1x -> ~1.14x Polish (our 64k result) on Claude Sonnet.
    r = estimate_savings(
        2.68,  # Llama-3.1 Polish tok/word (baseline)
        1.42,  # our BPE-64k Polish tok/word
        pricing=PRICING_PRESETS["claude-sonnet"],
        input_words_per_month=1_000_000,
        output_words_per_month=300_000,
    )
    assert 45 < r.savings_pct < 48  # ~47% fewer tokens
    assert r.throughput_gain_x > 1.8
    assert r.monthly_savings > 0


def test_nonpositive_fertility_raises() -> None:
    with pytest.raises(ValueError, match="positive"):
        estimate_savings(
            0.0,
            1.0,
            pricing=PRICING_PRESETS["gpt-4o"],
            input_words_per_month=1,
            output_words_per_month=1,
        )
