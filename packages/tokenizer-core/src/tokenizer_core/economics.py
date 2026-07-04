"""Inference-cost economics of tokenizer fertility — the datacenter optimizer.

A lower-fertility tokenizer produces fewer tokens for the same text, and almost
every inference cost scales with token count. This module turns a fertility
improvement into the concrete wins a datacenter cares about:

- **API / compute cost** scales *linearly* with tokens, so a tokenizer that
  cuts tokens by ``r`` cuts spend by ``r`` (both the input and output legs).
- **Throughput / serving capacity** scales *inversely*: ``fertility_old /
  fertility_new`` more requests-per-GPU for the same traffic (first order;
  higher in practice from smaller KV cache + better batch packing).
- **Effective context** scales inversely too: a fixed ``K``-token window holds
  ``K / fertility`` words, so halving fertility ≈ doubles usable context.
- **KV-cache memory** per request scales linearly with sequence length, so it
  drops by the same ``r`` — which is what unlocks the extra batch capacity.

Grounding (see ``docs/datacenter-optimizer.md`` for citations): output tokens
cost ~3-5× input (decode is sequential/memory-bound); billing is strictly
linear ``cost = in/1e6·P_in + out/1e6·P_out``. Frontier prices move fast — pass
live ``Pricing`` at call time rather than trusting the presets here.

Pure-python, model-free: it reasons about token *counts*, so it works from a
fertility number alone (e.g. our FLORES Polish measurements).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Pricing:
    """Per-1M-token price for a served model (USD)."""

    name: str
    input_usd_per_1m: float
    output_usd_per_1m: float


# Representative presets (2025/2026, USD/1M tokens). Indicative only — pin to
# the provider's live pricing page at runtime; frontier prices move monthly.
PRICING_PRESETS: dict[str, Pricing] = {
    "gpt-4o": Pricing("gpt-4o", 2.50, 10.00),
    "gpt-4.1": Pricing("gpt-4.1", 2.00, 8.00),
    "claude-sonnet": Pricing("claude-sonnet", 3.00, 15.00),
    "claude-haiku": Pricing("claude-haiku", 0.25, 1.25),
    # Open-weight, hosted.
    "llama-70b-together": Pricing("llama-70b-together", 0.88, 0.88),
    "llama-70b-groq": Pricing("llama-70b-groq", 0.59, 0.79),
    # Self-hosted 70B on 8xH100 (~$2.30/1M, symmetric) — see the research brief.
    "selfhost-70b-h100": Pricing("selfhost-70b-h100", 2.30, 2.30),
}


@dataclass(frozen=True)
class SavingsReport:
    """The wins from moving a workload to a lower-fertility tokenizer.

    Attributes:
        baseline_fertility: Tokens/word of the current tokenizer.
        optimized_fertility: Tokens/word of the optimized tokenizer.
        token_reduction_pct: Percentage fewer tokens for the same content.
        monthly_cost_before: USD/month at baseline fertility.
        monthly_cost_after: USD/month at optimized fertility.
        monthly_savings: ``before - after`` USD/month.
        savings_pct: Percentage cost reduction.
        throughput_gain_x: Requests-per-GPU multiplier (first order).
        effective_context_gain_x: Usable-context multiplier for a fixed window.
        kv_memory_reduction_pct: Per-request KV-cache memory reduction.
    """

    baseline_fertility: float
    optimized_fertility: float
    token_reduction_pct: float
    monthly_cost_before: float
    monthly_cost_after: float
    monthly_savings: float
    savings_pct: float
    throughput_gain_x: float
    effective_context_gain_x: float
    kv_memory_reduction_pct: float

    def as_dict(self) -> dict[str, float]:
        """Return the report as a plain JSON-serializable dict."""
        return {
            "baseline_fertility": self.baseline_fertility,
            "optimized_fertility": self.optimized_fertility,
            "token_reduction_pct": self.token_reduction_pct,
            "monthly_cost_before": self.monthly_cost_before,
            "monthly_cost_after": self.monthly_cost_after,
            "monthly_savings": self.monthly_savings,
            "savings_pct": self.savings_pct,
            "throughput_gain_x": self.throughput_gain_x,
            "effective_context_gain_x": self.effective_context_gain_x,
            "kv_memory_reduction_pct": self.kv_memory_reduction_pct,
        }


def _cost_usd(words: float, fertility: float, price_per_1m: float) -> float:
    """USD to serve ``words`` at ``fertility`` tokens/word at ``price``/1M."""
    return words * fertility / 1_000_000.0 * price_per_1m


def estimate_savings(
    baseline_fertility: float,
    optimized_fertility: float,
    *,
    pricing: Pricing,
    input_words_per_month: float,
    output_words_per_month: float,
) -> SavingsReport:
    """Estimate the datacenter wins of a fertility improvement on a workload.

    Args:
        baseline_fertility: Current tokenizer tokens/word (target language).
        optimized_fertility: Optimized tokenizer tokens/word.
        pricing: Per-1M-token price (input + output legs).
        input_words_per_month: Prompt words served per month (target language).
        output_words_per_month: Generated words per month (target language).

    Returns:
        A :class:`SavingsReport`.

    Raises:
        ValueError: If either fertility is not positive.
    """
    if baseline_fertility <= 0 or optimized_fertility <= 0:
        raise ValueError("fertilities must be positive")

    before = _cost_usd(
        input_words_per_month, baseline_fertility, pricing.input_usd_per_1m
    ) + _cost_usd(
        output_words_per_month, baseline_fertility, pricing.output_usd_per_1m
    )
    after = _cost_usd(
        input_words_per_month, optimized_fertility, pricing.input_usd_per_1m
    ) + _cost_usd(
        output_words_per_month, optimized_fertility, pricing.output_usd_per_1m
    )

    ratio = optimized_fertility / baseline_fertility  # < 1 when improved
    return SavingsReport(
        baseline_fertility=baseline_fertility,
        optimized_fertility=optimized_fertility,
        token_reduction_pct=(1.0 - ratio) * 100.0,
        monthly_cost_before=before,
        monthly_cost_after=after,
        monthly_savings=before - after,
        savings_pct=(1.0 - ratio) * 100.0,
        throughput_gain_x=baseline_fertility / optimized_fertility,
        effective_context_gain_x=baseline_fertility / optimized_fertility,
        kv_memory_reduction_pct=(1.0 - ratio) * 100.0,
    )
