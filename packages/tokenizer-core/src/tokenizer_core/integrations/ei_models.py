"""ei-models integration: reproduce the PL-vs-EN token penalty, then remove it.

The ``ei-models`` research measured that Polish costs ~**2.05×** the tokens of
English (fertility 1.47× × verbosity 1.20× — ~¾ of the penalty is the
*tokenizer*). This module reproduces the **tokenizer** half of that framing —
the fertility penalty on sentence-aligned content — for *any* tokenizer, so an
ei-models experiment can swap in our Polish tokenizer and confirm the penalty
is gone.

Drop-in usage from the ei-models repo::

    from tokenizer_core.integrations.ei_models import measure_penalty, compare

    baseline = measure_penalty(hf_tokenizer, pol_docs, eng_docs)  # ~2.0x
    result = compare(hf_tokenizer, our_tokenizer, pol_docs, eng_docs)
    # result["penalty_removed_pct"] -> how much of the fertility penalty is gone

Any object with ``encode(text) -> ids`` works (Hugging Face fast tokenizers,
``tokenizers.Tokenizer``, tiktoken, a plain callable) — the same contract as
:func:`tokenizer_core.benchmark.fertility.fertility`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from tokenizer_core.benchmark.fertility import (
    AlignedFertilityReport,
    EncodeFn,
    Encoder,
    aligned_fertility,
)

# The ei-models measured total Polish token premium and its tokenizer share,
# for context in reports (see the project research memory).
EI_MODELS_TOTAL_PREMIUM_X = 2.05
EI_MODELS_FERTILITY_PREMIUM_X = 1.47


@dataclass(frozen=True)
class TokenizerPenalty:
    """One tokenizer's target-language fertility penalty vs English.

    Attributes:
        target_fertility: Tokens/word on the target-language corpus.
        english_fertility: Tokens/word on the aligned English corpus.
        penalty_x: ``target_fertility / english_fertility`` — the "target costs
            Nx English" fertility multiplier (ei-models' framing).
    """

    target_fertility: float
    english_fertility: float
    penalty_x: float

    def as_dict(self) -> dict[str, float]:
        """Return the penalty as a plain JSON-serializable dict."""
        return {
            "target_fertility": self.target_fertility,
            "english_fertility": self.english_fertility,
            "penalty_x": self.penalty_x,
        }


def _from_aligned(report: AlignedFertilityReport) -> TokenizerPenalty:
    return TokenizerPenalty(
        target_fertility=report.target.fertility,
        english_fertility=report.english.fertility,
        penalty_x=report.fertility_vs_english,
    )


def measure_penalty(
    encoder: Encoder | EncodeFn,
    target_docs: Iterable[str],
    english_docs: Iterable[str],
) -> TokenizerPenalty:
    """Measure one tokenizer's target-vs-English fertility penalty.

    Args:
        encoder: The tokenizer to score (``encode``-capable or a callable).
        target_docs: Target-language documents (e.g. Polish FLORES).
        english_docs: The sentence-aligned English documents (same length).

    Returns:
        A :class:`TokenizerPenalty`.
    """
    return _from_aligned(aligned_fertility(encoder, target_docs, english_docs))


def compare(
    baseline_encoder: Encoder | EncodeFn,
    optimized_encoder: Encoder | EncodeFn,
    target_docs: Iterable[str],
    english_docs: Iterable[str],
) -> dict[str, object]:
    """Compare a baseline tokenizer to our optimized one on the penalty.

    Args:
        baseline_encoder: The model's current tokenizer (the ~2× penalty).
        optimized_encoder: Our target-language tokenizer.
        target_docs: Target-language documents.
        english_docs: Aligned English documents.

    Returns:
        A dict with ``baseline`` and ``optimized`` :class:`TokenizerPenalty`
        dicts, the ``penalty_x`` of each, and ``penalty_removed_pct`` — how much
        of the *excess* fertility penalty (above English parity) the optimized
        tokenizer eliminates. ``100`` means Polish now costs the same as English
        or less; ``0`` means no improvement.
    """
    target_docs = list(target_docs)
    english_docs = list(english_docs)
    base = measure_penalty(baseline_encoder, target_docs, english_docs)
    opt = measure_penalty(optimized_encoder, target_docs, english_docs)

    base_excess = base.penalty_x - 1.0
    opt_excess = opt.penalty_x - 1.0
    if base_excess <= 0:
        removed_pct = 0.0
    else:
        removed_pct = max(0.0, (1.0 - opt_excess / base_excess)) * 100.0

    return {
        "baseline": base.as_dict(),
        "optimized": opt.as_dict(),
        "baseline_penalty_x": base.penalty_x,
        "optimized_penalty_x": opt.penalty_x,
        "penalty_removed_pct": removed_pct,
    }
