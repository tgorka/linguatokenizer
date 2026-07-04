"""Quality gates for tokenizer transplants — is the swap safe to adopt?

Two complementary, CPU-cheap checks:

- **bits-per-byte (bpb)**: the model's cross-entropy normalized per UTF-8 byte
  rather than per token. Because it is per *byte*, it is directly comparable
  across tokenizers (a model that uses fewer, bigger tokens is not unfairly
  advantaged), which is exactly what you need to ask "did replacing the
  tokenizer change the model's language modeling quality?". Computing it needs
  a model runner (torch); :func:`bits_per_byte` / :func:`quality_gate` take the
  already-summed loss so the metric + gate stay dependency-free and unit-tested,
  and a runner plugs in on top.

- **base-language preservation**: a model-free proxy usable *right now*. The
  extend adapter keeps every base token id, so on the base language (e.g.
  English for gpt2) the only quality risk is that appended tokens re-tokenize
  some text into new tokens whose embeddings are freshly initialized.
  :func:`base_language_preservation` measures exactly that fraction, so a gate
  can require "≥ 99% of base-language tokens unchanged" with no model at all.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass


def bits_per_byte(total_nll_nats: float, total_bytes: int) -> float:
    """Convert a summed negative-log-likelihood to bits per byte.

    Args:
        total_nll_nats: Sum of per-token negative log-likelihoods (natural
            log / nats) over the evaluation text.
        total_bytes: Total UTF-8 bytes of that text.

    Returns:
        ``total_nll_nats / ln(2) / total_bytes`` — bits per byte. ``0.0`` when
        ``total_bytes`` is zero.
    """
    if total_bytes <= 0:
        return 0.0
    return total_nll_nats / math.log(2) / total_bytes


@dataclass(frozen=True)
class QualityReport:
    """Before/after quality comparison for a tokenizer transplant.

    Attributes:
        before_bpb: Bits per byte with the original tokenizer/model.
        after_bpb: Bits per byte after the transplant.
        delta_bpb: ``after_bpb - before_bpb`` (positive = worse).
        delta_pct: Percentage change relative to ``before_bpb``.
        budget_pct: The maximum tolerated increase, in percent.
        passed: ``True`` when ``delta_pct <= budget_pct``.
    """

    before_bpb: float
    after_bpb: float
    delta_bpb: float
    delta_pct: float
    budget_pct: float
    passed: bool

    def as_dict(self) -> dict[str, float | bool]:
        """Return the report as a plain JSON-serializable dict."""
        return {
            "before_bpb": self.before_bpb,
            "after_bpb": self.after_bpb,
            "delta_bpb": self.delta_bpb,
            "delta_pct": self.delta_pct,
            "budget_pct": self.budget_pct,
            "passed": self.passed,
        }


def quality_gate(
    before_bpb: float,
    after_bpb: float,
    *,
    budget_pct: float = 1.0,
) -> QualityReport:
    """Gate a transplant on bits-per-byte regression.

    Quality *drops* when bits-per-byte *rises*, so the gate passes iff
    ``after`` is no more than ``budget_pct`` percent above ``before``.

    Args:
        before_bpb: Baseline bits per byte.
        after_bpb: Post-transplant bits per byte.
        budget_pct: Maximum tolerated percentage increase (default 1%).

    Returns:
        A :class:`QualityReport`.
    """
    delta = after_bpb - before_bpb
    delta_pct = (delta / before_bpb * 100.0) if before_bpb else 0.0
    return QualityReport(
        before_bpb=before_bpb,
        after_bpb=after_bpb,
        delta_bpb=delta,
        delta_pct=delta_pct,
        budget_pct=budget_pct,
        passed=delta_pct <= budget_pct,
    )


def base_language_preservation(
    base_token_ids: Iterable[list[int]],
    adapted_token_ids: Iterable[list[int]],
    *,
    base_vocab_size: int,
) -> float:
    """Fraction of base-language tokens left unchanged by a transplant.

    A model-free proxy for "no base-language regression": for each document,
    compare the original tokenization to the adapted one and count how many of
    the original tokens are reproduced unchanged and still map to a base-vocab
    id (id ``< base_vocab_size``, whose embedding is preserved). Tokens that
    shift to newly-appended ids (fresh embeddings) count as *not* preserved.

    Args:
        base_token_ids: Per-document id lists from the original tokenizer.
        adapted_token_ids: Per-document id lists from the adapted tokenizer,
            in the same order.
        base_vocab_size: The id boundary; ids ``< base_vocab_size`` are the
            preserved base vocabulary.

    Returns:
        Preserved base tokens / total base tokens, in ``[0, 1]``; ``1.0`` when
        there are no base tokens.
    """
    preserved = 0
    total = 0
    for base_ids, adapted_ids in zip(
        base_token_ids, adapted_token_ids, strict=True
    ):
        total += len(base_ids)
        # Longest common prefix of unchanged base-vocab ids is a conservative,
        # order-preserving lower bound on how much of the doc is untouched.
        for b, a in zip(base_ids, adapted_ids, strict=False):
            if b == a and a < base_vocab_size:
                preserved += 1
            else:
                break
    return (preserved / total) if total else 1.0
