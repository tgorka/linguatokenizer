"""Tokenizer encode-speed timing.

Tokenization is a real part of the serving pipeline, and a
lower-fertility tokenizer that is also slower to run could erode its wins. This
measures **encode wall-clock speed** (tokens/sec, ms/document) — CPU-only, no
model. It is *not* end-to-end inference latency (prefill + decode), which needs
a served model on a GPU; that is projected analytically by
:mod:`tokenizer_core.economics`.

Times the best of ``repeats`` passes (least affected by scheduler noise) over
any ``encode``-capable object — the same contract as
:func:`tokenizer_core.benchmark.fertility.fertility`.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass

from tokenizer_core.benchmark.fertility import (
    EncodeFn,
    Encoder,
    _token_count,
)


@dataclass(frozen=True)
class TimingReport:
    """Encode-speed measurement over a corpus.

    Attributes:
        n_docs: Number of documents encoded.
        n_tokens: Total tokens produced.
        seconds: Best (minimum) wall-clock time across the timed passes.
        tokens_per_sec: ``n_tokens / seconds``.
        ms_per_doc: Milliseconds per document (``seconds / n_docs * 1000``).
    """

    n_docs: int
    n_tokens: int
    seconds: float
    tokens_per_sec: float
    ms_per_doc: float

    def as_dict(self) -> dict[str, float | int]:
        """Return the report as a plain JSON-serializable dict."""
        return {
            "n_docs": self.n_docs,
            "n_tokens": self.n_tokens,
            "seconds": self.seconds,
            "tokens_per_sec": self.tokens_per_sec,
            "ms_per_doc": self.ms_per_doc,
        }


def encode_timing(
    encoder: Encoder | EncodeFn,
    corpus: Iterable[str],
    *,
    repeats: int = 3,
) -> TimingReport:
    """Time how fast ``encoder`` tokenizes ``corpus``.

    Args:
        encoder: A tokenizer exposing ``encode`` or a ``str -> tokens``
            callable.
        corpus: Iterable of documents (materialized once).
        repeats: Number of timed passes; the fastest is reported.

    Returns:
        A :class:`TimingReport`. All fields are ``0`` for an empty corpus.

    Raises:
        ValueError: If ``repeats < 1``.
    """
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    docs = list(corpus)
    if not docs:
        return TimingReport(0, 0, 0.0, 0.0, 0.0)

    best = float("inf")
    n_tokens = 0
    for _ in range(repeats):
        start = time.perf_counter()
        total = 0
        for doc in docs:
            total += _token_count(encoder, doc)
        elapsed = time.perf_counter() - start
        best = min(best, elapsed)
        n_tokens = total

    return TimingReport(
        n_docs=len(docs),
        n_tokens=n_tokens,
        seconds=best,
        tokens_per_sec=(n_tokens / best) if best > 0 else 0.0,
        ms_per_doc=(best / len(docs) * 1000.0),
    )
