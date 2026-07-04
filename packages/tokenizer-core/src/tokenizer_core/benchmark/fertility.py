"""Fertility and compression metrics — the North Star for tokenizer efficiency.

*Fertility* is the average number of tokens a tokenizer emits per input word.
A universal tokenizer that never learned a language's morphology over-segments
it: Polish text costs the Mistral/Llama tokenizers ~1.5-3.2 tokens/word versus
~1.3 for English, which is the root of the "Polish costs ~2x the tokens"
penalty (see the project README). The objective of this project is to bring
Polish fertility down to (or below) the English baseline without quality loss.

This module is deliberately dependency-free: it works against *any* tokenizer
exposing an ``encode(text) -> sequence`` method (Hugging Face ``tokenizers``,
``transformers`` fast tokenizers, ``tiktoken``, or a plain callable), so it can
score baselines and candidates side by side without pulling native deps into
the base install.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

NormalizationForm = Literal["NFC", "NFD", "NFKC", "NFKD"]

# A "word" for fertility accounting: a maximal run of letters/marks/digits
# (Unicode-aware, so Polish diacritics and non-Latin scripts count correctly),
# optionally carrying internal apostrophes/hyphens (e.g. "d'Artagnan").
_WORD_RE = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)


@runtime_checkable
class Encoder(Protocol):
    """Minimal structural type for anything that can tokenize text."""

    def encode(self, text: str) -> Any:
        """Return a token sequence (list of ids or tokens) for ``text``."""
        ...


EncodeFn = Callable[[str], Any]


def count_words(text: str) -> int:
    """Count Unicode words in ``text``.

    Args:
        text: Input string.

    Returns:
        The number of word tokens, script-agnostic (Latin, Cyrillic, CJK
        combining behavior aside, etc.).
    """
    return len(_WORD_RE.findall(text))


def count_chars(text: str, *, normalize: NormalizationForm = "NFC") -> int:
    """Count characters in ``text`` after Unicode normalization.

    Args:
        text: Input string.
        normalize: Unicode normalization form applied before counting so that
            composed and decomposed diacritics compare equally.

    Returns:
        The number of Unicode scalar values after normalization.
    """
    return len(unicodedata.normalize(normalize, text))


def _token_count(encoder: Encoder | EncodeFn, text: str) -> int:
    """Return the number of tokens ``encoder`` produces for ``text``."""
    if callable(encoder) and not isinstance(encoder, Encoder):
        result = encoder(text)
    else:
        result = encoder.encode(text)
    # Hugging Face fast tokenizers return an ``Encoding`` with ``.ids``.
    ids = getattr(result, "ids", None)
    if ids is not None:
        return len(ids)
    return len(result)


@dataclass(frozen=True)
class FertilityReport:
    """Aggregate efficiency metrics for a tokenizer over a corpus.

    Attributes:
        n_docs: Number of documents scored.
        n_words: Total Unicode words across the corpus.
        n_chars: Total normalized characters across the corpus.
        n_tokens: Total tokens emitted by the tokenizer.
        fertility: Tokens per word (``n_tokens / n_words``); lower is better.
        chars_per_token: Characters per token (``n_chars / n_tokens``); higher
            means denser encoding.
        bytes_per_token: UTF-8 bytes per token; a script-independent density
            measure comparable across languages and alphabets.
    """

    n_docs: int
    n_words: int
    n_chars: int
    n_tokens: int
    fertility: float
    chars_per_token: float
    bytes_per_token: float

    def as_dict(self) -> dict[str, float | int]:
        """Return the report as a plain JSON-serializable dict."""
        return {
            "n_docs": self.n_docs,
            "n_words": self.n_words,
            "n_chars": self.n_chars,
            "n_tokens": self.n_tokens,
            "fertility": self.fertility,
            "chars_per_token": self.chars_per_token,
            "bytes_per_token": self.bytes_per_token,
        }


def fertility(
    encoder: Encoder | EncodeFn,
    corpus: Iterable[str],
    *,
    normalize: NormalizationForm = "NFC",
) -> FertilityReport:
    """Score a tokenizer's fertility over a corpus.

    Args:
        encoder: A tokenizer exposing ``encode`` or a plain ``str -> tokens``
            callable. The token count uses ``.ids`` when present (Hugging Face
            ``Encoding``), else ``len`` of the returned sequence.
        corpus: Iterable of documents (strings).
        normalize: Unicode normalization form for the character count.

    Returns:
        A :class:`FertilityReport`. Ratios are ``0.0`` when their denominator
        is zero (empty corpus / no tokens) rather than raising.
    """
    n_docs = n_words = n_chars = n_tokens = n_bytes = 0
    for doc in corpus:
        n_docs += 1
        n_words += count_words(doc)
        n_chars += count_chars(doc, normalize=normalize)
        n_tokens += _token_count(encoder, doc)
        n_bytes += len(doc.encode("utf-8"))

    return FertilityReport(
        n_docs=n_docs,
        n_words=n_words,
        n_chars=n_chars,
        n_tokens=n_tokens,
        fertility=(n_tokens / n_words) if n_words else 0.0,
        chars_per_token=(n_chars / n_tokens) if n_tokens else 0.0,
        bytes_per_token=(n_bytes / n_tokens) if n_tokens else 0.0,
    )
