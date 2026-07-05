"""Domain routing: pick a domain-specialized tokenizer per input.

A single tokenizer trained on prose over-segments code (and vice versa), so one
option is to train a *general* (prose) and a *technical* (code) tokenizer and
route each input to the one that fits — the "technical vs general" router from
the project brief (Story 2.4). This module is the thin router + a
:class:`RoutedEncoder` that dispatches ``encode`` accordingly.

The router is a cheap, dependency-free heuristic: the density of code-signal
characters (brackets, operators, punctuation rare in prose). Whether routing is
worth the deployment cost (the model must handle both vocabularies) is an
empirical question — see ``results/routing-experiment.md`` for the measured
keep/drop decision.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

Domain = Literal["general", "technical"]

# Characters common in code, rare in natural-language prose.
_CODE_CHARS = frozenset("{}[]()<>;=+-*/%&|^~`\\_")


def code_score(text: str) -> float:
    """Return the fraction of ``text`` made of code-signal characters.

    Args:
        text: Input string.

    Returns:
        ``0.0`` for empty text, else code-signal chars / total chars in
        ``[0, 1]``. Prose scores near ``0``; source code scores much higher.
    """
    if not text:
        return 0.0
    return sum(c in _CODE_CHARS for c in text) / len(text)


def route(text: str, *, threshold: float = 0.06) -> Domain:
    """Route ``text`` to ``"technical"`` or ``"general"`` by code density.

    Args:
        text: Input string.
        threshold: Minimum :func:`code_score` to be routed ``"technical"``.

    Returns:
        The chosen :data:`Domain`.
    """
    return "technical" if code_score(text) >= threshold else "general"


class RoutedEncoder:
    """Encode by routing each input to a domain-specialized encoder.

    Args:
        encoders: Mapping of :data:`Domain` to an ``encode``-capable tokenizer
            (or ``str -> tokens`` callable). Must contain both ``"general"``
            and ``"technical"``.
        threshold: Passed to :func:`route`.

    Raises:
        ValueError: If either domain encoder is missing.
    """

    def __init__(
        self,
        encoders: dict[Domain, Any],
        *,
        threshold: float = 0.06,
    ) -> None:
        """Store the per-domain encoders and the routing threshold."""
        missing = {"general", "technical"} - set(encoders)
        if missing:
            raise ValueError(f"routing needs encoders for {sorted(missing)}")
        self._encoders = encoders
        self._threshold = threshold

    def route(self, text: str) -> Domain:
        """Return the domain :func:`route` picks for ``text``."""
        return route(text, threshold=self._threshold)

    def encode(self, text: str) -> Any:
        """Encode ``text`` with the domain encoder :meth:`route` selects."""
        encoder = self._encoders[self.route(text)]
        if callable(encoder):
            return encoder(text)
        return encoder.encode(text)


def router_accuracy(
    labeled: list[tuple[str, Domain]],
    *,
    threshold: float = 0.06,
    router: Callable[[str], Domain] | None = None,
) -> float:
    """Fraction of ``(text, gold_domain)`` pairs the router classifies right.

    Args:
        labeled: ``(text, gold_domain)`` pairs.
        threshold: Passed to the default :func:`route`.
        router: Optional custom router; defaults to :func:`route`.

    Returns:
        Accuracy in ``[0, 1]``; ``1.0`` for an empty list.
    """
    if not labeled:
        return 1.0

    def _default(text: str) -> Domain:
        return route(text, threshold=threshold)

    fn = router or _default
    correct = sum(1 for text, gold in labeled if fn(text) == gold)
    return correct / len(labeled)
