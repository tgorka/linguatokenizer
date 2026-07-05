"""Tests for domain routing."""

from __future__ import annotations

import pytest

from tokenizer_core.routing import (
    RoutedEncoder,
    code_score,
    route,
    router_accuracy,
)

_PROSE = "The quick brown fox jumps over the lazy dog near the river bank."
_CODE = "def f(x): return [y*2 for y in x if y>0];  # doubles positives"


def test_code_score_prose_low_code_high() -> None:
    assert code_score(_PROSE) < 0.03
    assert code_score(_CODE) > 0.06
    assert code_score("") == 0.0


def test_route_prose_and_code() -> None:
    assert route(_PROSE) == "general"
    assert route(_CODE) == "technical"


class _Tagger:
    def __init__(self, tag: str) -> None:
        self._tag = tag

    def encode(self, text: str) -> list[str]:
        return [self._tag, *text.split()]


def test_routed_encoder_dispatches() -> None:
    enc = RoutedEncoder({"general": _Tagger("G"), "technical": _Tagger("T")})
    assert enc.encode(_PROSE)[0] == "G"
    assert enc.encode(_CODE)[0] == "T"


def test_routed_encoder_accepts_callables() -> None:
    enc = RoutedEncoder(
        {"general": lambda t: ["g"], "technical": lambda t: ["t", "t"]}
    )
    assert enc.encode(_PROSE) == ["g"]
    assert enc.encode(_CODE) == ["t", "t"]


def test_routed_encoder_requires_both_domains() -> None:
    with pytest.raises(ValueError, match="technical"):
        RoutedEncoder({"general": _Tagger("G")})


def test_router_accuracy() -> None:
    labeled = [(_PROSE, "general"), (_CODE, "technical")]
    assert router_accuracy(labeled) == 1.0
    # A deliberately wrong custom router scores 0.
    assert router_accuracy(labeled, router=lambda t: "general") == 0.5
    assert router_accuracy([]) == 1.0
