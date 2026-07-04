"""Tests for the training-free FVT embedding transplant (``adapters.embed``).

The transplant math is exercised with synthetic embeddings + trivial fake
tokenizers (no model download, no ``torch``), so it runs anywhere ``numpy`` is
available and skips cleanly otherwise.
"""

from __future__ import annotations

import pytest

from tokenizer_core.adapters.base import AdapterResult
from tokenizer_core.adapters.embed import (
    EmbedError,
    transplant_embeddings,
)


class _FakeExtendedTokenizer:
    """Decodes a new token id back to its surface string."""

    def __init__(self, surfaces: list[str], base_vocab: int) -> None:
        self._surfaces = surfaces
        self._base = base_vocab

    def decode(self, ids: list[int]) -> str:
        return self._surfaces[ids[0] - self._base]


class _FakeBaseTokenizer:
    """Encodes a surface string into base sub-token ids."""

    def __init__(self, mapping: dict[str, list[int]]) -> None:
        self._mapping = mapping

    def encode(self, text: str) -> list[int]:
        return self._mapping.get(text, [])


def _result(added: list[str], base_vocab: int) -> AdapterResult:
    return AdapterResult(
        tokenizer=_FakeExtendedTokenizer(added, base_vocab),
        added_tokens=tuple(added),
        base_vocab_size=base_vocab,
    )


def test_fvt_preserves_base_rows_and_averages_subwords() -> None:
    np = pytest.importorskip("numpy")
    base_emb = np.arange(12, dtype=np.float32).reshape(4, 3)  # 4 tokens, dim 3
    result = _result([" kot", " pies"], base_vocab=4)
    base_tok = _FakeBaseTokenizer({" kot": [0, 1], " pies": [2]})

    out = transplant_embeddings(result, base_emb, base_tok, method="fvt")

    assert out.shape == (6, 3)
    assert np.array_equal(out[:4], base_emb)  # base rows unchanged
    assert np.allclose(
        out[4], base_emb[[0, 1]].mean(axis=0)
    )  # mean of subwords
    assert np.allclose(out[5], base_emb[2])


def test_fvt_falls_back_to_global_mean_when_no_subwords() -> None:
    np = pytest.importorskip("numpy")
    base_emb = np.arange(12, dtype=np.float32).reshape(4, 3)
    result = _result(["unknown"], base_vocab=4)
    base_tok = _FakeBaseTokenizer({})  # encodes nothing

    out = transplant_embeddings(result, base_emb, base_tok, method="fvt")

    assert np.allclose(out[4], base_emb.mean(axis=0))


def test_mean_method_sets_every_new_row_to_global_mean() -> None:
    np = pytest.importorskip("numpy")
    base_emb = np.arange(12, dtype=np.float32).reshape(4, 3)
    result = _result([" a", " b"], base_vocab=4)
    base_tok = _FakeBaseTokenizer({" a": [0], " b": [3]})

    out = transplant_embeddings(result, base_emb, base_tok, method="mean")

    assert np.allclose(out[4], base_emb.mean(axis=0))
    assert np.allclose(out[5], base_emb.mean(axis=0))


def test_out_of_range_subword_ids_are_ignored() -> None:
    np = pytest.importorskip("numpy")
    base_emb = np.arange(12, dtype=np.float32).reshape(4, 3)
    result = _result([" x"], base_vocab=4)
    # id 99 is out of range and must be dropped, leaving just id 1.
    base_tok = _FakeBaseTokenizer({" x": [1, 99]})

    out = transplant_embeddings(result, base_emb, base_tok, method="fvt")

    assert np.allclose(out[4], base_emb[1])


def test_too_few_base_rows_raises() -> None:
    np = pytest.importorskip("numpy")
    base_emb = np.zeros((3, 3), dtype=np.float32)
    result = _result([" a"], base_vocab=4)  # needs >= 4 rows
    with pytest.raises(EmbedError, match="base_vocab_size"):
        transplant_embeddings(result, base_emb, _FakeBaseTokenizer({}))
