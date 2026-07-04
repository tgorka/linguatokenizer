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
    """Encodes a surface string into base sub-token ids.

    Records the ``add_special_tokens`` it was called with so tests can assert
    FVT never lets special tokens contaminate the donor decomposition.
    """

    def __init__(self, mapping: dict[str, list[int]]) -> None:
        self._mapping = mapping
        self.add_special_calls: list[bool] = []

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        self.add_special_calls.append(add_special_tokens)
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


def test_replace_layout_base_vocab_zero_fvt_from_full_donor() -> None:
    # replace() hands base_vocab_size=0; the FVT donor must still be the full
    # base matrix (not an empty slice), so every new row is finite, not NaN.
    np = pytest.importorskip("numpy")
    base_emb = np.arange(12, dtype=np.float32).reshape(4, 3)
    result = _result([" kot", " pies"], base_vocab=0)
    base_tok = _FakeBaseTokenizer({" kot": [0, 1], " pies": [2]})

    out = transplant_embeddings(result, base_emb, base_tok, method="fvt")

    assert out.shape == (2, 3)  # nothing preserved, both rows are new
    assert np.isfinite(out).all()
    assert np.allclose(out[0], base_emb[[0, 1]].mean(axis=0))


def test_fvt_never_adds_special_tokens_to_donor_decomposition() -> None:
    # BOS/EOS from the base tokenizer must not be averaged into new-token inits
    # (they would for LLaMA/Mistral/Qwen/Gemma with the encode default).
    np = pytest.importorskip("numpy")
    base_emb = np.arange(12, dtype=np.float32).reshape(4, 3)
    result = _result([" kot"], base_vocab=4)
    base_tok = _FakeBaseTokenizer({" kot": [0, 1]})

    transplant_embeddings(result, base_emb, base_tok, method="fvt")

    assert base_tok.add_special_calls  # was called
    assert all(v is False for v in base_tok.add_special_calls)


def test_too_few_base_rows_raises() -> None:
    np = pytest.importorskip("numpy")
    base_emb = np.zeros((3, 3), dtype=np.float32)
    result = _result([" a"], base_vocab=4)  # needs >= 4 rows
    with pytest.raises(EmbedError, match="base_vocab_size"):
        transplant_embeddings(result, base_emb, _FakeBaseTokenizer({}))
