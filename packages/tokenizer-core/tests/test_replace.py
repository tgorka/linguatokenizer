"""Tests for the replace adapter (``adapters.replace``)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tokenizer_core.adapters.registry import AdapterError
from tokenizer_core.adapters.replace import replace
from tokenizer_core.train.artifact import TrainerMetadata, save_artifact
from tokenizer_core.train.config import Candidate
from tokenizer_core.train.trainers import _train_bpe


def _make_bpe_artifact(tmp_path: Path) -> Path:
    pytest.importorskip("tokenizers")
    corpus = ["ala ma kota", "kot ma ale", "zażółć gęślą jaźń"] * 40
    candidate = Candidate(
        name="pl-bpe",
        algorithm="bpe",
        vocab_size=400,
        byte_fallback=True,
        normalization="NFC",
    )
    tokenizer = _train_bpe(candidate, corpus)
    metadata = TrainerMetadata(
        name="pl-bpe",
        algorithm="bpe",
        vocab_size=400,
        byte_fallback=True,
        normalization="NFC",
        corpus_hash="0" * 64,
        seed=0,
        language_code="pl",
    )
    out = tmp_path / "artifact"
    save_artifact(tokenizer, metadata, out)
    return out


def test_replace_returns_polish_tokenizer_with_zero_base(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tokenizers")
    artifact = _make_bpe_artifact(tmp_path)

    result = replace("gpt2", artifact)

    # Whole Polish vocab becomes the model's vocab; nothing carried over.
    assert result.base_vocab_size == 0
    assert len(result.added_tokens) == result.tokenizer.get_vocab_size()
    # added_tokens is in ascending id order.
    for i, tok in enumerate(result.added_tokens):
        assert result.tokenizer.token_to_id(tok) == i


def test_replace_unknown_family_raises(tmp_path: Path) -> None:
    with pytest.raises(AdapterError, match="falcon"):
        replace("falcon", tmp_path)


def test_replace_bad_artifact_raises(tmp_path: Path) -> None:
    with pytest.raises(AdapterError, match="failed to load"):
        replace("gpt2", tmp_path / "does_not_exist")
