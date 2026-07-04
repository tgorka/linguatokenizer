"""Tests for the adapt-bundle export/load round-trip."""

from __future__ import annotations

from pathlib import Path

import pytest

from tokenizer_core.adapters.base import AdapterResult
from tokenizer_core.adapters.bundle import export_bundle, load_bundle


def _tiny_tokenizer():
    from tokenizers import (  # noqa: PLC0415
        Tokenizer,
        models,
        pre_tokenizers,
        trainers,
    )

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.train_from_iterator(
        ["ala ma kota"] * 50,
        trainers.BpeTrainer(
            vocab_size=300,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    return tok


def test_bundle_roundtrip(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("tokenizers")
    pytest.importorskip("safetensors")

    tok = _tiny_tokenizer()
    vocab = tok.get_vocab_size()
    result = AdapterResult(
        tokenizer=tok, added_tokens=(), base_vocab_size=vocab
    )
    emb = np.random.default_rng(0).normal(size=(vocab, 8)).astype("float32")

    dest = export_bundle(
        result,
        emb,
        tmp_path / "bundle",
        base_model="gpt2",
        method="extend+fvt",
        target_fertility=1.56,
    )

    loaded_tok, loaded_emb, manifest = load_bundle(dest)
    assert loaded_tok.get_vocab_size() == vocab
    assert loaded_emb.shape == (vocab, 8)
    assert np.allclose(loaded_emb, emb)
    assert manifest["base_model"] == "gpt2"
    assert manifest["method"] == "extend+fvt"
    assert manifest["vocab_size"] == vocab
    assert manifest["target_fertility"] == 1.56


def test_bundle_rejects_mismatched_embeddings(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("tokenizers")

    tok = _tiny_tokenizer()
    result = AdapterResult(
        tokenizer=tok, added_tokens=(), base_vocab_size=tok.get_vocab_size()
    )
    wrong = np.zeros((5, 8), dtype="float32")
    with pytest.raises(ValueError, match="rows but tokenizer vocab"):
        export_bundle(
            result, wrong, tmp_path / "b", base_model="gpt2", method="x"
        )
