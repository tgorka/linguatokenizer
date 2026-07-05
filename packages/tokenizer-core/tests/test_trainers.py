"""Tests for the BPE/Unigram trainers.

The training tests exercise the *real* ``tokenizers`` library on a tiny
in-memory corpus (fast, sub-second) rather than a faked module: faking a
tokenizer trainer proves little and is brittle against library-version quirks
(e.g. ``UnigramTrainer`` silently dropping a ``byte_fallback`` kwarg, the very
bug these trainers work around). They ``importorskip('tokenizers')`` so they
skip cleanly when the optional ``train`` extra is not installed (as in CI's
extras-free ``uv sync --all-packages``); the error-path and lazy-import tests
need no extra and always run.
"""

from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path
from typing import Literal

import pytest

from tokenizer_core.train.config import (
    Candidate,
    Language,
    Recipe,
    Targets,
)
from tokenizer_core.train.config import (
    CorpusSpec as _CorpusSpec,
)
from tokenizer_core.train.trainers import TrainerError, train_candidate


def _write_lines(path: Path, n: int) -> Path:
    # Varied Polish-ish content so BPE/Unigram have real merges to learn.
    words = ["ala", "ma", "kota", "zażółć", "gęślą", "jaźń", "pies", "dom"]
    lines = [
        " ".join(words[(i + j) % len(words)] for j in range(6))
        for i in range(n)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _local_spec(path: Path, weight: float = 1.0) -> _CorpusSpec:
    return _CorpusSpec.model_validate(
        {"source": "local", "id": str(path), "weight": weight}
    )


def _recipe(train_corpora: list[_CorpusSpec]) -> Recipe:
    return Recipe(
        language=Language(code="xx", name="Test", scripts=["Latn"]),
        train_corpora=train_corpora,
        eval_corpora=[_CorpusSpec(source="local", id="unused.txt", weight=1.0)],
        candidates=[
            Candidate(name="placeholder", algorithm="bpe", vocab_size=400)
        ],
        targets=Targets(max_fertility_vs_english=1.0, max_quality_drop_pct=1.0),
        baselines=["gpt2"],
    )


def _candidate(
    algorithm: Literal["bpe", "unigram", "superbpe"],
    *,
    byte_fallback: bool = True,
    vocab_size: int = 500,
) -> Candidate:
    # vocab must exceed the 256 byte tokens + specials for a byte-fallback
    # Unigram to have room for learned pieces.
    return Candidate(
        name=f"xx-{algorithm}",
        algorithm=algorithm,
        vocab_size=vocab_size,
        byte_fallback=byte_fallback,
        normalization="NFC",
    )


# --- real training (needs the `train` extra) -------------------------------


def test_train_bpe_produces_tokenizer_and_metadata(tmp_path: Path) -> None:
    pytest.importorskip("tokenizers")
    recipe = _recipe([_local_spec(_write_lines(tmp_path / "c.txt", 300))])

    tokenizer, metadata = train_candidate(
        recipe, _candidate("bpe"), n_docs=300, seed=42, data_dir=tmp_path
    )

    assert tokenizer.get_vocab_size() <= 500
    assert metadata.algorithm == "bpe"
    assert metadata.vocab_size == 500
    assert metadata.byte_fallback is True
    assert metadata.language_code == "xx"
    assert metadata.seed == 42
    assert len(metadata.corpus_hash) == 64  # sha256 hex


def test_train_bpe_byte_level_covers_exotic_chars_without_unk(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tokenizers")
    recipe = _recipe([_local_spec(_write_lines(tmp_path / "c.txt", 200))])
    tokenizer, _ = train_candidate(
        recipe, _candidate("bpe"), n_docs=200, seed=1, data_dir=tmp_path
    )
    # CJK/emoji never seen in training must still encode (byte-level), no <unk>.
    tokens = tokenizer.encode("字 \U0001f954").tokens
    assert "<unk>" not in tokens


def test_train_unigram_byte_fallback_flag_and_coverage_are_real(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tokenizers")
    recipe = _recipe([_local_spec(_write_lines(tmp_path / "c.txt", 300))])

    tokenizer, metadata = train_candidate(
        recipe, _candidate("unigram"), n_docs=300, seed=7, data_dir=tmp_path
    )

    # The whole point of the fix: byte_fallback is genuinely on in the model,
    # not just claimed in metadata.
    model = json.loads(tokenizer.to_str())["model"]
    assert model["byte_fallback"] is True
    assert metadata.byte_fallback is True
    # And it delivers real coverage: an unseen char -> byte tokens, no <unk>.
    tokens = tokenizer.encode("字").tokens
    assert "<unk>" not in tokens


def test_train_superbpe_forms_cross_word_tokens(tmp_path: Path) -> None:
    pytest.importorskip("tokenizers")
    # A recurring two-word phrase should become a single superword token.
    path = tmp_path / "c.txt"
    path.write_text("\n".join(["dobry pies i kot"] * 200), encoding="utf-8")
    recipe = _recipe([_local_spec(path)])
    tokenizer, metadata = train_candidate(
        recipe,
        _candidate("superbpe", vocab_size=600),
        n_docs=200,
        seed=5,
        data_dir=tmp_path,
    )
    assert metadata.algorithm == "superbpe"
    # At least one vocab token spans a space (byte-level space marker Ġ),
    # i.e. a genuine cross-word superword was learned.
    superwords = [
        t for t in tokenizer.get_vocab() if "Ġ" in t[1:] and len(t) > 3
    ]
    assert superwords, "expected at least one cross-word superword token"
    # And byte fallback still guarantees coverage of unseen characters.
    assert "<unk>" not in tokenizer.encode("字").tokens


def test_superbpe_rejects_too_small_vocab(tmp_path: Path) -> None:
    # Below the byte-alphabet + headroom minimum, superbpe can't honor the
    # requested vocab size, so it must raise rather than mismatch metadata.
    recipe = _recipe([_local_spec(_write_lines(tmp_path / "c.txt", 20))])
    with pytest.raises(TrainerError, match="vocab_size"):
        train_candidate(
            recipe,
            _candidate("superbpe", vocab_size=300),
            n_docs=20,
            seed=0,
            data_dir=tmp_path,
        )


def test_unigram_without_byte_fallback_has_flag_off(tmp_path: Path) -> None:
    pytest.importorskip("tokenizers")
    recipe = _recipe([_local_spec(_write_lines(tmp_path / "c.txt", 200))])
    tokenizer, metadata = train_candidate(
        recipe,
        _candidate("unigram", byte_fallback=False),
        n_docs=200,
        seed=3,
        data_dir=tmp_path,
    )
    assert metadata.byte_fallback is False
    assert json.loads(tokenizer.to_str())["model"]["byte_fallback"] is False


def test_deterministic_same_seed_same_corpus_hash(tmp_path: Path) -> None:
    pytest.importorskip("tokenizers")
    recipe = _recipe([_local_spec(_write_lines(tmp_path / "c.txt", 200))])
    _, m1 = train_candidate(
        recipe, _candidate("bpe"), n_docs=150, seed=42, data_dir=tmp_path
    )
    _, m2 = train_candidate(
        recipe, _candidate("bpe"), n_docs=150, seed=42, data_dir=tmp_path
    )
    assert m1.corpus_hash == m2.corpus_hash


# --- error paths (no extra needed: they raise before real training) --------


def test_unsupported_algorithm_raises_trainer_error(tmp_path: Path) -> None:
    # bpe/unigram/superbpe are all handled; the defensive branch only fires for
    # an algorithm the Candidate Literal would normally reject, so bypass
    # validation with model_construct to reach it.
    recipe = _recipe([_local_spec(_write_lines(tmp_path / "c.txt", 20))])
    bogus = Candidate.model_construct(
        name="xx-bogus",
        algorithm="wordpiece",
        vocab_size=500,
        byte_fallback=True,
        normalization="NFC",
    )
    with pytest.raises(TrainerError, match="wordpiece"):
        train_candidate(recipe, bogus, n_docs=20, seed=0, data_dir=tmp_path)


def test_zero_documents_raises_trainer_error(tmp_path: Path) -> None:
    # An hf source with no token configured is skipped -> zero documents.
    recipe = _recipe(
        [_CorpusSpec(source="hf", id="x/y", config="pl", split="train")]
    )
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)

    # Fake datasets so the hf source resolves to a graceful skip (empty).
    def load_dataset(*a: object, **k: object):
        raise RuntimeError("gated")

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(load_dataset=load_dataset),
    )
    try:
        with pytest.raises(TrainerError, match="zero documents"):
            train_candidate(
                recipe, _candidate("bpe"), n_docs=10, seed=0, data_dir=tmp_path
            )
    finally:
        monkeypatch.undo()


def test_import_does_not_require_tokenizers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The module must import with no `tokenizers` in sys.modules (lazy guard).
    monkeypatch.delitem(sys.modules, "tokenizers", raising=False)
    monkeypatch.delitem(
        sys.modules, "tokenizer_core.train.trainers", raising=False
    )
    module = importlib.import_module("tokenizer_core.train.trainers")
    assert hasattr(module, "train_candidate")


def test_bpe_split_words_toggles_pretokenizer_regex(tmp_path: Path) -> None:
    # split_words=False -> byte-dense scripts merge across the word regex,
    # producing fewer/longer tokens on repeated multi-word phrases.
    pytest.importorskip("tokenizers")
    path = tmp_path / "c.txt"
    path.write_text("\n".join(["alpha beta gamma delta"] * 300), "utf-8")
    recipe = _recipe([_local_spec(path)])

    split = _candidate("bpe")  # split_words defaults True
    nosplit = Candidate(
        name="xx-nosplit",
        algorithm="bpe",
        vocab_size=500,
        byte_fallback=True,
        normalization="NFC",
        split_words=False,
    )
    tok_split, _ = train_candidate(
        recipe, split, n_docs=300, seed=1, data_dir=tmp_path
    )
    tok_nosplit, _ = train_candidate(
        recipe, nosplit, n_docs=300, seed=1, data_dir=tmp_path
    )
    probe = "alpha beta gamma delta"
    # no-split can form cross-word tokens, so it needs <= the split count.
    assert len(tok_nosplit.encode(probe).ids) <= len(
        tok_split.encode(probe).ids
    )
    # and at least one no-split vocab token spans a space (Ġ marker).
    assert any("Ġ" in t[1:] and len(t) > 3 for t in tok_nosplit.get_vocab())
