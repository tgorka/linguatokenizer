"""Tests for the ``extend`` adapter (``adapters.extend``).

Mirrors ``test_trainers.py``'s "real library, tiny corpus" philosophy: the
merge algorithm is exercised against a real, tiny ``tokenizers.Tokenizer``
standing in for a target model's ``backend_tokenizer`` (a faked merge
algorithm would prove little), plus a genuine tiny Polish BPE/SuperBPE
artifact produced via ``train_candidate``/``save_artifact``. Only
``transformers.AutoTokenizer.from_pretrained`` is faked (mirroring
``test_registry.py``), so no network access or ``adapters`` extra is needed
beyond ``tokenizers`` itself.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Literal

import pytest

from tokenizer_core.adapters.base import AdapterResult
from tokenizer_core.adapters.extend import extend
from tokenizer_core.adapters.registry import AdapterError
from tokenizer_core.benchmark.fertility import fertility
from tokenizer_core.train.artifact import TrainerMetadata, save_artifact
from tokenizer_core.train.config import Candidate, Language, Recipe, Targets
from tokenizer_core.train.config import CorpusSpec as _CorpusSpec
from tokenizer_core.train.trainers import train_candidate

pytest.importorskip("tokenizers")

_POLISH_WORDS = ["ala", "ma", "kota", "zażółć", "gęślą", "jaźń", "pies", "dom"]


def _write_lines(path: Path, n: int) -> Path:
    lines = [
        " ".join(_POLISH_WORDS[(i + j) % len(_POLISH_WORDS)] for j in range(6))
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
        language=Language(code="pl", name="Polish", scripts=["Latn"]),
        train_corpora=train_corpora,
        eval_corpora=[_CorpusSpec(source="local", id="unused.txt", weight=1.0)],
        candidates=[
            Candidate(name="placeholder", algorithm="bpe", vocab_size=400)
        ],
        targets=Targets(max_fertility_vs_english=1.0, max_quality_drop_pct=1.0),
        baselines=["gpt2"],
    )


def _polish_candidate(
    algorithm: Literal["bpe", "unigram", "superbpe"] = "bpe",
    *,
    vocab_size: int = 500,
) -> Candidate:
    return Candidate(
        name=f"pl-{algorithm}",
        algorithm=algorithm,
        vocab_size=vocab_size,
        byte_fallback=True,
        normalization="NFC",
    )


def _make_polish_artifact(
    tmp_path: Path,
    *,
    algorithm: Literal["bpe", "unigram", "superbpe"] = "bpe",
    n_docs: int = 300,
    seed: int = 42,
) -> Path:
    corpus_path = _write_lines(tmp_path / "corpus.txt", n_docs)
    recipe = _recipe([_local_spec(corpus_path)])
    # superbpe needs vocab_size >= 512 (byte alphabet + subwords + superwords).
    vocab_size = 600 if algorithm == "superbpe" else 500
    tokenizer, metadata = train_candidate(
        recipe,
        _polish_candidate(algorithm, vocab_size=vocab_size),
        n_docs=n_docs,
        seed=seed,
        data_dir=tmp_path,
    )
    out_dir = tmp_path / "artifact"
    save_artifact(tokenizer, metadata, out_dir)
    return out_dir


def _make_unigram_artifact(tmp_path: Path) -> Path:
    from tokenizer_core.train.trainers import _train_unigram  # noqa: PLC0415

    tokenizer = _train_unigram(_polish_candidate("unigram"), _polish_corpus())
    metadata = TrainerMetadata(
        name="pl-unigram",
        algorithm="unigram",
        vocab_size=500,
        byte_fallback=True,
        normalization="NFC",
        corpus_hash="0" * 64,
        seed=0,
        language_code="pl",
    )
    out_dir = tmp_path / "unigram_artifact"
    save_artifact(tokenizer, metadata, out_dir)
    return out_dir


def _polish_corpus(n: int = 300) -> list[str]:
    return [
        " ".join(_POLISH_WORDS[(i + j) % len(_POLISH_WORDS)] for j in range(6))
        for i in range(n)
    ]


def _make_target_tokenizer():
    """Build a real, tiny GPT-2-style byte-level BPE target tokenizer.

    Trained only on plain English/ASCII content, so it has no coverage of
    Polish diacritics -- exactly the gap ``extend`` exists to close.
    """
    from tokenizers import (  # noqa: PLC0415
        Tokenizer,
        decoders,
        models,
        pre_tokenizers,
        trainers,
    )

    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    english_docs = [
        "the quick brown fox jumps over the lazy dog",
        "hello world this is a test of the tokenizer",
        "a small target vocabulary for testing purposes only",
    ] * 50
    trainer = trainers.BpeTrainer(
        vocab_size=300,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(english_docs, trainer)
    return tokenizer


def _fake_target_module(backend_tokenizer, *, is_fast: bool = True):
    """Fake ``transformers`` module returning a fast-tokenizer stand-in."""
    stand_in = types.SimpleNamespace(
        is_fast=is_fast, backend_tokenizer=backend_tokenizer
    )

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(name: str, token: str | None = None) -> object:
            del name, token
            return stand_in

    return types.SimpleNamespace(AutoTokenizer=_FakeAutoTokenizer)


# --- happy path ---------------------------------------------------------


def test_extend_preserves_target_ids_and_appends_polish_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _make_target_tokenizer()
    monkeypatch.setitem(
        sys.modules, "transformers", _fake_target_module(target)
    )
    artifact_dir = _make_polish_artifact(tmp_path)

    result = extend("gpt2", artifact_dir)

    assert isinstance(result, AdapterResult)
    target_vocab = target.get_vocab()
    for token, token_id in target_vocab.items():
        assert result.tokenizer.token_to_id(token) == token_id

    assert result.base_vocab_size == max(target_vocab.values()) + 1
    assert len(result.added_tokens) > 0
    # Added tokens are Polish-only (not present in the target's own vocab).
    for tok in result.added_tokens:
        assert tok not in target_vocab
    # Added tokens get sequential ids starting at base_vocab_size, in order.
    for offset, tok in enumerate(result.added_tokens):
        assert (
            result.tokenizer.token_to_id(tok) == result.base_vocab_size + offset
        )


def test_extend_works_for_superbpe_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _make_target_tokenizer()
    monkeypatch.setitem(
        sys.modules, "transformers", _fake_target_module(target)
    )
    artifact_dir = _make_polish_artifact(tmp_path, algorithm="superbpe")

    result = extend("gpt2", artifact_dir)

    assert len(result.added_tokens) > 0


# --- unknown family -------------------------------------------------------


def test_extend_unknown_family_raises_before_any_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Any accidental import attempt should blow up loudly.
    monkeypatch.setitem(sys.modules, "transformers", None)

    with pytest.raises(AdapterError, match="falcon"):
        extend("falcon", tmp_path / "does_not_exist")


# --- non-hf_tokenizers target format ---------------------------------------


def test_extend_sentencepiece_target_raises_named_family_and_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "transformers", None)

    with pytest.raises(AdapterError, match="mistral") as exc_info:
        extend("mistral", tmp_path / "does_not_exist")
    assert "sentencepiece" in str(exc_info.value)


def test_extend_tiktoken_target_raises_named_family_and_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "transformers", None)

    with pytest.raises(AdapterError, match="kimi") as exc_info:
        extend("kimi", tmp_path / "does_not_exist")
    assert "tiktoken" in str(exc_info.value)


# --- unigram polish artifact ------------------------------------------------


def test_extend_unigram_artifact_raises_before_target_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # transformers is poisoned: any target-load attempt would blow up loudly,
    # proving the algorithm check happens first.
    monkeypatch.setitem(sys.modules, "transformers", None)
    artifact_dir = _make_unigram_artifact(tmp_path)

    with pytest.raises(AdapterError, match="unigram"):
        extend("gpt2", artifact_dir)


# --- overlapping vocab -------------------------------------------------------


def test_extend_overlapping_tokens_keep_target_id_no_duplicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _make_target_tokenizer()
    monkeypatch.setitem(
        sys.modules, "transformers", _fake_target_module(target)
    )
    artifact_dir = _make_polish_artifact(tmp_path)

    result = extend("gpt2", artifact_dir)

    target_vocab = target.get_vocab()
    merged_vocab = result.tokenizer.get_vocab()
    # Every target token keeps its original id; overlapping Polish tokens
    # (ASCII subwords likely shared with English BPE) are not duplicated.
    for token, token_id in target_vocab.items():
        assert merged_vocab[token] == token_id
    # No id collisions: distinct tokens never map to the same id.
    ids = list(merged_vocab.values())
    assert len(ids) == len(set(ids))


# --- missing/corrupt polish artifact -----------------------------------------


def test_extend_missing_polish_artifact_chains_underlying_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # transformers is poisoned: no target load should be attempted.
    monkeypatch.setitem(sys.modules, "transformers", None)
    missing_dir = tmp_path / "nonexistent_artifact"

    with pytest.raises(AdapterError) as exc_info:
        extend("gpt2", missing_dir)
    assert exc_info.value.__cause__ is not None


# --- target has no fast backend ----------------------------------------------


def test_extend_target_without_fast_backend_raises_naming_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        _fake_target_module(backend_tokenizer=None, is_fast=False),
    )
    artifact_dir = _make_polish_artifact(tmp_path)

    with pytest.raises(AdapterError, match="gpt2"):
        extend("gpt2", artifact_dir)


# --- AC3: fertility improves --------------------------------------------------


def test_extend_improves_fertility_on_polish_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _make_target_tokenizer()
    monkeypatch.setitem(
        sys.modules, "transformers", _fake_target_module(target)
    )
    artifact_dir = _make_polish_artifact(tmp_path)

    result = extend("gpt2", artifact_dir)

    polish_eval_corpus = [
        "zażółć gęślą jaźń",
        "ala ma kota i psa",
        "gęślą jaźń zażółć pies dom",
    ]

    # Passed as the bound `.encode` method (an `EncodeFn`) rather than the
    # `Tokenizer` object itself: `Tokenizer.encode`'s first parameter is named
    # `sequence`, not `text`, which structurally mismatches the `Encoder`
    # protocol under a real `tokenizers` install without affecting runtime
    # behavior (`_token_count` calls it the same way either branch is taken).
    before = fertility(target.encode, polish_eval_corpus)
    after = fertility(result.tokenizer.encode, polish_eval_corpus)

    assert after.fertility < before.fertility


# --- special tokens beyond model.vocab's max --------------------------------


def test_extend_special_token_beyond_vocab_max_is_preserved_no_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A special token added via ``add_special_tokens`` lives in the
    # top-level ``added_tokens`` list, at an id one past ``model.vocab``'s
    # max -- exactly the shape ``base_vocab_size`` must also clear so a
    # newly appended Polish token can't collide with it.
    target = _make_target_tokenizer()
    special_id = max(target.get_vocab().values()) + 1
    target.add_special_tokens(["<|endoftext|>"])
    monkeypatch.setitem(
        sys.modules, "transformers", _fake_target_module(target)
    )
    artifact_dir = _make_polish_artifact(tmp_path)

    result = extend("gpt2", artifact_dir)

    assert result.tokenizer.token_to_id("<|endoftext|>") == special_id
    assert result.base_vocab_size > special_id
    for tok in result.added_tokens:
        assert result.tokenizer.token_to_id(tok) != special_id

    # The id must also survive a further save/reload of the returned
    # tokenizer (e.g. Story 3.4/4.2 persisting it to disk): `tokenizers`
    # silently renumbers `added_tokens` to sit immediately after
    # `model.vocab` on every reload if a gap is left above `model.vocab`'s
    # size -- baking the special token into `model.vocab` (this module's
    # fix) is what keeps it stable across arbitrarily many round-trips.
    from tokenizers import Tokenizer  # noqa: PLC0415

    reloaded = Tokenizer.from_str(result.tokenizer.to_str())
    assert reloaded.token_to_id("<|endoftext|>") == special_id


# --- non-BPE hf_tokenizers-format target ------------------------------------


def test_extend_non_bpe_target_raises_clean_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: PLC0415

    word_level_target = Tokenizer(
        models.WordLevel(vocab={"a": 0, "b": 1}, unk_token="[UNK]")
    )
    word_level_target.pre_tokenizer = pre_tokenizers.Whitespace()
    monkeypatch.setitem(
        sys.modules, "transformers", _fake_target_module(word_level_target)
    )
    artifact_dir = _make_polish_artifact(tmp_path)

    with pytest.raises(AdapterError, match="WordLevel"):
        extend("gpt2", artifact_dir)
