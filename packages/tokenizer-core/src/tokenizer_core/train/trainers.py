"""Train a BPE or Unigram tokenizer candidate from a recipe's train corpora.

`train/config.py` (Story 1.1) validates a recipe's candidates and
`corpora/sampling.py` (Story 1.2) can draw a seeded/weighted sample of a
recipe's `train_corpora`, but nothing yet turns that sample into an actual
Hugging Face `tokenizers.Tokenizer`. This module is that missing piece:
:func:`train_candidate` samples the recipe's train corpora, builds a BPE or
Unigram pipeline per the candidate's `vocab_size`/`byte_fallback`/
`normalization`, and trains it via `train_from_iterator`.

SuperBPE (Story 2.2) is out of scope: any candidate whose `algorithm` isn't
`bpe`/`unigram` raises :class:`TrainerError` naming the candidate and
algorithm, so the CLI can skip it and continue with the rest.

``tokenizers`` is imported lazily inside function bodies (not at module
level), matching ``corpora/sources.py``'s guard convention, so this module
stays importable (and ``task check`` stays green) with no ``train`` extra
installed.

``corpora.sampling`` is also imported lazily inside :func:`train_candidate`
rather than at module level: ``corpora/sources.py`` imports
``tokenizer_core.train.config``, and ``tokenizer_core.train``'s own
``__init__.py`` re-exports this module -- a module-level
``tokenizer_core.corpora`` import here would close that loop into a circular
import (``train`` -> ``corpora`` -> ``train``, the last edge via package
``__init__.py`` initialization, not just submodules).
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import TYPE_CHECKING

from tokenizer_core.train.artifact import TrainerMetadata

if TYPE_CHECKING:
    from tokenizers import Tokenizer

    from tokenizer_core.train.config import Candidate, Recipe


class TrainerError(Exception):
    """Raised when a candidate cannot be trained.

    Covers both an unsupported ``algorithm`` (e.g. ``superbpe``, out of scope
    for this story) and a training corpus that sampled zero documents.
    """


def _build_normalizer(normalization: str):
    """Return a ``tokenizers.normalizers`` instance for ``normalization``."""
    from tokenizers import normalizers  # noqa: PLC0415

    mapping = {
        "NFC": normalizers.NFC,
        "NFD": normalizers.NFD,
        "NFKC": normalizers.NFKC,
        "NFKD": normalizers.NFKD,
    }
    try:
        normalizer_cls = mapping[normalization]
    except KeyError as err:
        raise TrainerError(
            f"unsupported normalization form: {normalization!r}"
        ) from err
    return normalizer_cls()


def _train_bpe(candidate: Candidate, documents: list[str]) -> Tokenizer:
    """Train a GPT-2-style byte-level BPE tokenizer for ``candidate``."""
    from tokenizers import (  # noqa: PLC0415
        Tokenizer,
        decoders,
        models,
        pre_tokenizers,
        trainers,
    )

    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.normalizer = _build_normalizer(candidate.normalization)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=candidate.vocab_size,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(documents, trainer)
    return tokenizer


def _train_unigram(candidate: Candidate, documents: list[str]) -> Tokenizer:
    """Train a Unigram tokenizer for ``candidate``.

    ``tokenizers``' ``UnigramTrainer`` silently drops a ``byte_fallback``
    kwarg (it prints "Ignored unknown kwargs option byte_fallback" and the
    trained model's ``byte_fallback`` stays ``false``). To make byte fallback
    real we instead (1) seed the 256 ``<0xNN>`` byte tokens as special tokens
    so they enter the vocab during training and (2) flip the model's
    ``byte_fallback`` flag on afterwards (there is no live setter, so via a
    JSON round-trip) plus attach a ``ByteFallback`` decoder. The result maps
    any out-of-vocabulary character to byte tokens instead of ``<unk>`` —
    verified by :func:`_assert_byte_coverage`.
    """
    from tokenizers import (  # noqa: PLC0415
        Tokenizer,
        decoders,
        models,
        trainers,
    )

    tokenizer = Tokenizer(models.Unigram())
    tokenizer.normalizer = _build_normalizer(candidate.normalization)

    special_tokens = ["<unk>"]
    if candidate.byte_fallback:
        special_tokens += [f"<0x{i:02X}>" for i in range(256)]
    trainer = trainers.UnigramTrainer(
        vocab_size=candidate.vocab_size,
        unk_token="<unk>",
        special_tokens=special_tokens,
    )
    tokenizer.train_from_iterator(documents, trainer)

    if not candidate.byte_fallback:
        return tokenizer

    # Enable the model-level byte_fallback flag (no live setter exists).
    data = json.loads(tokenizer.to_str())
    data["model"]["byte_fallback"] = True
    rebuilt = Tokenizer.from_str(json.dumps(data))
    rebuilt.decoder = decoders.ByteFallback()
    return rebuilt


def _assert_byte_coverage(tokenizer: Tokenizer, name: str) -> None:
    """Verify a byte_fallback tokenizer never emits ``<unk>``.

    Encodes a probe of characters guaranteed to be outside a Polish-trained
    vocab (CJK, emoji, rare symbols); byte fallback must decompose them into
    ``<0xNN>`` tokens rather than a single ``<unk>``.

    Raises:
        TrainerError: If the probe produces an ``<unk>`` token, i.e. coverage
            is not actually complete despite ``byte_fallback`` being requested.
    """
    # CJK, emoji, and a rare symbol — none in a Polish-trained vocab.
    probe = "字 🥔 ℞"
    tokens = tokenizer.encode(probe).tokens
    if "<unk>" in tokens:
        raise TrainerError(
            f"candidate {name!r} requested byte_fallback but produced "
            "an <unk> token on the coverage probe (incomplete byte coverage)"
        )


def _corpus_hash(manifest_dict: dict[str, object]) -> str:
    """Return a deterministic sha256 fingerprint of a manifest dict."""
    encoded = json.dumps(
        manifest_dict, sort_keys=True, ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def train_candidate(
    recipe: Recipe,
    candidate: Candidate,
    *,
    n_docs: int,
    seed: int,
    data_dir: str | os.PathLike[str] | None = None,
) -> tuple[Tokenizer, TrainerMetadata]:
    """Train one candidate tokenizer from ``recipe.train_corpora``.

    Args:
        recipe: The language recipe naming the train corpora to sample from.
        candidate: The candidate to train (``algorithm`` must be ``bpe`` or
            ``unigram``).
        n_docs: Total number of documents to sample across
            ``recipe.train_corpora``.
        seed: Random seed for reservoir sampling (determinism).
        data_dir: Data-plane root directory. Defaults to
            ``$TOKENIZER_DATA_DIR`` when omitted.

    Returns:
        A ``(tokenizer, metadata)`` tuple: the trained
        ``tokenizers.Tokenizer`` and its :class:`TrainerMetadata`.

    Raises:
        TrainerError: If ``candidate.algorithm`` is not ``bpe``/``unigram``
            (e.g. ``superbpe``, Story 2.2's scope), or if the sampled
            training corpus has zero documents.
    """
    # Imported here, not at module level, to avoid a circular import (see the
    # module docstring): `corpora.sources` imports `train.config`, which
    # runs `train/__init__.py`, which re-exports this module.
    from tokenizer_core.corpora.sampling import sample_corpora  # noqa: PLC0415

    manifest, documents = sample_corpora(
        recipe.train_corpora,
        n_docs=n_docs,
        seed=seed,
        language_code=recipe.language.code,
        data_dir=data_dir,
    )

    if not documents:
        raise TrainerError(
            f"cannot train candidate {candidate.name!r}: "
            "sampled training corpus has zero documents"
        )

    if candidate.algorithm == "bpe":
        tokenizer = _train_bpe(candidate, documents)
    elif candidate.algorithm == "unigram":
        tokenizer = _train_unigram(candidate, documents)
    else:
        raise TrainerError(
            f"candidate {candidate.name!r} has unsupported algorithm "
            f"{candidate.algorithm!r} (only 'bpe'/'unigram' are trainable "
            "here; 'superbpe' is Story 2.2)"
        )

    # When byte_fallback is requested the metadata records it as true, so make
    # it a *verified* guarantee (not just a claimed flag): byte-level BPE and
    # byte-fallback Unigram both must decompose any character to bytes rather
    # than emit <unk>.
    if candidate.byte_fallback:
        _assert_byte_coverage(tokenizer, candidate.name)

    metadata = TrainerMetadata(
        name=candidate.name,
        algorithm=candidate.algorithm,
        vocab_size=candidate.vocab_size,
        byte_fallback=candidate.byte_fallback,
        normalization=candidate.normalization,
        corpus_hash=_corpus_hash(manifest.as_dict()),
        seed=seed,
        language_code=recipe.language.code,
    )

    return tokenizer, metadata
