"""Extend adapter: graft a Polish BPE/SuperBPE artifact onto a target tokenizer.

:func:`extend` merges a trained Polish candidate (Story 2.1/2.2) into a
target model's ``hf_tokenizers``-format tokenizer (resolved via Story 3.1's
:func:`~tokenizer_core.adapters.registry.load_target_tokenizer`) by appending
the Polish vocab/merges after the target's own: new tokens get ids starting
at ``max(target_id) + 1``, and every existing target id is preserved
unchanged. This is an append-only merge -- the target's own BPE merge-rank
priority is untouched, so anything the target already tokenized a certain way
keeps tokenizing exactly that way; new Polish merges only ever fire on
sequences the target's own merge list never matched.

Only merge-based Polish algorithms (``bpe``/``superbpe``) are supported here:
``unigram`` has no merge list to append. Only ``hf_tokenizers``-format targets
are supported: the merge operates on the target's raw ``tokenizers.Tokenizer``
JSON state, which sentencepiece/tiktoken targets do not expose.
"""

from __future__ import annotations

import json
import os
from typing import Any

from tokenizer_core.adapters.base import AdapterResult
from tokenizer_core.adapters.registry import (
    REGISTRY,
    AdapterError,
    load_target_tokenizer,
)
from tokenizer_core.train.artifact import load_artifact

_MERGE_BASED_ALGORITHMS = {"bpe", "superbpe"}


def extend(
    family: str, polish_artifact_dir: str | os.PathLike[str]
) -> AdapterResult:
    """Graft a Polish BPE/SuperBPE artifact onto a target model's tokenizer.

    Args:
        family: A :data:`~tokenizer_core.adapters.registry.REGISTRY` key
            (e.g. ``gpt2``) naming the target model.
        polish_artifact_dir: Directory containing a Polish candidate's
            ``tokenizer.json``/``metadata.json``, as written by
            :func:`~tokenizer_core.train.artifact.save_artifact`.

    Returns:
        An :class:`~tokenizer_core.adapters.base.AdapterResult` whose
        ``tokenizer`` preserves every target vocab id unchanged and appends
        Polish-only tokens starting at ``base_vocab_size``.

    Raises:
        AdapterError: If ``family`` is unknown, the target's format is not
            ``hf_tokenizers``, the target has no fast/backend tokenizer, the
            target's model isn't BPE-backed, the Polish artifact's algorithm
            is not merge-based (``bpe``/``superbpe``), or the Polish artifact
            fails to load (chained from the underlying exception).
    """
    _check_family_format(family)

    _polish_tokenizer, _metadata = _load_polish_artifact(polish_artifact_dir)
    backend_tokenizer = _get_target_backend_tokenizer(family)

    target_data = json.loads(backend_tokenizer.to_str())
    polish_data = json.loads(_polish_tokenizer.to_str())

    target_model_type = target_data["model"].get("type")
    if target_model_type != "BPE":
        raise AdapterError(
            f"extend requires a BPE-backed target tokenizer, but family "
            f"{family!r}'s hf_tokenizers model type is {target_model_type!r}"
        )

    original_target_vocab, original_added_tokens = _bake_added_tokens(
        family, target_data
    )
    target_vocab: dict[str, int] = target_data["model"]["vocab"]
    polish_vocab: dict[str, int] = polish_data["model"]["vocab"]

    base_vocab_size, added_tokens = _append_polish_vocab(
        target_vocab, polish_vocab
    )
    _append_polish_merges(target_data["model"], polish_data["model"])

    merged = _reconstruct(family, target_data)
    _verify_id_preservation(
        family, merged, {**original_target_vocab, **original_added_tokens}
    )

    return AdapterResult(
        tokenizer=merged,
        added_tokens=tuple(added_tokens),
        base_vocab_size=base_vocab_size,
    )


def _check_family_format(family: str) -> None:
    """Raise ``AdapterError`` unless ``family`` is a registered BPE target."""
    entry = REGISTRY.get(family)
    if entry is None:
        raise AdapterError(f"unknown target model family: {family!r}")
    if entry.format != "hf_tokenizers":
        raise AdapterError(
            f"extend only supports hf_tokenizers-format targets, but "
            f"family {family!r} has format {entry.format!r}"
        )


def _load_polish_artifact(
    polish_artifact_dir: str | os.PathLike[str],
) -> tuple[Any, Any]:
    """Load and validate a Polish artifact is merge-based (bpe/superbpe)."""
    try:
        tokenizer, metadata = load_artifact(polish_artifact_dir)
    except Exception as err:
        raise AdapterError(
            f"failed to load Polish artifact from {polish_artifact_dir!r}: "
            f"{err}"
        ) from err

    if metadata.algorithm not in _MERGE_BASED_ALGORITHMS:
        raise AdapterError(
            "extend only supports merge-based Polish algorithms "
            f"({sorted(_MERGE_BASED_ALGORITHMS)}), but the artifact at "
            f"{polish_artifact_dir!r} used algorithm "
            f"{metadata.algorithm!r}"
        )
    return tokenizer, metadata


def _get_target_backend_tokenizer(family: str) -> Any:
    """Resolve ``family``'s raw ``tokenizers.Tokenizer`` backend."""
    target = load_target_tokenizer(family)
    if not getattr(target, "is_fast", False):
        raise AdapterError(
            f"target tokenizer for family {family!r} is not a fast "
            "tokenizer (no backend_tokenizer to merge into)"
        )
    backend_tokenizer = getattr(target, "backend_tokenizer", None)
    if backend_tokenizer is None:
        raise AdapterError(
            f"target tokenizer for family {family!r} has no "
            "backend_tokenizer attribute"
        )
    return backend_tokenizer


def _bake_added_tokens(
    family: str, target_data: dict[str, Any]
) -> tuple[dict[str, int], dict[str, int]]:
    """Bake pre-existing special/added tokens into ``model.vocab``.

    Special tokens (bos/eos/pad/...) can live only in the top-level
    ``added_tokens`` list, at an id beyond ``model.vocab``'s max -- e.g. when
    attached via ``add_special_tokens`` rather than baked into the trained
    merge vocab (unlike e.g. real gpt2, whose ``<|endoftext|>`` is baked
    directly into ``model.vocab``). ``tokenizers.Tokenizer.from_str`` does not
    honor an ``added_tokens`` entry's recorded id when it leaves a gap above
    ``model.vocab``'s size -- it silently renumbers added tokens to sit
    immediately after the (possibly since-grown) model vocab on every reload.
    Baking any such entries into ``model.vocab`` at their existing id first
    closes that gap, so their id stays stable both now and across any future
    save/reload of the returned tokenizer.

    Returns:
        A ``(original_target_vocab, original_added_tokens)`` snapshot pair,
        both taken before any Polish tokens are appended, for the later
        id-preservation invariant check.
    """
    target_vocab: dict[str, int] = target_data["model"]["vocab"]
    original_target_vocab = dict(target_vocab)
    original_added_tokens: dict[str, int] = {
        added["content"]: added["id"]
        for added in target_data.get("added_tokens", [])
    }

    for token, token_id in original_added_tokens.items():
        if token in target_vocab and target_vocab[token] != token_id:
            raise AdapterError(
                f"target tokenizer for family {family!r} has inconsistent "
                f"ids for added token {token!r}: {target_vocab[token]!r} in "
                f"model.vocab vs {token_id!r} in added_tokens"
            )
        target_vocab[token] = token_id

    return original_target_vocab, original_added_tokens


def _append_polish_vocab(
    target_vocab: dict[str, int], polish_vocab: dict[str, int]
) -> tuple[int, list[str]]:
    """Append Polish-only vocab entries to ``target_vocab`` in place.

    Returns:
        The ``(base_vocab_size, added_tokens)`` pair: the id boundary between
        the target's original vocab and the newly appended Polish tokens, and
        the appended token strings in ascending id order.
    """
    base_vocab_size = max(target_vocab.values()) + 1
    next_id = base_vocab_size
    added_tokens: list[str] = []
    for token, _ in sorted(polish_vocab.items(), key=lambda kv: kv[1]):
        if token not in target_vocab:
            target_vocab[token] = next_id
            added_tokens.append(token)
            next_id += 1
    return base_vocab_size, added_tokens


def _append_polish_merges(
    target_model: dict[str, Any], polish_model: dict[str, Any]
) -> None:
    """Append Polish-only merges to ``target_model["merges"]`` in place."""
    target_merges = target_model["merges"]
    polish_merges = polish_model["merges"]
    existing_merges = {_merge_key(m) for m in target_merges}
    for merge in polish_merges:
        key = _merge_key(merge)
        if key not in existing_merges:
            target_merges.append(merge)
            existing_merges.add(key)


def _reconstruct(family: str, target_data: dict[str, Any]) -> Any:
    """Reconstruct a live ``tokenizers.Tokenizer`` from the merged JSON state."""
    import tokenizers  # noqa: PLC0415

    try:
        return tokenizers.Tokenizer.from_str(json.dumps(target_data))
    except Exception as err:
        raise AdapterError(
            f"failed to reconstruct the extended tokenizer for family "
            f"{family!r}: {err}"
        ) from err


def _verify_id_preservation(
    family: str, merged: Any, original_ids: dict[str, int]
) -> None:
    """Assert every pre-existing token still resolves to its original id.

    This is the existing-id-preservation invariant (AC2): every target vocab
    entry and every pre-existing special/added token (captured before any
    Polish tokens were appended) must still resolve to the same id on the
    reconstructed tokenizer. This must never trigger given the append-only
    merge above; it is a hard guard, not a silent skip.
    """
    for token, token_id in original_ids.items():
        if merged.token_to_id(token) != token_id:
            raise AdapterError(
                f"existing-id-preservation invariant violated for target "
                f"family {family!r}: token {token!r} expected id "
                f"{token_id}, got {merged.token_to_id(token)}"
            )


def _merge_key(merge: str | list[str]) -> tuple[str, ...]:
    """Return a hashable identity key for a ``model.merges`` entry.

    ``tokenizers`` versions differ on whether a merge is serialized as a
    ``"a b"`` string or a ``["a", "b"]`` pair; normalize both to a tuple so
    membership comparisons work regardless of the on-disk shape.
    """
    if isinstance(merge, str):
        return tuple(merge.split(" "))
    return tuple(merge)
