"""Replace adapter: swap a target model's tokenizer wholesale for the Polish one.

Where :func:`~tokenizer_core.adapters.extend.extend` (Story 3.2) *appends* the
Polish vocabulary and preserves the target's token ids — so most base-language
behavior is untouched and English stays ~unchanged — ``replace`` hands the model
the Polish tokenizer outright. That reaches the standalone Polish fertility
(~1.56 tok/word for ``pl-bpe-32k``, versus ~2.77 for extended gpt2) at the cost
of a fresh embedding table: base-language ids are **not** preserved, so a short
continued-pretraining pass is expected to realize the model's full quality on
the new tokenization. Story 3.4's :func:`~tokenizer_core.adapters.embed.
transplant_embeddings` FVT-warm-starts that table from the donor model (with
``base_vocab_size == 0``, every row is initialized) so the retrain starts from a
sane point rather than random.

**When to prefer which:** ``extend`` when base-language quality must be
preserved with zero training; ``replace`` when maximum efficiency on the target
language is the goal and a short retrain is acceptable.

Unlike ``extend`` (which needs merge-based Polish artifacts to splice merges),
``replace`` works with any trained Polish tokenizer — bpe, unigram, or superbpe.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from tokenizer_core.adapters.base import AdapterResult
from tokenizer_core.adapters.registry import REGISTRY, AdapterError
from tokenizer_core.train.artifact import load_artifact

if TYPE_CHECKING:
    from tokenizers import Tokenizer


def replace(
    family: str, polish_artifact_dir: str | os.PathLike[str]
) -> AdapterResult:
    """Replace ``family``'s tokenizer wholesale with a Polish artifact.

    Args:
        family: A :data:`~tokenizer_core.adapters.registry.REGISTRY` key naming
            the target model whose tokenizer is being replaced.
        polish_artifact_dir: Directory containing a Polish candidate's
            ``tokenizer.json``/``metadata.json`` (any algorithm), as written by
            :func:`~tokenizer_core.train.artifact.save_artifact`.

    Returns:
        An :class:`~tokenizer_core.adapters.base.AdapterResult` whose
        ``tokenizer`` is the Polish tokenizer, ``base_vocab_size`` is ``0``
        (nothing carried over from the target), and ``added_tokens`` lists every
        Polish token in ascending id order — the init plan for
        :func:`~tokenizer_core.adapters.embed.transplant_embeddings`, which then
        FVT-initializes the whole new embedding table from the donor model.

    Raises:
        AdapterError: If ``family`` is not registered, or the Polish artifact
            cannot be loaded.
    """
    if family not in REGISTRY:
        raise AdapterError(f"unknown target model family: {family!r}")

    try:
        tokenizer, _metadata = load_artifact(polish_artifact_dir)
    except Exception as err:
        raise AdapterError(
            f"failed to load Polish artifact from {polish_artifact_dir!r}: "
            f"{err}"
        ) from err

    added = _vocab_in_id_order(tokenizer)
    return AdapterResult(
        tokenizer=tokenizer,
        added_tokens=added,
        base_vocab_size=0,
    )


def _vocab_in_id_order(tokenizer: Tokenizer) -> tuple[str, ...]:
    """Return every vocab token string ordered by ascending id."""
    vocab: dict[str, int] = tokenizer.get_vocab()
    return tuple(tok for tok, _ in sorted(vocab.items(), key=lambda kv: kv[1]))
