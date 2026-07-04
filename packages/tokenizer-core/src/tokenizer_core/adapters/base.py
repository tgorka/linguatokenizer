"""Shared adapter return contract: :class:`AdapterResult`.

Both :func:`tokenizer_core.adapters.extend.extend` (Story 3.2) and Story 3.3's
``replace`` adapter hand back the same shape: the grafted/rebuilt tokenizer,
the new Polish token strings it gained, and the id boundary between "target's
original vocab" and "Polish-only additions". Story 3.4's OMP embedding
initializer consumes exactly this shape, so it is factored out here rather
than duplicated per-adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tokenizers import Tokenizer


@dataclass(frozen=True)
class AdapterResult:
    """Result of grafting a Polish tokenizer onto a target model's tokenizer.

    Attributes:
        tokenizer: The resulting merged/rebuilt ``tokenizers.Tokenizer``.
        added_tokens: The new Polish token strings introduced by the adapter,
            in ascending id order (i.e. ``added_tokens[0]`` is the token at
            id ``base_vocab_size``).
        base_vocab_size: The id boundary between the target's original vocab
            and the appended Polish-only tokens -- every id strictly less
            than this belongs to the target's original vocab; every id at or
            above it is one of ``added_tokens``.
    """

    tokenizer: Tokenizer
    added_tokens: tuple[str, ...]
    base_vocab_size: int
