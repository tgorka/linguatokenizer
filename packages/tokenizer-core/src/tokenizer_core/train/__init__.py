"""Recipe schema and training config — the typed source of truth for a language.

Re-exports the recipe models and loader so ``tokenizer_core.train`` is the
documented import surface for downstream stories.
"""

from __future__ import annotations

from tokenizer_core.train.config import (
    Candidate,
    CorpusSpec,
    Language,
    Recipe,
    RecipeError,
    Targets,
)

__all__ = [
    "Candidate",
    "CorpusSpec",
    "Language",
    "Recipe",
    "RecipeError",
    "Targets",
]
