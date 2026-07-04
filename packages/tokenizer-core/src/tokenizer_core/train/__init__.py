"""Recipe schema, training config, and trainers.

The typed source of truth for a language plus the entry points that turn it
into a tokenizer. Re-exports the recipe models/loader and the trainer/
artifact functions so ``tokenizer_core.train`` is the documented import
surface for downstream stories.
"""

from __future__ import annotations

from tokenizer_core.train.artifact import (
    TrainerMetadata,
    load_artifact,
    save_artifact,
)
from tokenizer_core.train.config import (
    Candidate,
    CorpusSpec,
    Language,
    Recipe,
    RecipeError,
    Targets,
)
from tokenizer_core.train.trainers import TrainerError, train_candidate

__all__ = [
    "Candidate",
    "CorpusSpec",
    "Language",
    "Recipe",
    "RecipeError",
    "Targets",
    "TrainerError",
    "TrainerMetadata",
    "load_artifact",
    "save_artifact",
    "train_candidate",
]
