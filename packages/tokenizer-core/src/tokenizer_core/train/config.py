"""Typed, validated schema for a language's ``recipe.yaml``.

A *recipe* is the committed source of truth for one language's tokenizer(s):
which corpora to train and evaluate on, the candidate algorithms and
hyperparameters, the fertility/quality targets, and the baseline tokenizers to
beat. This module models that structure with Pydantic v2 so a typo or wrong
type is caught up front — via :meth:`Recipe.load` — rather than deep in
corpus acquisition or training.

Every model forbids unknown fields (``extra="forbid"``) so a stray or
misspelled key fails validation instead of being silently dropped. Only base
dependencies (``pydantic``, ``pyyaml``) are used; no heavy training extras are
imported here.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

Source = Literal["hf", "url", "local"]
Domain = Literal[
    "general", "technical", "web", "wiki", "legal", "conversational"
]
Algorithm = Literal["bpe", "unigram", "superbpe"]
NormalizationForm = Literal["NFC", "NFD", "NFKC", "NFKD"]


class RecipeError(ValueError):
    """Raised when a recipe file is missing, unparseable, or invalid."""


class Language(BaseModel):
    """The language a recipe describes.

    Attributes:
        code: ISO 639-1 language code (e.g. ``pl``).
        name: Human-readable language name.
        scripts: ISO 15924 scripts the language is written in.
    """

    model_config = ConfigDict(extra="forbid")

    code: str
    name: str
    scripts: list[str]


class CorpusSpec(BaseModel):
    """A single corpus used for training or evaluation.

    Attributes:
        source: Where the corpus is resolved from (``hf``, ``url``, ``local``).
        id: The corpus identifier (Hugging Face dataset id, URL, or path).
        config: Optional dataset config/subset name (needed for ``hf``).
        split: Optional dataset split (e.g. ``train``, ``test``).
        weight: Positive sampling weight controlling the corpus mix.
        domain: Coarse content domain used for routing experiments.
    """

    model_config = ConfigDict(extra="forbid")

    source: Source
    id: str
    config: str | None = None
    split: str | None = None
    weight: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    domain: Domain = "general"


class Candidate(BaseModel):
    """A candidate tokenizer to train and compare.

    Attributes:
        name: Human-readable candidate name (e.g. ``pl-bpe-32k``).
        algorithm: Training algorithm (``bpe``, ``unigram``, ``superbpe``).
        vocab_size: Target vocabulary size (positive).
        byte_fallback: Whether to enable byte fallback for OOV coverage.
        normalization: Unicode normalization form applied before training.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    algorithm: Algorithm
    vocab_size: int = Field(gt=0)
    byte_fallback: bool = True
    normalization: NormalizationForm = "NFC"


class Targets(BaseModel):
    """The efficiency and quality targets a winning candidate must meet.

    Attributes:
        max_fertility_vs_english: Maximum allowed fertility relative to the
            English baseline on aligned eval content (positive).
        max_quality_drop_pct: Maximum tolerated downstream quality drop, in
            percent, when the tokenizer is transplanted (non-negative).
    """

    model_config = ConfigDict(extra="forbid")

    max_fertility_vs_english: float = Field(gt=0, allow_inf_nan=False)
    max_quality_drop_pct: float = Field(ge=0, allow_inf_nan=False)


class Recipe(BaseModel):
    """A complete, validated language tokenizer recipe.

    Attributes:
        language: The language this recipe describes.
        train_corpora: Corpora used to train candidate tokenizers (non-empty).
        eval_corpora: Corpora held out for benchmarking (non-empty).
        candidates: Candidate tokenizers to train and compare (non-empty).
        targets: The efficiency/quality targets to check against.
        baselines: Baseline tokenizers to beat (non-empty).
    """

    model_config = ConfigDict(extra="forbid")

    language: Language
    train_corpora: list[CorpusSpec] = Field(min_length=1)
    eval_corpora: list[CorpusSpec] = Field(min_length=1)
    candidates: list[Candidate] = Field(min_length=1)
    targets: Targets
    baselines: list[str] = Field(min_length=1)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> Recipe:
        """Load and validate a recipe from a YAML file.

        Args:
            path: Filesystem path to the recipe YAML file.

        Returns:
            The validated :class:`Recipe`.

        Raises:
            RecipeError: If the file is missing, cannot be read or decoded as
                UTF-8, is not valid YAML, is not a YAML mapping, or fails schema
                validation. The path is included in the message and the
                underlying error is chained.
        """
        p = Path(path)
        if not p.is_file():
            raise RecipeError(f"recipe not found: {p}")
        try:
            text = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as err:
            raise RecipeError(f"cannot read recipe {p}: {err}") from err
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as err:
            raise RecipeError(f"recipe {p} is not valid YAML: {err}") from err
        if not isinstance(data, dict):
            raise RecipeError(f"recipe {p} must be a YAML mapping")
        try:
            return cls.model_validate(data)
        except ValidationError as err:
            raise RecipeError(f"invalid recipe {p}:\n{err}") from err
