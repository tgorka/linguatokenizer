"""Corpus acquisition and sampling — the data plane for a recipe's corpora.

Re-exports source resolution and weighted sampling so
``tokenizer_core.corpora`` is the documented import surface for downstream
stories.
"""

from __future__ import annotations

from tokenizer_core.corpora.sampling import (
    CorpusManifest,
    SourceManifestEntry,
    sample_corpora,
)
from tokenizer_core.corpora.sources import SourceError, resolve_source

__all__ = [
    "CorpusManifest",
    "SourceError",
    "SourceManifestEntry",
    "resolve_source",
    "sample_corpora",
]
