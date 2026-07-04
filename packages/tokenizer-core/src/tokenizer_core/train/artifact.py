"""Portable tokenizer artifacts: ``tokenizer.json`` + a ``metadata.json`` sidecar.

A trained Hugging Face ``tokenizers.Tokenizer`` is only useful downstream
(Story 2.3's scoring, Epic 3's adapters) if it is saved alongside the
provenance that produced it -- which candidate, which algorithm/vocab size/
byte-fallback/normalization settings, which corpus (by hash), and which seed.
This module is that save/load contract: :func:`save_artifact` writes the
tokenizer via its own ``.save()`` and the metadata as plain JSON;
:func:`load_artifact` reverses both.

``tokenizers`` is imported lazily inside function bodies (not at module
level), matching ``corpora/sources.py``'s guard convention, so this module
stays importable (and ``task check`` stays green) with no ``train`` extra
installed.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tokenizers import Tokenizer

_TOKENIZER_FILENAME = "tokenizer.json"
_METADATA_FILENAME = "metadata.json"


@dataclass(frozen=True)
class TrainerMetadata:
    """Provenance recorded alongside a trained tokenizer artifact.

    Attributes:
        name: Candidate name (e.g. ``pl-bpe-32k``).
        algorithm: Training algorithm (``bpe`` or ``unigram``).
        vocab_size: Target vocabulary size the candidate was trained for.
        byte_fallback: Whether byte-fallback coverage was enabled.
        normalization: Unicode normalization form applied before training.
        corpus_hash: Content hash of the sampled training corpus (from the
            :class:`~tokenizer_core.corpora.sampling.CorpusManifest`), so a
            run's saved artifact can be tied back to exactly what it saw.
        seed: Random seed used for corpus sampling.
        language_code: ISO 639-1 language code the candidate was trained for.
    """

    name: str
    algorithm: str
    vocab_size: int
    byte_fallback: bool
    normalization: str
    corpus_hash: str
    seed: int
    language_code: str

    def as_dict(self) -> dict[str, object]:
        """Return the metadata as a plain JSON-serializable dict."""
        return asdict(self)


def save_artifact(
    tokenizer: Tokenizer,
    metadata: TrainerMetadata,
    out_dir: str | os.PathLike[str],
) -> tuple[Path, Path]:
    """Save a trained tokenizer and its metadata under ``out_dir``.

    Args:
        tokenizer: The trained Hugging Face ``tokenizers.Tokenizer``.
        metadata: The :class:`TrainerMetadata` describing how it was trained.
        out_dir: Destination directory (created if missing).

    Returns:
        A ``(tokenizer_path, metadata_path)`` tuple of the two files written.
    """
    dest_dir = Path(out_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_path = dest_dir / _TOKENIZER_FILENAME
    tokenizer.save(str(tokenizer_path))

    metadata_path = dest_dir / _METADATA_FILENAME
    metadata_path.write_text(
        json.dumps(metadata.as_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return tokenizer_path, metadata_path


def load_artifact(
    artifact_dir: str | os.PathLike[str],
) -> tuple[Tokenizer, TrainerMetadata]:
    """Load a tokenizer and its metadata previously written by :func:`save_artifact`.

    Args:
        artifact_dir: Directory containing ``tokenizer.json`` and
            ``metadata.json``.

    Returns:
        A ``(tokenizer, metadata)`` tuple.

    Raises:
        FileNotFoundError: If either file is missing under ``artifact_dir``.
    """
    import tokenizers  # noqa: PLC0415

    dir_path = Path(artifact_dir)
    tokenizer_path = dir_path / _TOKENIZER_FILENAME
    metadata_path = dir_path / _METADATA_FILENAME

    tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))
    metadata_dict = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata = TrainerMetadata(**metadata_dict)

    return tokenizer, metadata
