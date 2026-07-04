"""Weighted, seeded corpus mixing and provenance manifests.

Draws a deterministic, reproducible mix of documents across a recipe's
``CorpusSpec`` entries — via normalized weight shares and reservoir sampling
— and records what was drawn (per-source counts + content hashes) in a JSON
manifest under the ``$TOKENIZER_DATA_DIR`` data plane, so downstream training
and benchmarking can trust exactly what a run saw.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections.abc import Iterator
from dataclasses import dataclass

from tokenizer_core.corpora.sources import (
    SourceError,
    _resolve_data_dir,
    resolve_source,
)
from tokenizer_core.train.config import CorpusSpec


@dataclass(frozen=True)
class SourceManifestEntry:
    """Provenance record for one corpus's contribution to a sampled mix.

    Attributes:
        source: Where the corpus is resolved from (``hf``, ``url``, ``local``).
        id: The corpus identifier (Hugging Face dataset id, URL, or path).
        config: Optional dataset config/subset name.
        split: Optional dataset split.
        domain: Coarse content domain.
        weight: The corpus's declared sampling weight.
        n_docs_quota: Documents requested from this source by its weight
            share, before sampling.
        n_docs_sampled: Number of documents actually drawn from this source.
            May be less than ``n_docs_quota`` when the source is ``skipped``
            or its stream held fewer documents than the quota.
        skipped: Whether the source was gracefully skipped (e.g. gated hf).
        skip_reason: Human-readable reason when ``skipped`` is ``True``, else
            ``None``.
        content_sha256: sha256 hex digest of this source's sampled documents
            (order-sensitive, collision-safe across document boundaries).
    """

    source: str
    id: str
    config: str | None
    split: str | None
    domain: str
    weight: float
    n_docs_quota: int
    n_docs_sampled: int
    skipped: bool
    skip_reason: str | None
    content_sha256: str

    def as_dict(self) -> dict[str, object]:
        """Return the entry as a plain JSON-serializable dict."""
        return {
            "source": self.source,
            "id": self.id,
            "config": self.config,
            "split": self.split,
            "domain": self.domain,
            "weight": self.weight,
            "n_docs_quota": self.n_docs_quota,
            "n_docs_sampled": self.n_docs_sampled,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "content_sha256": self.content_sha256,
        }


@dataclass(frozen=True)
class CorpusManifest:
    """A reproducible record of a seeded, weighted corpus mix.

    Attributes:
        language_code: ISO 639-1 language code the mix was drawn for.
        seed: The random seed used for reservoir sampling.
        n_docs_requested: The total document count requested (``n_docs``).
        n_docs_sampled: The total document count actually drawn across all
            sources (may be less than ``n_docs_requested`` when sources are
            skipped, since skipped shares are not redistributed).
        sources: Per-source manifest entries, in input order.
    """

    language_code: str
    seed: int
    n_docs_requested: int
    n_docs_sampled: int
    sources: tuple[SourceManifestEntry, ...]

    def as_dict(self) -> dict[str, object]:
        """Return the manifest as a plain JSON-serializable dict."""
        return {
            "language_code": self.language_code,
            "seed": self.seed,
            "n_docs_requested": self.n_docs_requested,
            "n_docs_sampled": self.n_docs_sampled,
            "sources": [entry.as_dict() for entry in self.sources],
        }


def _reservoir(stream: Iterator[str], k: int, rng: random.Random) -> list[str]:
    """Draw ``k`` items from ``stream`` via reservoir sampling (algorithm R).

    Args:
        stream: An iterator of documents of unknown/unbounded length.
        k: Number of items to draw (0 draws nothing).
        rng: Seeded random source, so the draw is deterministic.

    Returns:
        A list of up to ``k`` sampled documents (fewer if the stream was
        shorter than ``k``).
    """
    if k <= 0:
        return []
    sample: list[str] = []
    for i, doc in enumerate(stream):
        if i < k:
            sample.append(doc)
        else:
            j = rng.randint(0, i)
            if j < k:
                sample[j] = doc
    return sample


def _content_hash(docs: list[str]) -> str:
    r"""Return the sha256 hex digest of ``docs`` (order-sensitive).

    Documents are length-prefixed before hashing so no separator choice can
    make two distinct document lists collide (e.g. a document containing an
    embedded newline would collide under a naive ``"\n".join``).
    """
    hasher = hashlib.sha256()
    for doc in docs:
        encoded = doc.encode("utf-8")
        hasher.update(len(encoded).to_bytes(8, "big"))
        hasher.update(encoded)
    return hasher.hexdigest()


def sample_corpora(
    corpora: list[CorpusSpec],
    *,
    n_docs: int,
    seed: int,
    language_code: str,
    data_dir: str | os.PathLike[str] | None = None,
) -> tuple[CorpusManifest, list[str]]:
    """Draw a deterministic, seeded, weighted mix across ``corpora``.

    Weight shares are normalized across *all* sources passed in (including
    ones later skipped). Each source's document quota is
    ``k_i = floor(share_i * n_docs)``; the remainder
    ``n_docs - sum(k_i)`` is handed out one-by-one to sources in
    descending-weight order (ties broken by input order), so totals sum
    exactly to ``n_docs`` when nothing is skipped. A skipped source
    contributes 0 docs and its slice is not redistributed.

    Args:
        corpora: The corpus specs to mix (typically a recipe's
            ``train_corpora`` or ``eval_corpora``).
        n_docs: Total number of documents requested across all sources.
        seed: Random seed for reservoir sampling; identical seed and source
            content reproduce an identical mix.
        language_code: ISO 639-1 language code the manifest is written under.
        data_dir: Data-plane root directory. Defaults to
            ``$TOKENIZER_DATA_DIR`` when omitted.

    Returns:
        A ``(manifest, documents)`` tuple: the :class:`CorpusManifest` (also
        written to ``{data_dir}/corpora/{language_code}/manifest.json``) and
        the actual sampled documents, concatenated in source order -- exactly
        the text each source's ``content_sha256`` was hashed from.

    Raises:
        ValueError: If ``corpora`` is empty or ``n_docs`` is negative.
        SourceError: If the manifest cannot be written to ``data_dir``
            (e.g. permission denied).
    """
    if not corpora:
        raise ValueError("sample_corpora requires at least one CorpusSpec")
    if n_docs < 0:
        raise ValueError(f"n_docs must be >= 0, got {n_docs}")

    resolved_dir = _resolve_data_dir(data_dir)

    total_weight = sum(spec.weight for spec in corpora)
    shares = [spec.weight / total_weight for spec in corpora]
    quotas = [int(share * n_docs) for share in shares]

    remainder = n_docs - sum(quotas)
    order_by_weight = sorted(
        range(len(corpora)), key=lambda i: (-corpora[i].weight, i)
    )
    for i in order_by_weight[:remainder]:
        quotas[i] += 1

    entries: list[SourceManifestEntry] = []
    all_sampled: list[str] = []
    n_docs_sampled = 0
    for spec, k in zip(corpora, quotas, strict=True):
        rng = random.Random(seed)
        documents, skip_reason = resolve_source(spec, data_dir=resolved_dir)
        skipped = skip_reason is not None
        sampled = [] if skipped else _reservoir(documents, k, rng)
        n_docs_sampled += len(sampled)
        all_sampled.extend(sampled)
        entries.append(
            SourceManifestEntry(
                source=spec.source,
                id=spec.id,
                config=spec.config,
                split=spec.split,
                domain=spec.domain,
                weight=spec.weight,
                n_docs_quota=k,
                n_docs_sampled=len(sampled),
                skipped=skipped,
                skip_reason=skip_reason,
                content_sha256=_content_hash(sampled),
            )
        )

    manifest = CorpusManifest(
        language_code=language_code,
        seed=seed,
        n_docs_requested=n_docs,
        n_docs_sampled=n_docs_sampled,
        sources=tuple(entries),
    )

    manifest_path = resolved_dir / "corpora" / language_code / "manifest.json"
    try:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest.as_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except OSError as err:
        raise SourceError(
            f"failed to write corpus manifest to {manifest_path}"
        ) from err

    return manifest, all_sampled
