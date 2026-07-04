"""Resolve a :class:`~tokenizer_core.train.config.CorpusSpec` into text.

A recipe's ``train_corpora``/``eval_corpora`` declare corpora abstractly
(``hf``/``url``/``local`` plus an id); this module turns one such declaration
into an actual stream of text documents, caching ``url`` downloads under the
``$TOKENIZER_DATA_DIR`` data plane so repeated runs don't re-fetch.

The heavy ``datasets``/``huggingface-hub`` dependencies (the ``corpora``
extra) are imported lazily inside :func:`_resolve_hf` so the base install —
and this module's import — stay light.
"""

from __future__ import annotations

import hashlib
import itertools
import os
import urllib.request
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from tokenizer_core.train.config import CorpusSpec


class SourceError(RuntimeError):
    """Raised when a corpus source cannot be resolved into text."""


def _resolve_data_dir(data_dir: str | os.PathLike[str] | None) -> Path:
    """Resolve the data-plane directory, defaulting to the env var.

    Args:
        data_dir: Explicit data directory, or ``None`` to fall back to
            ``$TOKENIZER_DATA_DIR``.

    Returns:
        The resolved data directory as a :class:`~pathlib.Path`.

    Raises:
        SourceError: If ``data_dir`` is ``None`` and ``$TOKENIZER_DATA_DIR``
            is also unset.
    """
    if data_dir is not None:
        return Path(data_dir)
    env_value = os.environ.get("TOKENIZER_DATA_DIR")
    if not env_value:
        raise SourceError(
            "data_dir was not given and $TOKENIZER_DATA_DIR is unset"
        )
    return Path(env_value)


def resolve_source(
    spec: CorpusSpec,
    *,
    data_dir: str | os.PathLike[str] | None = None,
) -> tuple[Iterator[str], str | None]:
    """Turn one :class:`CorpusSpec` into a stream of text documents.

    Args:
        spec: The corpus to resolve.
        data_dir: Data-plane root directory. Defaults to
            ``$TOKENIZER_DATA_DIR`` when omitted.

    Returns:
        A ``(documents, skip_reason)`` tuple. ``documents`` yields one
        non-empty, stripped line per document for ``url``/``local`` sources,
        or one example's text field for ``hf``. ``skip_reason`` is ``None``
        unless the source was gracefully skipped (gated/inaccessible ``hf``
        source with no credentials configured), in which case ``documents``
        is empty.

    Raises:
        SourceError: If ``spec.source`` is ``hf`` with no ``split``, if the
            ``hf`` load fails and credentials ARE configured, if a ``url``
            cannot be fetched, if a ``local`` path does not exist, or (on
            first iteration, not at call time) if a ``local``/cached ``url``
            file is not valid UTF-8.

    Note:
        Only ``url`` sources are cached under ``data_dir``, so it is only
        resolved (and only needs to be resolvable — explicit or via
        ``$TOKENIZER_DATA_DIR``) for ``url`` specs. ``hf`` sources rely on
        ``datasets``'s own on-disk cache (``$HF_HOME``); ``local`` sources
        are already on disk and are read directly.
    """
    if spec.source == "hf":
        return _resolve_hf(spec)
    if spec.source == "url":
        return _resolve_url(spec, _resolve_data_dir(data_dir)), None
    return _resolve_local(spec), None


def _resolve_hf(spec: CorpusSpec) -> tuple[Iterator[str], str | None]:
    """Resolve an ``hf`` :class:`CorpusSpec` via streaming ``datasets``.

    See :func:`resolve_source` for the return contract. Any failure to load
    or iterate the dataset is treated as a gated/inaccessible source (and
    gracefully skipped) unless ``HF_TOKEN``/``HUGGINGFACE_HUB_TOKEN`` is set,
    in which case it is raised as a real failure.
    """
    if spec.split is None:
        raise SourceError(
            f"hf source {spec.id!r} has no split configured (ambiguous)"
        )

    import datasets  # noqa: PLC0415

    try:
        ds: Iterable[Mapping[str, Any]] = datasets.load_dataset(
            spec.id, name=spec.config, split=spec.split, streaming=True
        )
        examples = _iter_hf_examples(ds)
        # Streaming datasets often only surface auth/gating failures lazily,
        # on the first fetch rather than at `load_dataset()` time. Force
        # that first fetch now so such failures still hit the except below.
        head = list(itertools.islice(examples, 1))
    except Exception as err:
        if not (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        ):
            return (
                iter(()),
                f"skipping gated/inaccessible hf source {spec.id!r}: "
                f"{err} (set HF_TOKEN to retry)",
            )
        raise SourceError(f"failed to load hf source {spec.id!r}") from err

    rest = _wrap_hf_iteration_errors(examples, spec)
    return itertools.chain(head, rest), None


def _wrap_hf_iteration_errors(
    examples: Iterator[str], spec: CorpusSpec
) -> Iterator[str]:
    """Re-raise a mid-stream ``hf`` iteration failure as ``SourceError``.

    The first-fetch probe in :func:`_resolve_hf` already applies the
    gated/inaccessible skip-vs-raise contract to failures at or before the
    first document. A stream that fails partway through — after already
    yielding real documents to the caller — can no longer be represented as
    a clean skip (``iter(())``), so any later failure is simply wrapped as
    the module's typed error instead of escaping as a raw exception.
    """
    try:
        yield from examples
    except Exception as err:
        raise SourceError(
            f"failed while iterating hf source {spec.id!r}"
        ) from err


def _iter_hf_examples(dataset: Iterable[Mapping[str, Any]]) -> Iterator[str]:
    """Yield the text field of each example in a streaming hf dataset.

    Uses the ``"text"`` column when present, otherwise falls back to the
    first ``str``-valued column.
    """
    for example in dataset:
        if "text" in example:
            yield example["text"]
            continue
        for value in example.values():
            if isinstance(value, str):
                yield value
                break


def _cache_path_for_url(url: str, data_dir: Path) -> Path:
    """Return the cache file path a ``url`` source's raw bytes live at."""
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return data_dir / "corpora" / "url" / f"{digest}.txt"


def _resolve_url(spec: CorpusSpec, data_dir: Path) -> Iterator[str]:
    """Fetch (and cache) a ``url`` source, streaming its non-empty lines."""
    cache_path = _cache_path_for_url(spec.id, data_dir)
    if not cache_path.is_file():
        try:
            with urllib.request.urlopen(spec.id) as response:
                raw = response.read()
        except Exception as err:
            raise SourceError(
                f"failed to fetch url source {spec.id!r}"
            ) from err
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(raw)

    return _iter_lines(cache_path)


def _resolve_local(spec: CorpusSpec) -> Iterator[str]:
    """Stream non-empty lines directly from a ``local`` file path.

    ``~`` and environment variables (e.g. ``${TOKENIZER_DATA_DIR}``) in the
    path are expanded first, so committed recipes can reference the data plane
    portably instead of hard-coding an absolute machine path.
    """
    path = Path(os.path.expandvars(spec.id)).expanduser()
    if not path.is_file():
        raise SourceError(f"local source path not found: {path}")
    return _iter_lines(path)


def _iter_lines(path: Path) -> Iterator[str]:
    """Yield non-empty, stripped lines from a UTF-8 text file.

    Raises:
        SourceError: If ``path`` cannot be read or is not valid UTF-8.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as err:
        raise SourceError(f"cannot read source file {path}: {err}") from err
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            yield stripped
