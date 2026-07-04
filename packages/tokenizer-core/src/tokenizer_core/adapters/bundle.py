"""Adapt-bundle export: a portable artifact a serving stack can load.

An "adapt bundle" is everything needed to run a target model with the adapted
Polish tokenizer:

    <bundle>/
      tokenizer.json        # the extended/replaced tokenizer
      embeddings.safetensors  # the transplanted [vocab, dim] input embeddings
      manifest.json         # base model, method, vocab boundary, provenance

The embedding matrix can be large, so bundles are written wherever the caller
points (typically the Nextcloud data plane, ``$TOKENIZER_DATA_DIR``), not into
git. ``numpy``/``safetensors`` are imported lazily behind the ``adapters``
extra guard.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from tokenizer_core.adapters.base import AdapterResult

_TOKENIZER_FILE = "tokenizer.json"
_EMBEDDINGS_FILE = "embeddings.safetensors"
_MANIFEST_FILE = "manifest.json"


def export_bundle(  # noqa: PLR0913  (each field is real bundle provenance)
    result: AdapterResult,
    embeddings: np.ndarray,
    out_dir: str | Path,
    *,
    base_model: str,
    method: str,
    target_fertility: float | None = None,
) -> Path:
    """Write an adapt bundle to ``out_dir``.

    Args:
        result: The extend/replace :class:`~tokenizer_core.adapters.base.
            AdapterResult`.
        embeddings: The transplanted ``[vocab, dim]`` embedding matrix (rows
            must match ``result.tokenizer``'s vocab size).
        out_dir: Destination directory (created if missing). Point this at the
            data plane for large matrices.
        base_model: The donor model id (e.g. ``gpt2``), recorded in the
            manifest.
        method: The adapter path used (e.g. ``extend+fvt``, ``replace+fvt``).
        target_fertility: Optional measured tokens/word to record.

    Returns:
        The bundle directory as a :class:`~pathlib.Path`.

    Raises:
        ValueError: If ``embeddings`` row count doesn't match the tokenizer's
            vocab size.
    """
    # Validate before importing the heavy dep so a shape mismatch fails fast
    # (and without needing the safetensors extra installed).
    vocab_size = result.tokenizer.get_vocab_size()
    if embeddings.shape[0] != vocab_size:
        raise ValueError(
            f"embeddings has {embeddings.shape[0]} rows but tokenizer vocab "
            f"is {vocab_size}"
        )

    from safetensors.numpy import save_file  # noqa: PLC0415

    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)

    result.tokenizer.save(str(dest / _TOKENIZER_FILE))
    save_file({"embeddings": embeddings}, str(dest / _EMBEDDINGS_FILE))

    manifest = {
        "base_model": base_model,
        "method": method,
        "vocab_size": vocab_size,
        "base_vocab_size": result.base_vocab_size,
        "added_tokens": len(result.added_tokens),
        "embedding_dim": int(embeddings.shape[1]),
        "target_fertility": target_fertility,
    }
    (dest / _MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return dest


def load_bundle(bundle_dir: str | Path) -> tuple[Any, np.ndarray, dict]:
    """Load an adapt bundle written by :func:`export_bundle`.

    Args:
        bundle_dir: The bundle directory.

    Returns:
        A ``(tokenizer, embeddings, manifest)`` tuple: the
        ``tokenizers.Tokenizer``, the ``[vocab, dim]`` embedding array, and the
        parsed manifest dict.
    """
    from safetensors.numpy import load_file  # noqa: PLC0415
    from tokenizers import Tokenizer  # noqa: PLC0415

    src = Path(bundle_dir)
    tokenizer = Tokenizer.from_file(str(src / _TOKENIZER_FILE))
    embeddings = load_file(str(src / _EMBEDDINGS_FILE))["embeddings"]
    manifest = json.loads((src / _MANIFEST_FILE).read_text(encoding="utf-8"))
    return tokenizer, embeddings, manifest
