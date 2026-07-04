"""Training-free embedding initialization for grafted vocabulary tokens.

After :func:`~tokenizer_core.adapters.extend.extend` (or ``replace``) grafts a
Polish tokenizer onto a target model, the new Polish tokens have **no**
embeddings. This module initializes them **without retraining** via FVT (Fast
Vocabulary Transfer): each new token's embedding is the mean of the base
model's embeddings of the base sub-tokens the new token's surface string
decomposes into under the base tokenizer. Tokens the base can't cover fall back
to the mean of all base embeddings.

Base token rows are copied through unchanged, so the target model's behavior on
its original vocabulary (e.g. English for gpt2) is **identical by
construction** — the transplant is quality-preserving on the base language with
zero training. Adapting the *new* Polish embeddings for strong generative
Polish output is the optional short-retrain step (the gpt2 PoC), on top of this
free starting point.

``numpy`` only; the base embedding matrix is read from the model's
``safetensors`` weights (no ``torch``). All heavy deps are imported lazily
behind the ``adapters`` extra guard.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import numpy as np

    from tokenizer_core.adapters.base import AdapterResult

# Common names of the input token-embedding weight across model families.
_EMBED_WEIGHT_NAMES = (
    "wte.weight",  # gpt2
    "transformer.wte.weight",
    "model.embed_tokens.weight",  # llama / mistral / qwen / gemma
    "embed_tokens.weight",
    "tok_embeddings.weight",  # some llama exports
    "model.tok_embeddings.weight",
)

InitMethod = Literal["fvt", "mean"]


class EmbedError(RuntimeError):
    """Raised when embeddings cannot be read or transplanted."""


def read_token_embeddings(
    hf_id: str, *, token: str | None = None
) -> np.ndarray:
    """Read a model's input token-embedding matrix as a numpy array.

    Loads the model's ``safetensors`` weights directly (no ``torch``) and
    returns the first embedding weight found among :data:`_EMBED_WEIGHT_NAMES`.

    Args:
        hf_id: Hugging Face model id (e.g. ``gpt2``).
        token: Optional HF token for gated repos.

    Returns:
        A ``[vocab_size, hidden_dim]`` float array.

    Raises:
        EmbedError: If the weights can't be downloaded/opened or no known
            embedding weight name is present.
    """
    import numpy as np  # noqa: PLC0415
    from huggingface_hub import hf_hub_download  # noqa: PLC0415
    from safetensors.numpy import load_file  # noqa: PLC0415

    try:
        path = hf_hub_download(hf_id, "model.safetensors", token=token)
    except Exception as err:
        raise EmbedError(
            f"could not download safetensors for {hf_id!r}: {err}"
        ) from err
    tensors = load_file(path)
    for name in _EMBED_WEIGHT_NAMES:
        if name in tensors:
            return np.asarray(tensors[name], dtype=np.float32)
    raise EmbedError(
        f"no known token-embedding weight in {hf_id!r} "
        f"(looked for {list(_EMBED_WEIGHT_NAMES)})"
    )


def _surface(tokenizer: Any, token_id: int) -> str:
    """Decode a single token id back to its surface string (byte-level)."""
    return tokenizer.decode([token_id])


def fvt_embedding(
    surface: str,
    base_tokenizer: Any,
    base_embeddings: np.ndarray,
    *,
    fallback: np.ndarray,
) -> np.ndarray:
    """Return the FVT init embedding for one new token's surface string.

    The mean of the base model's embeddings of the base sub-tokens ``surface``
    encodes to; ``fallback`` (typically the global mean embedding) when the
    base tokenizer yields no ids.
    """
    import numpy as np  # noqa: PLC0415

    ids = base_tokenizer.encode(surface)
    ids = [i for i in ids if 0 <= i < base_embeddings.shape[0]]
    if not ids:
        return fallback
    return np.mean(base_embeddings[ids], axis=0)


def transplant_embeddings(
    result: AdapterResult,
    base_embeddings: np.ndarray,
    base_tokenizer: Any,
    *,
    method: InitMethod = "fvt",
) -> np.ndarray:
    """Build an extended embedding matrix for a grafted tokenizer.

    Args:
        result: The :class:`~tokenizer_core.adapters.base.AdapterResult` from
            an extend/replace adapter (its ``base_vocab_size`` and
            ``added_tokens`` define the id layout).
        base_embeddings: The target model's ``[base_vocab, dim]`` input
            embeddings (see :func:`read_token_embeddings`). Must have at least
            ``result.base_vocab_size`` rows.
        base_tokenizer: The target model's *original* tokenizer (used to
            decompose each new token's surface string for FVT).
        method: ``"fvt"`` (mean of base sub-token embeddings) or ``"mean"``
            (every new token initialized to the global mean embedding).

    Returns:
        A ``[base_vocab_size + len(added_tokens), dim]`` float array whose
        first ``base_vocab_size`` rows equal ``base_embeddings`` (unchanged)
        and whose remaining rows are the initialized Polish-token embeddings.

    Raises:
        EmbedError: If ``base_embeddings`` has fewer than ``base_vocab_size``
            rows.
    """
    import numpy as np  # noqa: PLC0415

    base_vocab = result.base_vocab_size
    if base_embeddings.shape[0] < base_vocab:
        raise EmbedError(
            f"base_embeddings has {base_embeddings.shape[0]} rows, "
            f"need at least base_vocab_size={base_vocab}"
        )

    base = base_embeddings[:base_vocab]
    global_mean = np.mean(base, axis=0)
    dim = base.shape[1]
    new_rows = np.empty((len(result.added_tokens), dim), dtype=base.dtype)

    for i in range(len(result.added_tokens)):
        if method == "mean":
            new_rows[i] = global_mean
            continue
        surface = _surface(result.tokenizer, base_vocab + i)
        new_rows[i] = fvt_embedding(
            surface, base_tokenizer, base, fallback=global_mean
        )

    return np.concatenate([base, new_rows], axis=0)
