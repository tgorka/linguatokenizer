"""Static target-model registry: format/vocab/special-tokens metadata.

Epic 3's adapters (extend/replace/OMP embedding init) need to know each
target model's tokenizer format, vocab size, and special tokens before they
can graft the Polish tokenizer onto it. This module is that lookup: a
static, dependency-light :data:`REGISTRY` mapping a model family name to a
:class:`TargetModelEntry`, plus :func:`load_target_tokenizer` -- a single-call
loader that resolves a family to a live tokenizer object via the same
lazy-import-behind-guard convention as
:func:`tokenizer_core.benchmark.compare._load_baseline_encoder`.

``REGISTRY`` is intentionally independent of any ``languages/<iso>/recipe.
yaml`` -- it is cross-language, cross-recipe model metadata (which tokenizer
format a family uses), whereas a recipe's ``baselines`` names specific HF
checkpoints to benchmark against for one language.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from tokenizer_core.benchmark.fertility import Encoder

TokenizerFormat = Literal["hf_tokenizers", "sentencepiece", "tiktoken"]


class AdapterError(ValueError):
    """Raised when a target model's tokenizer cannot be resolved or loaded."""


class TargetModelEntry(BaseModel):
    """Metadata describing one target model family's tokenizer.

    Attributes:
        family: Short registry key for the model family (e.g. ``gpt2``).
        hf_id: Hugging Face model id to load the tokenizer from.
        format: The tokenizer format/backend the family uses.
        vocab_size: The family's tokenizer vocabulary size, or ``None`` if
            not confidently known.
        special_tokens: Mapping of special-token role (``bos``, ``eos``,
            ``pad``, ``unk``, ...) to literal token string, or ``None`` if
            not confidently known.
        validated_today: Whether this family has been confirmed to load and
            work with the current adapter code today (``True``), versus
            being format-ready metadata for a family not yet validated
            end-to-end (``False``).
    """

    model_config = ConfigDict(extra="forbid")

    family: str
    hf_id: str
    format: TokenizerFormat
    vocab_size: int | None
    special_tokens: dict[str, str] | None
    validated_today: bool


REGISTRY: dict[str, TargetModelEntry] = {
    "gpt2": TargetModelEntry(
        family="gpt2",
        hf_id="gpt2",
        format="hf_tokenizers",
        vocab_size=50257,
        special_tokens={"bos": "<|endoftext|>", "eos": "<|endoftext|>"},
        validated_today=True,
    ),
    "llama": TargetModelEntry(
        family="llama",
        hf_id="meta-llama/Llama-3.1-8B",
        format="hf_tokenizers",
        vocab_size=128256,
        special_tokens={
            "bos": "<|begin_of_text|>",
            "eos": "<|end_of_text|>",
        },
        validated_today=True,
    ),
    "mistral": TargetModelEntry(
        family="mistral",
        hf_id="mistralai/Mistral-7B-v0.1",
        format="sentencepiece",
        vocab_size=32000,
        special_tokens={"bos": "<s>", "eos": "</s>", "unk": "<unk>"},
        validated_today=True,
    ),
    "qwen": TargetModelEntry(
        family="qwen",
        hf_id="Qwen/Qwen2.5-7B",
        format="hf_tokenizers",
        vocab_size=151643,
        special_tokens={"bos": "<|endoftext|>", "eos": "<|endoftext|>"},
        validated_today=True,
    ),
    "gemma": TargetModelEntry(
        family="gemma",
        hf_id="google/gemma-7b",
        format="sentencepiece",
        vocab_size=256000,
        special_tokens={"bos": "<bos>", "eos": "<eos>", "unk": "<unk>"},
        validated_today=True,
    ),
    "deepseek": TargetModelEntry(
        family="deepseek",
        hf_id="deepseek-ai/DeepSeek-V3",
        format="hf_tokenizers",
        vocab_size=129280,
        special_tokens={
            "bos": "<︙begin▁of▁sentence︙>",
            "eos": "<︙end▁of▁sentence︙>",
        },
        validated_today=True,
    ),
    # -- Format-ready families (validated_today=False): metadata anchored to
    # the most recent real, publicly documented checkpoint in each family,
    # confirmed via WebFetch against the live Hugging Face config/tokenizer
    # config files rather than guessed. See Design Notes in
    # spec-3-1-target-model-registry.md.
    "glm": TargetModelEntry(
        family="glm",
        hf_id="zai-org/GLM-4.6",
        format="hf_tokenizers",
        vocab_size=151552,
        special_tokens={"eos": "<|endoftext|>", "pad": "<|endoftext|>"},
        validated_today=False,
    ),
    "nemotron": TargetModelEntry(
        family="nemotron",
        hf_id="nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        format="hf_tokenizers",
        vocab_size=131072,
        special_tokens={"bos": "<s>", "eos": "</s>", "unk": "<unk>"},
        validated_today=False,
    ),
    "kimi": TargetModelEntry(
        family="kimi",
        hf_id="moonshotai/Kimi-K2-Instruct",
        format="tiktoken",
        vocab_size=163840,
        special_tokens={"bos": "[BOS]", "eos": "[EOS]"},
        validated_today=False,
    ),
    "mimo": TargetModelEntry(
        family="mimo",
        hf_id="XiaomiMiMo/MiMo-7B-RL",
        format="hf_tokenizers",
        vocab_size=151680,
        special_tokens={"eos": "<|im_end|>", "pad": "<|endoftext|>"},
        validated_today=False,
    ),
}


def load_target_tokenizer(family: str) -> Encoder:
    """Load a live tokenizer object for a registered target model family.

    Args:
        family: A :data:`REGISTRY` key (e.g. ``gpt2``, ``llama``, ``kimi``).

    Returns:
        The loaded tokenizer object (a ``transformers`` ``AutoTokenizer``
        instance), usable directly as an
        :class:`~tokenizer_core.benchmark.fertility.Encoder`.

    Raises:
        AdapterError: If ``family`` is not a :data:`REGISTRY` key (raised
            before any import is attempted), or if the underlying
            ``transformers.AutoTokenizer.from_pretrained`` call fails for
            any reason (network error, gated repo, missing token, bad id,
            missing ``transformers`` extra, ...) -- the original exception
            is chained via ``from err``.
    """
    entry = REGISTRY.get(family)
    if entry is None:
        raise AdapterError(f"unknown target model family: {family!r}")

    try:
        import transformers  # noqa: PLC0415
    except ImportError as err:
        raise AdapterError(
            f"cannot load target tokenizer for family {family!r}: "
            f"the 'adapters' extra (transformers) is not installed: {err}"
        ) from err

    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            entry.hf_id, token=os.environ.get("HF_TOKEN")
        )
    except Exception as err:
        raise AdapterError(
            f"failed to load target tokenizer for family {family!r} "
            f"(hf_id={entry.hf_id!r}): {err}"
        ) from err

    # `AutoTokenizer.from_pretrained` raises rather than returning `None` in
    # every real `transformers` code path; this guard exists only to narrow
    # the stub-inferred `T | None` return type to this function's declared
    # `Encoder` return type, and to fail loudly rather than propagate `None`
    # in case a caller-supplied test double breaks that contract.
    if tokenizer is None:
        raise AdapterError(
            f"failed to load target tokenizer for family {family!r} "
            f"(hf_id={entry.hf_id!r}): from_pretrained returned None"
        )
    return tokenizer
