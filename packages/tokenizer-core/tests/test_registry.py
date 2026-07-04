"""Tests for the target-model registry (``adapters.registry``).

The ``transformers``-backed tokenizer loading is exercised via a fake module
injected into ``sys.modules`` (mirroring ``test_compare.py``'s
``_fake_transformers_module`` pattern), so these tests never need the real
``adapters`` extra or network access.
"""

from __future__ import annotations

import sys
import types

import pytest
from pydantic import ValidationError

from tokenizer_core.adapters.registry import (
    REGISTRY,
    AdapterError,
    TargetModelEntry,
    load_target_tokenizer,
)

_VALIDATED_FAMILIES = {"gpt2", "llama", "mistral", "qwen", "gemma", "deepseek"}
_FORMAT_READY_FAMILIES = {"glm", "nemotron", "kimi", "mimo"}


def _fake_transformers_module(
    tokenizers_by_id: dict[str, object] | None = None,
    error_ids: set[str] | None = None,
):
    """Fake ``transformers`` module for ``sys.modules`` injection."""
    tokenizers_by_id = tokenizers_by_id or {}
    error_ids = error_ids or set()

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(name: str, token: str | None = None) -> object:
            if name in error_ids:
                raise RuntimeError(f"gated repo {name!r}, terms not accepted")
            return tokenizers_by_id.get(name, types.SimpleNamespace(name=name))

    return types.SimpleNamespace(AutoTokenizer=_FakeAutoTokenizer)


# --- REGISTRY shape ----------------------------------------------------------


def test_registry_has_exactly_the_expected_family_keys() -> None:
    assert set(REGISTRY) == _VALIDATED_FAMILIES | _FORMAT_READY_FAMILIES


def test_registry_validated_today_split() -> None:
    for family in _VALIDATED_FAMILIES:
        assert REGISTRY[family].validated_today is True
    for family in _FORMAT_READY_FAMILIES:
        assert REGISTRY[family].validated_today is False


def test_registry_entries_expose_expected_fields() -> None:
    for entry in REGISTRY.values():
        assert entry.format in ("hf_tokenizers", "sentencepiece", "tiktoken")
        assert entry.vocab_size is None or isinstance(entry.vocab_size, int)
        assert entry.special_tokens is None or isinstance(
            entry.special_tokens, dict
        )
        assert isinstance(entry.validated_today, bool)


def test_registry_entry_family_matches_its_own_dict_key() -> None:
    for key, entry in REGISTRY.items():
        assert entry.family == key


# --- TargetModelEntry validation ---------------------------------------------


def test_target_model_entry_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        TargetModelEntry.model_validate(
            {
                "family": "gpt2",
                "hf_id": "gpt2",
                "format": "hf_tokenizers",
                "vocab_size": 50257,
                "special_tokens": None,
                "validated_today": True,
                "unknown_field": 1,
            }
        )


# --- load_target_tokenizer: success path -------------------------------------


def test_load_target_tokenizer_success(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = types.SimpleNamespace(name="fake-gpt2-tokenizer")
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        _fake_transformers_module(tokenizers_by_id={"gpt2": sentinel}),
    )

    tokenizer = load_target_tokenizer("gpt2")

    assert tokenizer is sentinel


# --- load_target_tokenizer: unknown family -----------------------------------


def test_load_target_tokenizer_unknown_family_raises_without_importing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Ensure any accidental import attempt would blow up loudly rather than
    # silently succeed, proving the unknown-family check happens first.
    monkeypatch.setitem(sys.modules, "transformers", None)

    with pytest.raises(AdapterError, match="falcon"):
        load_target_tokenizer("falcon")


# --- load_target_tokenizer: registered family, load fails --------------------


def test_load_target_tokenizer_load_failure_chains_original_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gemma_hf_id = REGISTRY["gemma"].hf_id
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        _fake_transformers_module(error_ids={gemma_hf_id}),
    )

    with pytest.raises(AdapterError) as exc_info:
        load_target_tokenizer("gemma")

    assert "gemma" in str(exc_info.value)
    assert gemma_hf_id in str(exc_info.value)
    assert exc_info.value.__cause__ is not None
    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_load_target_tokenizer_format_ready_family_load_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kimi_hf_id = REGISTRY["kimi"].hf_id
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        _fake_transformers_module(error_ids={kimi_hf_id}),
    )

    with pytest.raises(AdapterError) as exc_info:
        load_target_tokenizer("kimi")

    assert "kimi" in str(exc_info.value)
    assert exc_info.value.__cause__ is not None


# --- load_target_tokenizer: transformers extra not installed -----------------


def test_load_target_tokenizer_missing_extra_raises_adapter_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "transformers", None)

    with pytest.raises(AdapterError) as exc_info:
        load_target_tokenizer("gpt2")

    assert exc_info.value.__cause__ is not None
    assert isinstance(exc_info.value.__cause__, ImportError)
