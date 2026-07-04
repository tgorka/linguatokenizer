"""Tests for the tokenizer.json + metadata.json artifact save/load contract.

Uses a minimal fake tokenizer object (``save``/``from_file``) injected into
``sys.modules`` as a fake ``tokenizers`` module, so these tests never need the
real ``tokenizers`` extra installed, per the repo's CI (``uv sync
--all-packages``, no extras).
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from tokenizer_core.train.artifact import (
    TrainerMetadata,
    load_artifact,
    save_artifact,
)


class _FakeTokenizer:
    """Minimal stand-in for ``tokenizers.Tokenizer``: records ``save`` calls
    and can be reconstructed via ``from_file``."""

    def __init__(self, payload: str = "fake-tokenizer-payload") -> None:
        self.payload = payload
        self.saved_to: str | None = None

    def save(self, path: str) -> None:
        self.saved_to = path
        Path(path).write_text(
            json.dumps({"payload": self.payload}), encoding="utf-8"
        )

    @classmethod
    def from_file(cls, path: str) -> _FakeTokenizer:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(payload=data["payload"])


def _make_fake_tokenizers_module() -> types.SimpleNamespace:
    return types.SimpleNamespace(Tokenizer=_FakeTokenizer)


def _metadata(**overrides: object) -> TrainerMetadata:
    fields: dict[str, object] = {
        "name": "pl-bpe-32k",
        "algorithm": "bpe",
        "vocab_size": 100,
        "byte_fallback": True,
        "normalization": "NFC",
        "corpus_hash": "deadbeef",
        "seed": 42,
        "language_code": "pl",
    }
    fields.update(overrides)
    return TrainerMetadata(**fields)  # type: ignore[arg-type]


def test_metadata_as_dict_round_trips_all_fields() -> None:
    metadata = _metadata()
    as_dict = metadata.as_dict()

    assert as_dict == {
        "name": "pl-bpe-32k",
        "algorithm": "bpe",
        "vocab_size": 100,
        "byte_fallback": True,
        "normalization": "NFC",
        "corpus_hash": "deadbeef",
        "seed": 42,
        "language_code": "pl",
    }


def test_save_artifact_writes_both_files(tmp_path: Path) -> None:
    tokenizer = _FakeTokenizer()
    metadata = _metadata()
    out_dir = tmp_path / "artifact"

    tokenizer_path, metadata_path = save_artifact(
        tokenizer,  # ty: ignore[invalid-argument-type]
        metadata,
        out_dir,
    )

    assert tokenizer_path == out_dir / "tokenizer.json"
    assert metadata_path == out_dir / "metadata.json"
    assert tokenizer_path.is_file()
    assert metadata_path.is_file()
    assert tokenizer.saved_to == str(tokenizer_path)

    on_disk_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert on_disk_metadata == metadata.as_dict()


def test_save_artifact_creates_missing_parent_dirs(tmp_path: Path) -> None:
    tokenizer = _FakeTokenizer()
    metadata = _metadata()
    out_dir = tmp_path / "a" / "b" / "c"

    tokenizer_path, metadata_path = save_artifact(
        tokenizer,  # ty: ignore[invalid-argument-type]
        metadata,
        out_dir,
    )

    assert tokenizer_path.is_file()
    assert metadata_path.is_file()


def test_load_artifact_round_trip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(
        sys.modules, "tokenizers", _make_fake_tokenizers_module()
    )

    original_tokenizer = _FakeTokenizer(payload="hello-world")
    original_metadata = _metadata(name="pl-unigram-32k", algorithm="unigram")
    out_dir = tmp_path / "artifact"
    save_artifact(
        original_tokenizer,  # ty: ignore[invalid-argument-type]
        original_metadata,
        out_dir,
    )

    loaded_tokenizer, loaded_metadata = load_artifact(out_dir)

    assert isinstance(loaded_tokenizer, _FakeTokenizer)
    assert loaded_tokenizer.payload == "hello-world"
    assert loaded_metadata.as_dict() == original_metadata.as_dict()
    assert loaded_metadata == original_metadata
