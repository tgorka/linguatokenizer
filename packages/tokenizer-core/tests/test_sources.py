"""Tests for corpus source resolution (``hf``/``url``/``local``).

The ``hf`` path is exercised via a fake module injected into ``sys.modules``
so these tests never need the real ``datasets``/``huggingface-hub`` extra or
network access, per the repo's CI (``uv sync --all-packages``, no extras).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from tokenizer_core.corpora.sources import SourceError, resolve_source
from tokenizer_core.train.config import CorpusSpec


def _make_fake_datasets(
    examples: list[dict[str, object]] | None, error: Exception | None = None
) -> types.SimpleNamespace:
    """Build a fake ``datasets`` module for ``sys.modules`` injection."""

    def load_dataset(
        path: str,
        name: str | None = None,
        split: str | None = None,
        streaming: bool = True,
    ) -> list[dict[str, object]]:
        if error is not None:
            raise error
        return list(examples or [])

    return types.SimpleNamespace(load_dataset=load_dataset)


def _hf_spec(**overrides: object) -> CorpusSpec:
    fields: dict[str, object] = {
        "source": "hf",
        "id": "some/corpus",
        "config": "xx",
        "split": "train",
        "weight": 1.0,
        "domain": "general",
    }
    fields.update(overrides)
    return CorpusSpec.model_validate(fields)


# --- hf ----------------------------------------------------------------


def test_hf_valid_spec_yields_text_field(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    examples: list[dict[str, object]] = [{"text": f"doc {i}"} for i in range(5)]
    monkeypatch.setitem(sys.modules, "datasets", _make_fake_datasets(examples))

    documents, skip_reason = resolve_source(_hf_spec(), data_dir=tmp_path)

    assert skip_reason is None
    assert list(documents) == [f"doc {i}" for i in range(5)]


def test_hf_missing_text_field_falls_back_to_first_str_column(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    examples: list[dict[str, object]] = [
        {"sentence": "hello"},
        {"sentence": "world"},
    ]
    monkeypatch.setitem(sys.modules, "datasets", _make_fake_datasets(examples))

    documents, skip_reason = resolve_source(_hf_spec(), data_dir=tmp_path)

    assert skip_reason is None
    assert list(documents) == ["hello", "world"]


def test_hf_split_none_raises(tmp_path: Path):
    with pytest.raises(SourceError, match="split"):
        resolve_source(_hf_spec(split=None), data_dir=tmp_path)


def test_hf_gated_no_token_skips_gracefully(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        _make_fake_datasets(None, error=RuntimeError("403 gated")),
    )

    documents, skip_reason = resolve_source(_hf_spec(), data_dir=tmp_path)

    assert list(documents) == []
    assert skip_reason is not None
    assert "some/corpus" in skip_reason
    assert "HF_TOKEN" in skip_reason


def test_hf_gated_with_token_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        _make_fake_datasets(None, error=RuntimeError("403 gated")),
    )

    with pytest.raises(SourceError):
        resolve_source(_hf_spec(), data_dir=tmp_path)


def _make_lazily_failing_datasets(error: Exception) -> types.SimpleNamespace:
    """Fake ``datasets`` whose ``load_dataset`` succeeds but whose returned
    iterable raises on first iteration (mirrors a streaming auth failure)."""

    class _FailingIterable:
        def __iter__(self):
            raise error

    def load_dataset(
        path: str,
        name: str | None = None,
        split: str | None = None,
        streaming: bool = True,
    ) -> _FailingIterable:
        return _FailingIterable()

    return types.SimpleNamespace(load_dataset=load_dataset)


def test_hf_lazy_iteration_failure_no_token_skips_gracefully(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        _make_lazily_failing_datasets(RuntimeError("403 gated")),
    )

    documents, skip_reason = resolve_source(_hf_spec(), data_dir=tmp_path)

    assert list(documents) == []
    assert skip_reason is not None


def test_hf_lazy_iteration_failure_with_token_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        _make_lazily_failing_datasets(RuntimeError("403 gated")),
    )

    with pytest.raises(SourceError):
        resolve_source(_hf_spec(), data_dir=tmp_path)


def test_hf_iteration_failure_after_first_item_raises_source_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """A failure past the first item can't be a clean skip (docs already
    yielded), so it must surface as the module's typed error, not raw."""

    def _examples():
        yield {"text": "doc 0"}
        raise RuntimeError("connection reset")

    def load_dataset(
        path: str,
        name: str | None = None,
        split: str | None = None,
        streaming: bool = True,
    ):
        return _examples()

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(load_dataset=load_dataset),
    )

    documents, skip_reason = resolve_source(_hf_spec(), data_dir=tmp_path)

    assert skip_reason is None
    with pytest.raises(SourceError, match="some/corpus"):
        list(documents)


# --- url -----------------------------------------------------------------


def _local_url_spec(path: Path) -> CorpusSpec:
    return CorpusSpec.model_validate(
        {
            "source": "url",
            "id": f"file://{path}",
            "weight": 1.0,
            "domain": "general",
        }
    )


def test_url_valid_spec_yields_non_empty_lines(tmp_path: Path):
    source_file = tmp_path / "c.txt"
    source_file.write_text(
        "line one\nline two\n\nline three\n", encoding="utf-8"
    )
    data_dir = tmp_path / "data"

    documents, skip_reason = resolve_source(
        _local_url_spec(source_file), data_dir=data_dir
    )

    assert skip_reason is None
    assert list(documents) == ["line one", "line two", "line three"]
    cache_dir = data_dir / "corpora" / "url"
    assert cache_dir.is_dir()
    assert len(list(cache_dir.iterdir())) == 1


def test_url_repeated_fetch_reuses_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source_file = tmp_path / "c.txt"
    source_file.write_text("line one\n", encoding="utf-8")
    data_dir = tmp_path / "data"
    spec = _local_url_spec(source_file)

    documents, _ = resolve_source(spec, data_dir=data_dir)
    list(documents)

    # Remove the source file: a second resolve must not re-fetch it, only
    # read from the cache written on the first call.
    source_file.unlink()

    documents2, skip_reason = resolve_source(spec, data_dir=data_dir)
    assert skip_reason is None
    assert list(documents2) == ["line one"]


# --- local -----------------------------------------------------------------


def _local_spec(path: Path) -> CorpusSpec:
    return CorpusSpec.model_validate(
        {
            "source": "local",
            "id": str(path),
            "weight": 1.0,
            "domain": "general",
        }
    )


def test_local_valid_spec_yields_non_empty_lines_without_copy(tmp_path: Path):
    source_file = tmp_path / "local.txt"
    source_file.write_text("a\n\nb\nc\n", encoding="utf-8")
    data_dir = tmp_path / "data"

    documents, skip_reason = resolve_source(
        _local_spec(source_file), data_dir=data_dir
    )

    assert skip_reason is None
    assert list(documents) == ["a", "b", "c"]
    assert not data_dir.exists()


def test_local_missing_path_raises(tmp_path: Path):
    missing = tmp_path / "does-not-exist.txt"
    with pytest.raises(SourceError, match=str(missing)):
        resolve_source(_local_spec(missing), data_dir=tmp_path)


def test_local_non_utf8_file_raises_source_error(tmp_path: Path):
    source_file = tmp_path / "bad.txt"
    source_file.write_bytes(b"\xff\xfe not valid utf-8")

    documents, skip_reason = resolve_source(
        _local_spec(source_file), data_dir=tmp_path
    )
    assert skip_reason is None
    with pytest.raises(SourceError):
        list(documents)


def test_data_dir_defaults_to_env_var(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    source_file = tmp_path / "local.txt"
    source_file.write_text("a\n", encoding="utf-8")
    monkeypatch.setenv("TOKENIZER_DATA_DIR", str(tmp_path / "data"))

    documents, skip_reason = resolve_source(_local_spec(source_file))

    assert skip_reason is None
    assert list(documents) == ["a"]


def test_missing_data_dir_and_env_raises_for_url(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Only ``url`` sources actually need a resolvable ``data_dir``."""
    monkeypatch.delenv("TOKENIZER_DATA_DIR", raising=False)
    source_file = tmp_path / "c.txt"
    source_file.write_text("a\n", encoding="utf-8")

    with pytest.raises(SourceError, match="TOKENIZER_DATA_DIR"):
        resolve_source(_local_url_spec(source_file))


def test_hf_no_data_dir_and_no_env_still_resolves(
    monkeypatch: pytest.MonkeyPatch,
):
    """``hf`` sources never touch ``data_dir``, so it must not be required."""
    monkeypatch.delenv("TOKENIZER_DATA_DIR", raising=False)
    monkeypatch.setitem(
        sys.modules, "datasets", _make_fake_datasets([{"text": "doc"}])
    )

    documents, skip_reason = resolve_source(_hf_spec())

    assert skip_reason is None
    assert list(documents) == ["doc"]


def test_local_no_data_dir_and_no_env_still_resolves(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """``local`` sources never touch ``data_dir``, so it must not be required."""
    monkeypatch.delenv("TOKENIZER_DATA_DIR", raising=False)
    source_file = tmp_path / "local.txt"
    source_file.write_text("a\n", encoding="utf-8")

    documents, skip_reason = resolve_source(_local_spec(source_file))

    assert skip_reason is None
    assert list(documents) == ["a"]
