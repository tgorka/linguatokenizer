"""Tests for weighted, seeded corpus mixing and manifest provenance."""

from __future__ import annotations

import json
import random
import sys
import types
from pathlib import Path

import pytest

from tokenizer_core.corpora.sampling import _reservoir, sample_corpora
from tokenizer_core.corpora.sources import SourceError
from tokenizer_core.train.config import CorpusSpec


def _local_spec(
    path: Path, weight: float, domain: str = "general"
) -> CorpusSpec:
    return CorpusSpec.model_validate(
        {
            "source": "local",
            "id": str(path),
            "weight": weight,
            "domain": domain,
        }
    )


def _write_lines(path: Path, n: int) -> Path:
    path.write_text(
        "\n".join(f"doc {i}" for i in range(n)) + "\n", encoding="utf-8"
    )
    return path


def test_weighted_seeded_mix_matches_expected_quotas(tmp_path: Path):
    a = _local_spec(_write_lines(tmp_path / "a.txt", 50), weight=0.5)
    b = _local_spec(_write_lines(tmp_path / "b.txt", 50), weight=0.3)
    c = _local_spec(_write_lines(tmp_path / "c.txt", 50), weight=0.2)
    data_dir = tmp_path / "data"

    manifest, docs = sample_corpora(
        [a, b, c],
        n_docs=10,
        seed=7,
        language_code="xx",
        data_dir=data_dir,
    )

    assert [e.n_docs_sampled for e in manifest.sources] == [5, 3, 2]
    assert manifest.n_docs_requested == 10
    assert manifest.n_docs_sampled == 10
    assert len(docs) == 10

    manifest_path = data_dir / "corpora" / "xx" / "manifest.json"
    assert manifest_path.is_file()
    on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert on_disk["n_docs_sampled"] == 10

    # Reproducible: a second run with the same seed yields identical hashes.
    manifest2, docs2 = sample_corpora(
        [a, b, c],
        n_docs=10,
        seed=7,
        language_code="xx",
        data_dir=data_dir,
    )
    assert [e.content_sha256 for e in manifest.sources] == [
        e.content_sha256 for e in manifest2.sources
    ]
    assert docs == docs2


def test_mix_with_one_skipped_source_not_redistributed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    a = _local_spec(_write_lines(tmp_path / "a.txt", 50), weight=0.5)
    b = _local_spec(_write_lines(tmp_path / "b.txt", 50), weight=0.3)
    gated = CorpusSpec.model_validate(
        {
            "source": "hf",
            "id": "some/gated-corpus",
            "config": "xx",
            "split": "train",
            "weight": 0.2,
            "domain": "general",
        }
    )
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)

    def load_dataset(*args: object, **kwargs: object) -> None:
        raise RuntimeError("403 gated")

    fake_datasets = types.SimpleNamespace(load_dataset=load_dataset)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    data_dir = tmp_path / "data"
    manifest, _docs = sample_corpora(
        [a, b, gated],
        n_docs=10,
        seed=7,
        language_code="xx",
        data_dir=data_dir,
    )

    gated_entry = manifest.sources[2]
    assert gated_entry.skipped is True
    assert gated_entry.n_docs_sampled == 0
    assert gated_entry.skip_reason is not None

    assert manifest.n_docs_requested == 10
    assert manifest.n_docs_sampled < 10
    assert manifest.n_docs_sampled == sum(
        e.n_docs_sampled for e in manifest.sources
    )


def test_empty_corpora_raises_value_error(tmp_path: Path):
    with pytest.raises(ValueError, match="at least one"):
        sample_corpora(
            [], n_docs=10, seed=7, language_code="xx", data_dir=tmp_path
        )


def test_negative_n_docs_raises_value_error(tmp_path: Path):
    a = _local_spec(_write_lines(tmp_path / "a.txt", 5), weight=1.0)
    with pytest.raises(ValueError, match="n_docs"):
        sample_corpora(
            [a], n_docs=-1, seed=7, language_code="xx", data_dir=tmp_path
        )


def test_zero_quota_source_is_not_iterated(tmp_path: Path):
    """A source whose weight rounds its quota to 0 must not be drained."""
    a = _local_spec(_write_lines(tmp_path / "a.txt", 5), weight=0.999)
    tiny = _local_spec(_write_lines(tmp_path / "tiny.txt", 5), weight=0.001)

    manifest, _docs = sample_corpora(
        [a, tiny],
        n_docs=1,
        seed=7,
        language_code="xx",
        data_dir=tmp_path / "data",
    )

    tiny_entry = manifest.sources[1]
    assert tiny_entry.n_docs_quota == 0
    assert tiny_entry.n_docs_sampled == 0


def test_manifest_write_failure_raises_source_error(tmp_path: Path):
    """A manifest write failure (e.g. bad `data_dir`) must surface as the
    module's typed error, not a raw `OSError`."""
    a = _local_spec(_write_lines(tmp_path / "a.txt", 5), weight=1.0)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    # Pre-create `corpora` as a plain file so the manifest's parent
    # directory (`corpora/<lang>`) cannot be created under it.
    (data_dir / "corpora").write_text("not a directory", encoding="utf-8")

    with pytest.raises(SourceError, match="manifest"):
        sample_corpora(
            [a], n_docs=1, seed=1, language_code="xx", data_dir=data_dir
        )


def test_reservoir_zero_k_does_not_consume_stream():
    calls = 0

    def _counting_stream():
        nonlocal calls
        for i in range(10_000):
            calls += 1
            yield f"doc {i}"

    result = _reservoir(_counting_stream(), k=0, rng=random.Random(1))

    assert result == []
    assert calls == 0


def test_different_seed_changes_sampled_docs(tmp_path: Path):
    a = _local_spec(_write_lines(tmp_path / "a.txt", 50), weight=0.5)
    b = _local_spec(_write_lines(tmp_path / "b.txt", 50), weight=0.3)
    c = _local_spec(_write_lines(tmp_path / "c.txt", 50), weight=0.2)

    manifest_seed7, _docs7 = sample_corpora(
        [a, b, c],
        n_docs=10,
        seed=7,
        language_code="xx",
        data_dir=tmp_path / "data7",
    )
    manifest_seed8, _docs8 = sample_corpora(
        [a, b, c],
        n_docs=10,
        seed=8,
        language_code="xx",
        data_dir=tmp_path / "data8",
    )

    hashes_seed7 = [e.content_sha256 for e in manifest_seed7.sources]
    hashes_seed8 = [e.content_sha256 for e in manifest_seed8.sources]
    assert hashes_seed7 != hashes_seed8
