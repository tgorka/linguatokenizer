"""Tests for the recipe schema and loader against real and invalid recipes."""

from __future__ import annotations

from pathlib import Path

import pytest

from tokenizer_core.train import (
    Candidate,
    CorpusSpec,
    Language,
    Recipe,
    RecipeError,
    Targets,
)

# packages/tokenizer-core/tests/test_recipe_config.py -> repo root is four
# parents up.
REPO_ROOT = Path(__file__).resolve().parents[3]
PL_RECIPE = REPO_ROOT / "languages" / "pl" / "recipe.yaml"
TEMPLATE_RECIPE = REPO_ROOT / "languages" / "_template" / "recipe.yaml"


def test_load_polish_recipe():
    recipe = Recipe.load(PL_RECIPE)

    assert isinstance(recipe, Recipe)
    assert isinstance(recipe.language, Language)
    assert recipe.language.code == "pl"
    assert recipe.language.name == "Polish"
    assert recipe.language.scripts == ["Latn"]

    assert len(recipe.train_corpora) == 4
    first = recipe.train_corpora[0]
    assert isinstance(first, CorpusSpec)
    assert first.source == "hf"
    assert first.id == "oscar-corpus/OSCAR-2301"
    assert first.config == "pl"
    assert first.split == "train"
    assert first.weight == 0.5
    assert first.domain == "web"

    assert len(recipe.eval_corpora) == 3
    # Aligned FLORES pair as ungated local files (config names the FLORES side).
    assert recipe.eval_corpora[0].source == "local"
    assert recipe.eval_corpora[0].config == "pol_Latn"
    assert recipe.eval_corpora[1].config == "eng_Latn"

    assert len(recipe.candidates) == 3
    candidate = recipe.candidates[0]
    assert isinstance(candidate, Candidate)
    assert candidate.name == "pl-bpe-32k"
    assert candidate.algorithm == "bpe"
    assert candidate.vocab_size == 32000
    assert candidate.byte_fallback is True
    assert candidate.normalization == "NFC"
    assert recipe.candidates[2].algorithm == "superbpe"

    assert isinstance(recipe.targets, Targets)
    assert recipe.targets.max_fertility_vs_english == 1.0
    assert recipe.targets.max_quality_drop_pct == 1.0

    assert "gpt2" in recipe.baselines
    assert "speakleash/Bielik-11B-v3.0-Instruct" in recipe.baselines


def test_load_template_recipe():
    recipe = Recipe.load(TEMPLATE_RECIPE)

    assert isinstance(recipe, Recipe)
    assert recipe.language.code == "xx"
    assert recipe.language.name == "Example Language"
    assert recipe.language.scripts == ["Latn"]

    assert len(recipe.train_corpora) == 1
    assert recipe.train_corpora[0].config == "xx"
    assert recipe.train_corpora[0].domain == "general"

    assert len(recipe.eval_corpora) == 1
    assert len(recipe.candidates) == 2
    assert recipe.candidates[0].name == "xx-bpe-32k"
    assert recipe.targets.max_fertility_vs_english == 1.0
    assert recipe.baselines[0] == "gpt2"


def test_load_accepts_str_path():
    recipe = Recipe.load(str(PL_RECIPE))
    assert recipe.language.code == "pl"


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "recipe.yaml"
    path.write_text(text, encoding="utf-8")
    return path


_VALID_BODY = """\
language:
  code: xx
  name: Example
  scripts: [Latn]
train_corpora:
  - source: hf
    id: some/corpus
    config: xx
    split: train
    weight: 1.0
    domain: general
eval_corpora:
  - source: hf
    id: some/eval
    config: xx
    split: test
    weight: 1.0
    domain: general
candidates:
  - name: xx-bpe-32k
    algorithm: bpe
    vocab_size: 32000
    byte_fallback: true
    normalization: NFC
targets:
  max_fertility_vs_english: 1.0
  max_quality_drop_pct: 1.0
baselines:
  - gpt2
"""


def test_valid_synthetic_recipe_loads(tmp_path: Path):
    # Sanity check that the synthetic body used by the edge cases is valid.
    recipe = Recipe.load(_write(tmp_path, _VALID_BODY))
    assert recipe.language.code == "xx"


def test_unknown_field_rejected(tmp_path: Path):
    body = _VALID_BODY + "surprise: nope\n"
    path = _write(tmp_path, body)
    with pytest.raises(RecipeError) as excinfo:
        Recipe.load(path)
    assert "surprise" in str(excinfo.value)


def test_unknown_nested_field_rejected(tmp_path: Path):
    body = _VALID_BODY.replace(
        "  scripts: [Latn]\n",
        "  scripts: [Latn]\n  bogus: 1\n",
    )
    path = _write(tmp_path, body)
    with pytest.raises(RecipeError) as excinfo:
        Recipe.load(path)
    assert "bogus" in str(excinfo.value)


def test_missing_required_field_rejected(tmp_path: Path):
    body = _VALID_BODY.replace(
        "targets:\n"
        "  max_fertility_vs_english: 1.0\n"
        "  max_quality_drop_pct: 1.0\n",
        "",
    )
    path = _write(tmp_path, body)
    with pytest.raises(RecipeError) as excinfo:
        Recipe.load(path)
    assert "targets" in str(excinfo.value)


def test_wrong_type_rejected(tmp_path: Path):
    body = _VALID_BODY.replace("vocab_size: 32000", 'vocab_size: "big"')
    path = _write(tmp_path, body)
    with pytest.raises(RecipeError) as excinfo:
        Recipe.load(path)
    assert "vocab_size" in str(excinfo.value)


def test_bad_enum_rejected(tmp_path: Path):
    body = _VALID_BODY.replace("algorithm: bpe", "algorithm: rnn")
    path = _write(tmp_path, body)
    with pytest.raises(RecipeError) as excinfo:
        Recipe.load(path)
    assert "algorithm" in str(excinfo.value)


def test_empty_list_field_rejected(tmp_path: Path):
    body = _VALID_BODY.replace("baselines:\n  - gpt2\n", "baselines: []\n")
    path = _write(tmp_path, body)
    with pytest.raises(RecipeError):
        Recipe.load(path)


def test_missing_path_rejected(tmp_path: Path):
    missing = tmp_path / "does-not-exist.yaml"
    with pytest.raises(RecipeError) as excinfo:
        Recipe.load(missing)
    assert str(missing) in str(excinfo.value)


def test_non_mapping_yaml_rejected(tmp_path: Path):
    path = _write(tmp_path, "- just\n- a\n- list\n")
    with pytest.raises(RecipeError) as excinfo:
        Recipe.load(path)
    assert "mapping" in str(excinfo.value)


def test_scalar_yaml_rejected(tmp_path: Path):
    path = _write(tmp_path, "just a scalar string\n")
    with pytest.raises(RecipeError) as excinfo:
        Recipe.load(path)
    assert "mapping" in str(excinfo.value)


def test_invalid_yaml_rejected(tmp_path: Path):
    path = _write(tmp_path, "language: [unclosed\n")
    with pytest.raises(RecipeError) as excinfo:
        Recipe.load(path)
    assert str(path) in str(excinfo.value)


def test_non_utf8_file_rejected(tmp_path: Path):
    # A non-UTF-8 recipe must surface as RecipeError, not a raw
    # UnicodeDecodeError, so the loader's single-exception contract holds.
    path = tmp_path / "recipe.yaml"
    path.write_bytes(b"\xff\xfe not utf-8")
    with pytest.raises(RecipeError) as excinfo:
        Recipe.load(path)
    assert str(path) in str(excinfo.value)


def test_infinite_weight_rejected(tmp_path: Path):
    # The first corpus weight becomes infinite; gt=0 alone would let it pass.
    body = _VALID_BODY.replace("weight: 1.0", "weight: .inf", 1)
    path = _write(tmp_path, body)
    with pytest.raises(RecipeError) as excinfo:
        Recipe.load(path)
    assert "weight" in str(excinfo.value)


def test_nan_target_rejected(tmp_path: Path):
    body = _VALID_BODY.replace(
        "max_fertility_vs_english: 1.0", "max_fertility_vs_english: .nan"
    )
    path = _write(tmp_path, body)
    with pytest.raises(RecipeError) as excinfo:
        Recipe.load(path)
    assert "max_fertility_vs_english" in str(excinfo.value)
