---
title: 'Recipe schema and loader'
type: 'feature'
created: '2026-07-04'
status: 'done'
baseline_revision: 75adec7ff1bed4072e590580e4f78305cf1305b4
final_revision: 07ea7d9d5927754af490be80f55e769dfc5b052a
review_loop_iteration: 0
followup_review_recommended: false
context: []
warnings: [oversized]
---

<intent-contract>

## Intent

**Problem:** A language's `recipe.yaml` (corpora, candidates, targets,
baselines) is the repo's source of truth, but nothing validates it — a typo or
wrong type is discovered only deep in acquisition/training. Stories 1.2 and 1.4
need typed, validated recipe objects to build on.

**Approach:** Add Pydantic v2 models for the recipe structure in
`tokenizer_core.train.config`, with a `Recipe.load(path)` classmethod that
reads YAML and validates it, rejecting unknown fields and failing fast with a
clear, path-annotated error.

## Boundaries & Constraints

**Always:** Pydantic v2 models with `extra="forbid"` on every model so unknown
fields are rejected. `Recipe.load(path)` accepts a `str | os.PathLike` and
returns a validated `Recipe`. Only base deps (`pydantic`, `pyyaml`) — both
already declared; no heavy extras. Google-style docstrings, `from __future__
import annotations`, double quotes, 80-col (matches `benchmark/fertility.py`).
Models must parse both `languages/pl/recipe.yaml` and
`languages/_template/recipe.yaml` unchanged.

**Block If:** The committed `languages/pl/recipe.yaml` or
`languages/_template/recipe.yaml` cannot be represented by a reasonable schema
without editing those files (would signal a schema/data mismatch needing a
human decision). Do not silently edit the recipes to fit the schema.

**Never:** Do not import `tokenizers`/`datasets`/`transformers`/`numpy`. Do not
implement corpus acquisition, sampling, or training (Stories 1.2+). Do not add
a CLI command. Do not commit trained binaries.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Valid Polish recipe | `languages/pl/recipe.yaml` | Returns `Recipe`; fields typed (language, corpora, candidates, targets, baselines) | No error expected |
| Valid template recipe | `languages/_template/recipe.yaml` | Returns `Recipe` with `xx` placeholders parsed | No error expected |
| Unknown field | recipe with an extra top-level or nested key | Raises `RecipeError` naming the unexpected field | Rejected via `extra="forbid"` |
| Missing required field | recipe with `targets` omitted | Raises `RecipeError` | Path + field named |
| Wrong type / bad enum | `vocab_size: "big"` or `algorithm: rnn` | Raises `RecipeError` | Path + field named |
| Missing / non-file path | path does not exist | Raises `RecipeError` naming the path | No traceback noise |
| Non-mapping YAML | file whose top level is a list or scalar | Raises `RecipeError` | Clear message |

</intent-contract>

## Code Map

- `packages/tokenizer-core/src/tokenizer_core/train/config.py` -- NEW: Pydantic models (`Language`, `CorpusSpec`, `Candidate`, `Targets`, `Recipe`), `RecipeError`, and `Recipe.load`.
- `packages/tokenizer-core/src/tokenizer_core/train/__init__.py` -- currently a placeholder docstring; re-export the public config symbols.
- `packages/tokenizer-core/tests/test_recipe_config.py` -- NEW: unit tests (real recipes + edge cases).
- `languages/pl/recipe.yaml`, `languages/_template/recipe.yaml` -- READ-ONLY fixtures the schema must match (structure: `language`, `train_corpora`, `eval_corpora`, `candidates`, `targets`, `baselines`).
- `packages/tokenizer-core/src/tokenizer_core/benchmark/fertility.py` -- style/`NormalizationForm` reference to mirror.

## Tasks & Acceptance

**Execution:**
- [x] `packages/tokenizer-core/src/tokenizer_core/train/config.py` -- Add Pydantic v2 models. `Language{code,name,scripts:list[str]}`; `CorpusSpec{source:Literal["hf","url","local"], id:str, config:str|None=None, split:str|None=None, weight:float(gt=0)=1.0, domain:Literal["general","technical","web","wiki","legal","conversational"]="general"}`; `Candidate{name:str, algorithm:Literal["bpe","unigram","superbpe"], vocab_size:int(gt=0), byte_fallback:bool=True, normalization:Literal["NFC","NFD","NFKC","NFKD"]="NFC"}`; `Targets{max_fertility_vs_english:float(gt=0), max_quality_drop_pct:float(ge=0)}`; `Recipe{language, train_corpora:list[CorpusSpec], eval_corpora:list[CorpusSpec], candidates:list[Candidate], targets, baselines:list[str]}` with the four list fields non-empty (`min_length=1`). Every model sets `model_config = ConfigDict(extra="forbid")`. Add `RecipeError(ValueError)` and `@classmethod load(cls, path) -> Recipe` -- rationale: typed, validated source of truth for all downstream stories.
- [x] `packages/tokenizer-core/src/tokenizer_core/train/__init__.py` -- Replace placeholder with re-exports (`Recipe`, `RecipeError`, `Language`, `CorpusSpec`, `Candidate`, `Targets`) and `__all__` -- rationale: `tokenizer_core.train` is the documented import surface.
- [x] `packages/tokenizer-core/tests/test_recipe_config.py` -- Cover every I/O Matrix row: load both committed recipes and assert key parsed values; assert `RecipeError` for unknown field, missing required field, bad type/enum, missing path, and non-mapping YAML (use `tmp_path` for the invalid cases). Reference the real recipes via a repo-root-relative path resolved from `__file__` -- rationale: proves AC1/AC2 against real data with no network.

**Acceptance Criteria:**
- Given the committed `languages/pl/recipe.yaml` and `languages/_template/recipe.yaml`, when `Recipe.load(path)` is called on each, then it returns a validated `Recipe` and neither file is modified.
- Given a recipe containing an unknown field, when it is loaded, then a `RecipeError` is raised that identifies the offending field (no partial/loose parse).
- Given `tokenizer_core.train`, when a developer imports it, then `Recipe`, `RecipeError`, and the component models are importable, and `Recipe.load` is a classmethod taking a path.
- Given `task check` (ruff lint + format + ty + pytest), when run, then it passes with the new module and tests.

## Spec Change Log

## Review Triage Log

### 2026-07-04 — Review pass
- intent_gap: 0
- bad_spec: 0
- patch: 2: (high 0, medium 0, low 2)
- defer: 5: (high 0, medium 2, low 3)
- reject: 10: (high 0, medium 0, low 10)
- addressed_findings:
  - `[low]` `[patch]` `Recipe.load` let a raw `UnicodeDecodeError`/`OSError` from `read_text` escape past its documented single-`RecipeError` contract — wrapped the file read in `except (OSError, UnicodeDecodeError)` re-raising `RecipeError` (path + chained cause), updated the `Raises:` docstring, added a non-UTF-8-file test.
  - `[low]` `[patch]` `inf`/`nan` floats slipped past `gt=0`/`ge=0` in a loader whose whole purpose is fail-fast — added `allow_inf_nan=False` to `CorpusSpec.weight`, `Targets.max_fertility_vs_english`, and `Targets.max_quality_drop_pct`, plus infinite-weight and NaN-target tests.

## Design Notes

`RecipeError` wraps three failure classes with the path for context: missing
file, non-mapping / unparseable YAML, and Pydantic `ValidationError`. Chain the
original (`raise RecipeError(...) from err`) so the underlying detail survives.

```python
@classmethod
def load(cls, path: str | os.PathLike[str]) -> Recipe:
    p = Path(path)
    if not p.is_file():
        raise RecipeError(f"recipe not found: {p}")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as err:
        raise RecipeError(f"recipe {p} is not valid YAML: {err}") from err
    if not isinstance(data, dict):
        raise RecipeError(f"recipe {p} must be a YAML mapping")
    try:
        return cls.model_validate(data)
    except ValidationError as err:
        raise RecipeError(f"invalid recipe {p}:\n{err}") from err
```

`CorpusSpec.config`/`split` are optional so `url`/`local` sources (which need
neither) validate; the committed recipes always set them.

## Verification

**Commands:**
- `uv run pytest packages/tokenizer-core/tests/test_recipe_config.py -q` -- expected: all tests pass.
- `uv run ruff check . && uv run ruff format --check .` -- expected: clean.
- `uv run ty check` -- expected: no errors.
- `task check` -- expected: full local gate green.

## Auto Run Result

Status: done

**Summary:** Added the typed recipe layer for the tokenizer library. New
Pydantic v2 models (`Language`, `CorpusSpec`, `Candidate`, `Targets`, `Recipe`)
in `tokenizer_core.train.config` with `extra="forbid"` on every model, plus
`Recipe.load(path)` which reads a YAML recipe and raises a single
path-annotated `RecipeError` for missing files, unreadable/non-UTF-8 files,
invalid YAML, non-mapping YAML, and schema-validation failures. Parses both the
committed `languages/pl/recipe.yaml` and `languages/_template/recipe.yaml`
unchanged. Base deps only (`pydantic`, `pyyaml`); no heavy extras imported.

**Files changed:**
- `packages/tokenizer-core/src/tokenizer_core/train/config.py` (new) — recipe models, `RecipeError`, `Recipe.load`.
- `packages/tokenizer-core/src/tokenizer_core/train/__init__.py` (modified) — re-export the public config symbols with `__all__`.
- `packages/tokenizer-core/tests/test_recipe_config.py` (new) — 17 unit tests: both real recipes + every I/O-matrix edge case + the two review patches.
- `_bmad-output/implementation-artifacts/epic-1-context.md` (new) — compiled Epic 1 planning context.
- `_bmad-output/implementation-artifacts/spec-1-1-recipe-schema-and-loader.md` (new) — this spec.
- `_bmad-output/implementation-artifacts/deferred-work.md` (new) — 5 deferred review findings.

**Review findings breakdown:** 2 patches applied (both low; UTF-8/OSError read-error wrapping, and `allow_inf_nan=False` on `weight`/targets), 5 deferred (train/eval disjointness, name/baseline uniqueness, duplicate-YAML-key collapse, empty-string/ISO field shape, hf-source `config`/`split` requiredness), 10 rejected as noise. 0 intent_gap, 0 bad_spec — no repair loopback.

**Verification:** `task check` green — ruff lint + `ruff format --check` clean, `ty check` clean, `pytest` 25 passed (17 in the new test module). Committed on branch `romantic-dolphin`; not pushed.

**Residual risks:** The schema validates structure/types but not the deferred cross-field semantic invariants (notably train/eval leak and duplicate-YAML-key collapse); recipe *ids* are not verified to resolve until Story 1.2 acquisition. `followup_review_recommended: false` — the only changes this pass were two localized low-severity patches.
