---
title: 'Fertility & compression metrics'
type: 'feature'
created: '2026-07-04'
status: 'done'
baseline_revision: 4c1bb369492b817c0543bb4813c430d74333ce9f
final_revision: 951860efdf39dab9c7f68c80345138e48b8b5fbf
review_loop_iteration: 0
followup_review_recommended: false
context: []
warnings: []
---

<intent-contract>

## Intent

**Problem:** `benchmark.fertility` already scores tokens/word for any
tokenizer over a single corpus (AC1, done), but there is no way to state the
"target language costs N× English" claim on sentence-aligned content (the
North Star comparison), and no way to persist a report as the committed JSON
artifact under `results/` that Story 1.4's baseline table will build on.

**Approach:** Add an `AlignedFertilityReport` + `aligned_fertility()` helper
that scores two positionally-aligned corpora (target language, English) with
one encoder and computes the fertility ratio, plus a `write_report()` helper
that serializes any report's `as_dict()` to a JSON file, mirroring
`corpora/sampling.py`'s manifest-write pattern.

## Boundaries & Constraints

**Always:**
- `aligned_fertility` reuses the existing `fertility()` function for each
  side — no duplicated token/word/char counting logic.
- The ratio field is named `fertility_vs_english`, mirroring
  `recipe.yaml`'s existing `targets.max_fertility_vs_english` field so the
  same term is used end-to-end. `fertility_vs_english = target.fertility /
  english.fertility`; `0.0` when `english.fertility` is `0` (same
  zero-denominator convention as `FertilityReport`).
- Aligned corpora must be the same length (sentence-aligned pairs, index-for-
  index) before any encoding happens — raise `ValueError` naming both lengths
  on mismatch, since a silent misalignment would corrupt the "identical
  content" comparison the whole metric depends on.
- `write_report` creates parent directories, writes pretty (`indent=2`)
  UTF-8 JSON with a trailing newline — the same style as
  `corpora/sampling.py`'s `json.dumps(..., indent=2, ensure_ascii=False) +
  "\n"`.
- New public names are re-exported from `tokenizer_core.benchmark`'s
  `__init__.py` alongside the existing ones.

**Never:**
- Do not implement FLORES dataset loading/acquisition here — that's
  `corpora/sources.py`'s job (Story 1.2, done). This story only adds metrics
  primitives that accept already-resolved document iterables.
- Do not implement the baseline-vs-candidate table (`benchmark/compare.py`,
  Story 1.4) — this story stops at the reusable aligned-ratio + serialization
  primitives it will consume.
- Do not add `tokenizers`/`datasets`/`transformers`/`numpy` imports;
  `fertility.py` stays dependency-free.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Equal-length aligned corpora | `target=["a","b"]`, `english=["c","d"]`, whitespace encoder | `AlignedFertilityReport` with `.target`/`.english` `FertilityReport`s and `fertility_vs_english == target.fertility / english.fertility` | No error |
| Mismatched-length corpora | `target` has 3 docs, `english` has 2 | -- | raises `ValueError` naming both counts, before encoding either corpus |
| Empty English corpus | `english=[]` | `fertility_vs_english == 0.0` | No error (mirrors `FertilityReport`'s zero-denominator convention) |
| `write_report` to a nested, not-yet-existing directory | `path=results/tmp/x/report.json` | parent dirs created, file written | No error |
| `write_report` round-trip | any `FertilityReport` or `AlignedFertilityReport` | `json.loads(path.read_text()) == report.as_dict()` | No error |

</intent-contract>

## Code Map

- `packages/tokenizer-core/src/tokenizer_core/benchmark/fertility.py` --
  extend: add `AlignedFertilityReport`, `aligned_fertility()`,
  `write_report()`.
- `packages/tokenizer-core/src/tokenizer_core/benchmark/__init__.py` -- add
  re-exports for the three new names, mirroring `corpora/__init__.py`'s
  pattern.
- `packages/tokenizer-core/tests/test_fertility.py` -- extend with the I/O
  Matrix rows above.
- `packages/tokenizer-core/src/tokenizer_core/corpora/sampling.py` --
  READ-ONLY reference for the JSON-write style (`json.dumps(...,
  indent=2, ensure_ascii=False) + "\n"`) to mirror in `write_report`.

## Tasks & Acceptance

**Execution:**
- [x] `packages/tokenizer-core/src/tokenizer_core/benchmark/fertility.py` --
  add `AlignedFertilityReport` frozen dataclass (`target: FertilityReport`,
  `english: FertilityReport`, `fertility_vs_english: float`) + `as_dict()`
  (nested dicts + the ratio) -- rationale: JSON-serializable pairing for the
  "target costs N× English" claim.
- [x] same file -- add `aligned_fertility(encoder, target_corpus,
  english_corpus, *, normalize="NFC") -> AlignedFertilityReport`, validating
  equal lengths (via `list(...)` materialization) before scoring either side
  -- rationale: AC2, a single strict-alignment entry point Story 1.4's
  `compare.py` builds the baseline table on.
- [x] same file -- add `write_report(report: FertilityReport |
  AlignedFertilityReport, path: str | Path) -> Path` serializing
  `report.as_dict()` as described in Boundaries -- rationale: AC3, gives
  every report a path into `results/`.
- [x] `packages/tokenizer-core/src/tokenizer_core/benchmark/__init__.py` --
  export `AlignedFertilityReport`, `aligned_fertility`, `write_report` --
  rationale: documented import surface.
- [x] `packages/tokenizer-core/tests/test_fertility.py` -- unit tests for
  every I/O Matrix row above, using `tmp_path` for `write_report` -- rationale:
  proves AC2/AC3 without real FLORES data or the `corpora`/`train` extras.

**Acceptance Criteria:**
- Given a tokenizer and two sentence-aligned corpora of equal length, when
  `aligned_fertility` runs, then it returns an `AlignedFertilityReport` whose
  `fertility_vs_english` equals `target.fertility / english.fertility` (or
  `0.0` if `english.fertility` is `0`).
- Given corpora of differing lengths, when `aligned_fertility` runs, then it
  raises `ValueError` naming both lengths before encoding either corpus.
- Given any report and a destination path (including nested non-existent
  directories), when `write_report` runs, then the file exists afterward and
  `json.loads(path.read_text()) == report.as_dict()`.
- Given `task check` (ruff lint + format + ty + pytest), when run without the
  `corpora`/`train` extras installed, then it passes (`fertility.py` stays
  dependency-free).

## Spec Change Log

## Review Triage Log

### 2026-07-04 — Review pass
- intent_gap: 0
- bad_spec: 0
- patch: 5: (high 0, medium 1, low 4)
- defer: 2: (high 0, medium 1, low 1)
- reject: 9: (high 0, medium 0, low 9)
- addressed_findings:
  - `[medium]` `[patch]` `AlignedFertilityReport.fertility_vs_english`'s docstring and `aligned_fertility`'s `Raises` section overclaimed: the `0.0` sentinel is ambiguous between "english is empty" and "target is empty", and "misalignment" was described as fully caught when only length mismatches are — clarified both docstrings to state the ambiguity and narrow the claim to length checking.
  - `[low]` `[patch]` `write_report`'s local variable/docstring said "resolved" `Path`, implying canonicalization (`Path.resolve()`) that never happens — renamed to `dest_path` and reworded the docstring.
  - `[low]` `[patch]` No test exercised `aligned_fertility`'s `normalize` parameter — added `test_aligned_fertility_passes_normalize_through_to_both_sides` (NFD decomposition of `"ą"`).
  - `[low]` `[patch]` Existing `aligned_fertility` tests used same-shape fixtures on both sides, which couldn't catch a target/english argument swap — added `test_aligned_fertility_does_not_swap_target_and_english` with asymmetric per-side fertility.
  - `[low]` `[patch]` `write_report` was only ever tested with `Path` objects despite accepting `str | Path` — added `test_write_report_accepts_str_path`.
  - deferred to `deferred-work.md`: `aligned_fertility` fully materializes both corpora into lists, departing from `fertility()`'s streaming design (medium; fine at FLORES eval-set scale, revisit if aligned corpora grow); `write_report` has no typed guard for filesystem edge cases (path-is-a-directory, non-directory parent) and is not atomic against concurrent writers (low; same class already deferred for `corpora/sampling.py`'s manifest write, but `benchmark` has no typed exception yet to standardize on).
  - rejected as noise: "no caller uses the new functions yet" (Story 1.4 is the explicit, spec-scoped-out consumer); "`write_report` silently overwrites existing files" (intentional -- `results/` is git-committed for provenance, matching `sampling.py`'s existing unconditional-overwrite manifest write); "no test asserts overwrite behavior" (moot given the above); "length check doesn't imply content alignment" (the spec's own Boundaries scope the check to length only; true content alignment is a corpus-source guarantee, not this helper's job); "encoder invoked twice as two full passes" (the spec's own Boundaries mandate reusing `fertility()` per side rather than a new merged pass); "same iterator passed to both parameters drains silently" (verified false -- the existing length check already raises `ValueError` in this case, since the second `list(...)` call reads 0 items from the exhausted iterator); "`write_report` should type-check `report` before calling `.as_dict()`" (internal call, enforced by `ty` static typing, not a system boundary); "non-`str` corpus items raise unclear errors deep in `count_words`" (pre-existing behavior of the already-reviewed `fertility()`, not introduced by this story); "public API growth needs a changelog" (repo has no changelog convention; docstrings are the documentation); "a raised exception mid-materialization/mid-encoding loses partial results" (expected Python failure semantics, not a contract this module promises beyond what `fertility()` itself already provides).

## Design Notes

`fertility_vs_english` intentionally mirrors `recipe.yaml`'s
`targets.max_fertility_vs_english` field name rather than a Polish-specific
name (`fertility_ratio`, `pl_vs_en`), since the metric must work for any
future `languages/<iso>/` target compared against the fixed English
reference, not just Polish.

## Verification

**Commands:**
- `uv run pytest packages/tokenizer-core/tests/test_fertility.py -q` --
  expected: all tests pass.
- `uv run task check` -- expected: lint + format + types + full suite pass
  with no `corpora`/`train` extras installed.

**Results (2026-07-04):**
- `uv run task check` is fully green: ruff lint ("All checks passed!"), ruff
  format ("15 files already formatted"), `ty check` ("All checks passed!"),
  and `pytest` (57 passed, including 15 tests in `test_fertility.py`, up from
  7 pre-existing).
- `fertility.py`'s only imports are stdlib (`json`, `re`, `unicodedata`,
  `collections.abc`, `dataclasses`, `pathlib`, `typing`) -- confirmed
  dependency-free, satisfying AC4 without the `corpora`/`train` extras
  installed.
- AC1 (equal-length ratio math): `test_aligned_fertility_equal_length_computes_ratio`
  asserts `report.target`/`report.english` match direct `fertility()` calls
  and `fertility_vs_english == target.fertility / english.fertility`.
- AC2 (mismatched-length raises before encoding): `test_aligned_fertility_mismatched_length_raises_naming_both_counts`
  asserts the `ValueError` message names both counts (`target=3`,
  `english=2`); `test_aligned_fertility_does_not_encode_before_length_check`
  uses an encoder that raises `AssertionError` if ever called, proving no
  encoding happens before the length check.
- AC3 (write_report + round-trip): `test_write_report_creates_nested_directories`
  covers non-existent nested dirs; `test_write_report_roundtrips_fertility_report`
  and `test_write_report_roundtrips_aligned_fertility_report` assert
  `json.loads(path.read_text()) == report.as_dict()` for both report types.
- AC4 (task check green, dependency-free): see `task check` output above.

## Auto Run Result

Status: done

**Summary:** Extended `benchmark.fertility` (already scoring single-corpus
tokens/word) with the two remaining AC2/AC3 primitives: `aligned_fertility()`
for the sentence-aligned target-language-vs-English fertility ratio, and
`write_report()` to serialize any report as JSON for `results/`. All work
stayed inside `fertility.py`/its exports/its tests, per the spec's `Never`
boundaries (no FLORES acquisition, no `compare.py`).

**Files changed:**
- `packages/tokenizer-core/src/tokenizer_core/benchmark/fertility.py` --
  added `AlignedFertilityReport`, `aligned_fertility()`, `write_report()`.
- `packages/tokenizer-core/src/tokenizer_core/benchmark/__init__.py` --
  re-exported the three new names.
- `packages/tokenizer-core/tests/test_fertility.py` -- 15 tests total (7
  pre-existing + 8 new), covering every I/O Matrix row plus review-driven
  additions (normalize passthrough, swap-detection, str-path input).
- `_bmad-output/implementation-artifacts/deferred-work.md` -- 2 new entries.

**Review findings breakdown:** 5 patched (1 medium docstring-accuracy fix,
4 low naming/test-coverage fixes), 2 deferred (corpus materialization at
scale; `write_report` filesystem-edge-case guards), 9 rejected as noise
(scope/consistency mismatches -- see Review Triage Log for detail). No
intent_gap, no bad_spec.

**Follow-up review recommendation:** false -- the patched findings were
docstring-accuracy and test-coverage additions with no behavior, API, or
data-impact change; volume (5) and severity (mostly low) don't warrant an
independent follow-up pass.

**Verification performed:** `uv run task check` green after patches (ruff
lint + format, `ty check`, pytest 60 passed across the whole workspace,
`packages/tokenizer-core/tests/test_fertility.py` 15 passed). All 4
acceptance criteria confirmed against the final code (see Verification
section above).

**Residual risks:** the two deferred items (corpus materialization
scalability, `write_report` filesystem-edge-case hardening) are tracked in
`deferred-work.md` and are non-blocking given current corpus/usage scale.
