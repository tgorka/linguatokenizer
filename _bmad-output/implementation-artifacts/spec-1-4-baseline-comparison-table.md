---
title: 'Baseline comparison table'
type: 'feature'
created: '2026-07-04'
status: 'blocked'
baseline_revision: a3ae107e1815d0929b010fcb8c4d44e90bd26ceb
review_loop_iteration: 0
followup_review_recommended: false
context: []
warnings: []
---

<intent-contract>

## Intent

**Problem:** `benchmark.fertility` can score any single tokenizer over
aligned FLORES PL/EN corpora (`aligned_fertility`, Story 1.3, done), but
nothing yet loads the recipe's real baseline tokenizers (gpt2, Llama,
Mistral, Qwen, Gemma, Bielik-v3) and turns their per-baseline scores into the
committed `results/baselines-pl-en.json` + markdown table that states the
Polish fertility penalty on real data.

**Approach:** Add `benchmark/compare.py` with `compare_baselines(recipe)`,
which resolves one FLORES PL/EN aligned pair via `corpora.resolve_source`,
loads each `recipe.baselines` entry as a Hugging Face tokenizer, scores it
with the existing `aligned_fertility`, and returns a `ComparisonReport`
serializable via `write_report` (JSON) and a new `write_comparison_markdown`
(table). Wire it behind the CLI's anticipated `benchmark` subcommand so
running it end-to-end produces the two committed artifacts.

## Boundaries & Constraints

**Always:**
- Reuse `aligned_fertility`/`fertility` for all scoring and
  `corpora.resolve_source` for FLORES document iterables — no duplicated
  token/word-counting or hf-loading logic in `compare.py`.
- Derive the FLORES English pair from the *same* `CorpusSpec` recipe entry
  used for the target language (`source: hf`, `id: facebook/flores`) by
  copying it with `config="eng_Latn"` — `recipe.yaml` keeps a single FLORES
  entry as the source of truth; no second eval-corpus entry is added.
- Each `recipe.baselines` entry is tried independently: a tokenizer that
  fails to load (network error, gated repo without accepted terms, missing
  token, bad id) is recorded as a skipped `BaselineOutcome` with a
  human-readable reason and does not abort scoring the remaining baselines.
  This intentionally differs from `corpora.sources._resolve_hf`'s "raise if
  `HF_TOKEN` is set but still fails" rule: one baseline failing must never
  corrupt or abort the whole comparison table the other baselines populate.
- Widen `fertility.write_report`'s `report` parameter from
  `FertilityReport | AlignedFertilityReport` to a new structural
  `Reportable` protocol (`as_dict() -> dict`) so `ComparisonReport` reuses it
  for JSON output — no second JSON-write implementation, no behavior change
  to existing callers.
- `transformers` is imported lazily inside the baseline-loading function,
  mirroring `corpora/sources.py`'s `import datasets` convention;
  `benchmark/fertility.py` stays dependency-free, only `compare.py` takes
  the new (optional) dependency.
- The CLI's `benchmark` subcommand (`--recipe`, default
  `languages/pl/recipe.yaml`; `--out-dir`, default `results`) makes one
  `compare_baselines` call and writes both
  `<out-dir>/baselines-<language.code>-en.json` and `...md` from its result.

**Block If:** none identified — the graceful-skip design above handles every
per-baseline failure mode without needing a human decision mid-run.

**Never:**
- Do not score `recipe.candidates` here — those aren't trained until Epic 2;
  `compare.py` only scores `recipe.baselines`.
- Do not add any workaround for gated-repo license acceptance (e.g. scraping
  the HF web UI) — an unaccepted gated baseline just skips.
- Do not change `write_report`'s JSON formatting/behavior (still `indent=2`,
  `ensure_ascii=False`, trailing newline, parent-dir creation) — only widen
  its type hint.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| All baselines load | 2 fake baseline tokenizers, both load fine, equal-length fake FLORES pair | `ComparisonReport.outcomes` has 2 `BaselineOutcome`s, each `report` set and `skip_reason=None` | No error |
| One baseline fails to load | 2 baselines, one fake loader raises | that outcome has `report=None`, `skip_reason` names the baseline and reason; the other baseline is still scored | No error (failure recorded, run continues) |
| Recipe has no FLORES eval corpus | `eval_corpora` has no `source=hf, id=facebook/flores` entry | -- | raises `CompareError` naming the missing source |
| FLORES pair length mismatch | fake target corpus has 3 docs, fake english corpus has 2 | -- | propagates `aligned_fertility`'s `ValueError` unchanged |
| Report round-trip | any `ComparisonReport` | `write_report(report, path)`; `json.loads(path.read_text()) == report.as_dict()` | No error |
| Markdown summary | `ComparisonReport` with one scored + one skipped outcome | `write_comparison_markdown` emits a table row per baseline; the skipped row shows its reason instead of numbers | No error |

</intent-contract>

## Code Map

- `packages/tokenizer-core/src/tokenizer_core/benchmark/fertility.py` --
  extend: add `Reportable` protocol, widen `write_report`'s `report` param
  type to `Reportable`.
- `packages/tokenizer-core/src/tokenizer_core/benchmark/compare.py` -- NEW:
  `CompareError`, `BaselineOutcome`, `ComparisonReport`,
  `_load_baseline_encoder`, `compare_baselines`, `write_comparison_markdown`.
- `packages/tokenizer-core/src/tokenizer_core/benchmark/__init__.py` -- add
  re-exports for the five new public names.
- `packages/tokenizer-core/src/tokenizer_core/cli.py` -- add a `benchmark`
  subcommand (`--recipe`, `--out-dir`, `--data-dir`) that loads the recipe,
  calls `compare_baselines`, writes both output files, and prints a
  one-line-per-baseline summary.
- `packages/tokenizer-core/tests/test_compare.py` -- NEW: unit tests for the
  I/O Matrix above, using fake `transformers`/`datasets` modules injected via
  `sys.modules` (mirroring `test_sources.py`'s pattern) -- no real network.
- `packages/tokenizer-core/tests/test_cli.py` -- NEW: argparse wiring smoke
  test for the `benchmark` subcommand (monkeypatched `compare_baselines`).
- `packages/tokenizer-core/src/tokenizer_core/corpora/sources.py`,
  `packages/tokenizer-core/src/tokenizer_core/train/config.py` -- READ-ONLY
  references for `resolve_source`'s `(docs, skip_reason)` contract and
  `Recipe`/`CorpusSpec` fields.
- `languages/pl/recipe.yaml` -- READ-ONLY; its existing `baselines` and
  `eval_corpora` (FLORES) entries drive the live verification run.

## Tasks & Acceptance

**Execution:**
- [x] `packages/tokenizer-core/src/tokenizer_core/benchmark/fertility.py` --
  add `Reportable(Protocol): def as_dict(self) -> dict`, change
  `write_report`'s signature to `write_report(report: Reportable, path: str |
  Path) -> Path` -- rationale: lets `ComparisonReport` reuse the existing
  JSON writer without duplicating it.
- [x] `packages/tokenizer-core/src/tokenizer_core/benchmark/compare.py` --
  add `CompareError(Exception)`; `BaselineOutcome` frozen dataclass (`name:
  str, report: AlignedFertilityReport | None, skip_reason: str | None`) +
  `as_dict()`; `ComparisonReport` frozen dataclass (`language_code: str,
  outcomes: tuple[BaselineOutcome, ...]`) + `as_dict()` -- rationale: AC1/AC2
  data model for the comparison table.
- [x] same file -- add `_load_baseline_encoder(name: str) -> tuple[Encoder |
  None, str | None]` (encoder, skip_reason), lazily importing `transformers`
  and calling `AutoTokenizer.from_pretrained(name, token=os.environ.get(
  "HF_TOKEN"))`, catching any exception into a skip reason -- rationale: the
  graceful, never-abort-the-run baseline-loading contract from Boundaries.
- [x] same file -- add `compare_baselines(recipe: Recipe, *, data_dir: str |
  Path | None = None) -> ComparisonReport`: find the `facebook/flores`
  `CorpusSpec` in `recipe.eval_corpora` (raise `CompareError` if absent),
  build its English twin (`config="eng_Latn"`), resolve both once via
  `corpora.resolve_source`, then for each `recipe.baselines` name load an
  encoder and either score it with `aligned_fertility` or record the skip --
  rationale: AC1, the end-to-end scoring entry point.
- [x] same file -- add `write_comparison_markdown(report: ComparisonReport,
  path: str | Path) -> Path`: a `# Baseline Comparison` heading + a table
  (Baseline | PL fertility | EN fertility | fertility_vs_english | Status),
  skipped rows showing their reason in Status -- rationale: AC2's markdown
  summary.
- [x] `packages/tokenizer-core/src/tokenizer_core/benchmark/__init__.py` --
  export `CompareError, BaselineOutcome, ComparisonReport,
  compare_baselines, write_comparison_markdown` -- rationale: documented
  import surface, matching `fertility`'s re-export pattern.
- [x] `packages/tokenizer-core/src/tokenizer_core/cli.py` -- add a
  `benchmark` subparser (`--recipe` default `languages/pl/recipe.yaml`,
  `--out-dir` default `results`, `--data-dir` default `None`) whose handler
  loads the recipe via `Recipe.load`, calls `compare_baselines`, writes
  `<out-dir>/baselines-<language.code>-en.json` via `write_report` and
  `...md` via `write_comparison_markdown`, and prints one summary line per
  baseline -- rationale: AC2, a runnable command producing the committed
  artifacts.
- [x] `packages/tokenizer-core/tests/test_compare.py` -- unit tests for every
  I/O Matrix row, using fake `transformers`/`datasets` modules in
  `sys.modules` and `tmp_path` for file-output assertions -- rationale:
  proves AC1/AC2's data flow without real HF network access.
- [x] `packages/tokenizer-core/tests/test_cli.py` -- test the `benchmark`
  subcommand parses its arguments and calls `compare_baselines` with a
  monkeypatched recipe/report -- rationale: proves the CLI wiring in
  isolation from the real scoring pipeline.

**Acceptance Criteria:**
- Given `languages/pl/recipe.yaml`'s baselines and FLORES eval corpus, when
  `compare_baselines` runs, then it returns a `ComparisonReport` with one
  `BaselineOutcome` per baseline name, each either scored via
  `aligned_fertility` or skipped with a reason.
- Given the live `benchmark` CLI run against `languages/pl/recipe.yaml` with
  `HF_TOKEN` resolved (`task secrets:run`), when it completes, then
  `results/baselines-pl-en.json` and `results/baselines-pl-en.md` exist and
  are committed, and the scored baselines' `fertility_vs_english` values are
  reported in the story's Verification section (reproducing the
  ~1.5–3.2 tokens/word Polish range where a given baseline scores).
- Given `uv run task check` (lint + format + types + tests) without the
  `corpora`/`adapters` extras installed, when run, then it passes --
  `compare.py`'s heavy imports (`transformers`, and transitively `datasets`
  via `corpora.resolve_source`) stay inside function bodies, never at module
  level.

## Design Notes

`compare_baselines` derives the English FLORES pair by copying the
recipe's `facebook/flores` `CorpusSpec` with `config="eng_Latn"` rather than
requiring a second `eval_corpora` entry — FLORES-200 uses one dataset id with
a per-language `config`, and the English reference is a project-wide
constant (per Story 1.3's design notes), not something a future
`languages/<iso>/recipe.yaml` should have to restate.

The skip-on-any-failure baseline-loading rule deliberately diverges from
`corpora.sources._resolve_hf`'s "raise if `HF_TOKEN` is set but still fails"
convention. That convention protects training-corpus integrity (a silently
missing corpus warps the weighted mix); a baseline tokenizer is just one row
in a reference table, so the safer default here is "always skip and keep
going, never abort the batch."

## Verification

**Commands:**
- `uv run pytest packages/tokenizer-core/tests/test_compare.py
  packages/tokenizer-core/tests/test_cli.py -q` -- expected: all new tests
  pass.
- `uv run task check` -- expected: lint + format + types + full suite pass
  with no `corpora`/`adapters` extras installed.
- `task setup:all && task secrets:run -- uv run tokenizer benchmark --recipe
  languages/pl/recipe.yaml` -- expected: `results/baselines-pl-en.json` +
  `.md` written; record which baselines scored vs. skipped and their
  `fertility_vs_english` values in this section, then `git add` the two
  result files as part of this story's commit.

**Manual checks (if no CLI):**
- Inspect `results/baselines-pl-en.md` for a readable table before
  committing it.

**Results (2026-07-04):**
- `uv run pytest packages/tokenizer-core/tests/test_compare.py
  packages/tokenizer-core/tests/test_cli.py -q` -- 12 passed (7 in
  `test_compare.py`, 5 in `test_cli.py`), covering every I/O Matrix row: all
  baselines load, one baseline fails to load (the other still scores), a
  recipe with no `facebook/flores` eval entry raises `CompareError`, a
  FLORES pair length mismatch propagates `aligned_fertility`'s `ValueError`
  unchanged, `write_report` round-trips a `ComparisonReport`, and
  `write_comparison_markdown` renders both a scored row and a skipped row
  (reason shown in Status).
- `uv run task check` is fully green in the base environment (`uv sync
  --all-packages`, no `corpora`/`train`/`adapters` extras): ruff lint ("All
  checks passed!"), ruff format ("18 files already formatted"), `ty check`
  ("All checks passed!"), and `pytest` (72 passed across the whole
  workspace, up from 60 pre-story). Confirmed `transformers`/`datasets`/
  `tokenizers`/`sentencepiece`/`tiktoken` are absent from the environment
  during this run (`uv pip list` shows none of them); `compare.py`'s only
  module-level imports are stdlib + `tokenizer_core.benchmark.fertility` +
  `tokenizer_core.corpora.sources` + `tokenizer_core.train.config` --
  `transformers` is imported solely inside `_load_baseline_encoder`'s
  function body.
- **Live run:** `task setup:all` (installed the `train`/`corpora`/`adapters`
  extras, incl. `transformers==5.13.0`, `datasets==5.0.0`) then
  `task secrets:run -- uv run tokenizer benchmark --recipe
  languages/pl/recipe.yaml`. `task secrets:check` confirmed `HF_TOKEN`
  resolved from the `neuraffica` 1Password vault, and `huggingface_hub.
  whoami(token=...)` confirmed the token is valid (`fineGrained.
  canReadGatedRepos: True`) for account `tgorka`.
  - The run did **not** reach baseline scoring: `resolve_source` on the
    recipe's `facebook/flores` eval corpus raised `SourceError` (`"failed to
    load hf source 'facebook/flores'"`), because `datasets.load_dataset`
    itself raised `DatasetNotFoundError`: `"Dataset 'facebook/flores' is a
    gated dataset on the Hub. Visit the dataset page at
    https://huggingface.co/datasets/facebook/flores to ask for access."`
    Direct API inspection (`GET /api/datasets/facebook/flores`) confirms
    `"gated": "auto"` on the repo; the `tgorka` HF account has not (yet)
    clicked through Facebook's gated-dataset agreement on the dataset's web
    page, so every request for the actual data (not just its metadata --
    `dataset_info()` succeeds, `load_dataset()` doesn't) 403s. This is
    `resolve_source`/`_resolve_hf`'s pre-existing (Story 1.2) "raise if
    `HF_TOKEN` is set but the load still fails" contract, not a bug
    introduced by this story, and per the spec's Boundaries no workaround
    (e.g. scraping the HF web UI to accept terms) was attempted.
  - Because the run never got past resolving the FLORES pair,
    `compare_baselines`'s own per-baseline graceful-skip logic was never
    exercised live end-to-end; `results/baselines-pl-en.json`/`.md` were
    **not** produced, so nothing was left in the working tree for this part.
  - To confirm the rest of the pipeline is sound independent of the FLORES
    blocker, each recipe baseline was loaded directly via
    `AutoTokenizer.from_pretrained(name, token=HF_TOKEN)` with the same
    token: `gpt2`, `meta-llama/Llama-3.1-8B`, `mistralai/Mistral-7B-v0.1`,
    and `Qwen/Qwen2.5-7B` all loaded successfully; `google/gemma-2-9b`
    failed with `OSError: You are trying to access a gated repo` (403 --
    gated, terms not accepted for this account); `speakleash/Bielik-11B-v3`
    failed with `OSError: ... is not a valid model identifier` (the repo id
    in the recipe does not resolve on the Hub as given). Both failure modes
    are exactly the shapes `_load_baseline_encoder` is designed to catch and
    turn into a skipped `BaselineOutcome` rather than raise -- this manual
    check exercises that the *reasons* `compare_baselines` would report are
    real and would read sensibly, even though the live CLI run itself
    couldn't reach that code path.
  - No `fertility_vs_english` values were obtained from a live run; none are
    fabricated here. Unblocking this requires a human accepting the
    `facebook/flores` gated-dataset terms on
    https://huggingface.co/datasets/facebook/flores for the `tgorka` account
    (and separately requesting `google/gemma-2-9b` access / confirming the
    correct `speakleash/Bielik-11B-v3` repo id if those baselines are to
    score too), then re-running `task secrets:run -- uv run tokenizer
    benchmark --recipe languages/pl/recipe.yaml`.
  - The environment was returned to the base, no-extras state (`uv sync
    --all-packages`) after this live-run attempt, and `task check` was
    re-verified green in that state (see above) -- the `train`/`corpora`/
    `adapters` extras installed for this attempt are not left in the
    committed environment state.

## Auto Run Result

Status: blocked

**Blocking condition:** implementation verification failed -- the second
Acceptance Criterion ("the live `benchmark` CLI run against
`languages/pl/recipe.yaml` ... `results/baselines-pl-en.json` and
`results/baselines-pl-en.md` exist and are committed") is not satisfied.
`facebook/flores` is a gated dataset on the Hugging Face Hub (`"gated":
"auto"`); the `tgorka` account behind the vault's `HF_TOKEN` has a valid,
correctly-scoped token but has not clicked through the dataset's license
agreement at https://huggingface.co/datasets/facebook/flores, so
`datasets.load_dataset("facebook/flores", ...)` 403s and `resolve_source`
raises `SourceError` before any baseline is even attempted. Per this
story's own Boundaries ("Never: do not add any workaround for gated-repo
license acceptance"), no code-side fix exists for this -- it requires a
human to accept the dataset's terms on the HF website for that account.

All code, tests, and the `task check` gate are complete and green (see
Tasks & Acceptance and Verification above); only the live-artifact
Acceptance Criterion is blocked. Separately, `google/gemma-2-9b` is also
gated (403, terms not accepted) and `speakleash/Bielik-11B-v3` does not
resolve as a valid model id on the Hub -- these two will report as skipped
`BaselineOutcome`s (by design) once the FLORES blocker is cleared, rather
than blocking the run further.

**To unblock:** a human should (1) visit
https://huggingface.co/datasets/facebook/flores while signed in as the
`tgorka` account (or whichever account owns the vault's `HF_TOKEN`) and
accept its gated-dataset terms; (2) optionally do the same for
`google/gemma-2-9b` and confirm/correct the `speakleash/Bielik-11B-v3` repo
id in `languages/pl/recipe.yaml` if that baseline should score too; then (3)
re-run `task setup:all && task secrets:run -- uv run tokenizer benchmark
--recipe languages/pl/recipe.yaml` and commit the resulting
`results/baselines-pl-en.json`/`.md` to close out this story.
