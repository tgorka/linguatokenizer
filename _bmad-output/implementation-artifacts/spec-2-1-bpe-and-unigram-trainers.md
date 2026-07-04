---
title: 'BPE and Unigram trainers'
type: 'feature'
created: '2026-07-04'
status: 'blocked'
baseline_revision: 4c36fb09a46fa2a71c151f90833c40cdcd8e9c97
review_loop_iteration: 0
followup_review_recommended: false
context: []
warnings: [oversized]
---

<intent-contract>

## Intent

**Problem:** `train/config.py` (Story 1.1) validates a recipe's `candidates`
(`bpe`/`unigram`/`superbpe`, vocab size, byte fallback, normalization), and
`corpora` (Story 1.2) can resolve/sample `train_corpora`, but nothing yet
actually trains a tokenizer from them -- there is no `train/trainers.py` or
`train/artifact.py`, so no candidate can be produced, scored, or handed to
Epic 3's adapters.

**Approach:** Add `train/trainers.py` with `train_candidate(recipe,
candidate, ...)`, which gathers a seeded/weighted sample of `recipe
.train_corpora` and trains a Hugging Face `tokenizers` BPE or Unigram model
per the candidate's `vocab_size`/`byte_fallback`/`normalization`. Add
`train/artifact.py` to save/load the resulting tokenizer as `tokenizer.json`
plus a `metadata.json` sidecar (algorithm, vocab size, corpus hash, seed).
Wire a new CLI `train` subcommand so training is one command per language.
SuperBPE (Story 2.2) is out of scope; unsupported candidates are skipped, not
fatal.

## Boundaries & Constraints

**Always:**
- `train_candidate` only handles `algorithm in ("bpe", "unigram")`; any other
  algorithm (`superbpe`) raises `TrainerError` naming the candidate and
  algorithm -- the CLI catches this per-candidate and skips it with a
  printed message, continuing the remaining candidates.
- Training documents come from a seeded, weighted sample of
  `recipe.train_corpora`, reusing `corpora.sample_corpora`'s existing
  reservoir-sampling/weight-normalization logic -- no duplicated
  corpus-mixing logic in `trainers.py`. `sample_corpora`'s return type widens
  from `CorpusManifest` to `tuple[CorpusManifest, list[str]]` (manifest +
  the actual sampled documents, concatenated in source order) so callers get
  exactly the text that was hashed; all existing `test_sampling.py` call
  sites are updated to unpack the tuple.
- `candidate.normalization` maps to `tokenizers.normalizers.{NFC,NFD,NFKC,
  NFKD}`, set on the tokenizer before training.
- A `bpe` candidate with `byte_fallback: true` uses the GPT-2-style
  byte-level pipeline (`pre_tokenizers.ByteLevel` + its full byte alphabet as
  `initial_alphabet` + `decoders.ByteLevel`) so every byte is representable
  without an `<unk>` token. A `unigram` candidate passes `byte_fallback`
  straight through to `tokenizers.trainers.UnigramTrainer`'s native
  `byte_fallback` parameter.
- The `metadata.json` sidecar records at least: candidate `name`,
  `algorithm`, `vocab_size`, `byte_fallback`, `normalization`,
  `corpus_hash` (from the widened `sample_corpora` manifest), `seed`, and
  `language_code`.
- Trained artifacts (`tokenizer.json` + `metadata.json`) are written under
  the data plane (`{data_dir}/tokenizers/{language_code}/{candidate.name}/`,
  `data_dir` defaulting to `$TOKENIZER_DATA_DIR`) -- never under `results/`
  and never committed to git; only the recipe, source, and Taskfile entry
  are committed (per `CLAUDE.md`: don't commit trained tokenizer binaries).
- `train/trainers.py` and `train/artifact.py` import `tokenizers` lazily
  inside function bodies (`# noqa: PLC0415`), matching
  `corpora/sources.py`'s guard convention, so `uv run task check` stays
  green with no `train`/`corpora`/`adapters` extras installed.
- The CLI's `train` subcommand (`--recipe`, `--data-dir`, `--n-docs` default
  `200000` -- comfortably above the current `pl` recipe's ~27k pre-staged
  documents, so `_reservoir`'s "fewer if the stream is shorter" behavior
  captures the whole corpus by default -- `--seed` default `42`) trains
  every `bpe`/`unigram` candidate in `recipe.candidates`, saves each via
  `save_artifact`, and prints one summary line per candidate (trained or
  skipped).

**Block If:** the installed `tokenizers` version's `BpeTrainer`/
`UnigramTrainer` does not expose the parameters this spec assumes (e.g. no
`byte_fallback` on `UnigramTrainer`, or `pre_tokenizers.ByteLevel.alphabet()`
missing) -- HALT with status `blocked`, since that is an API-version
mismatch requiring human re-verification, not a silent workaround.

**Never:**
- Do not implement SuperBPE training here -- that is Story 2.2.
- Do not commit trained `tokenizer.json` binaries to git.
- Do not re-implement weighted/reservoir corpus sampling in `trainers.py`.
- Do not add a new dependency/extra -- `tokenizers` is already in the
  `train` extra (`packages/tokenizer-core/pyproject.toml`).

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Train BPE candidate | fake `tokenizers` module; candidate `algorithm=bpe, vocab_size=100, byte_fallback=true, normalization=NFC` | `train_candidate` returns `(tokenizer, metadata)`; `metadata.algorithm == "bpe"`, `vocab_size == 100`, `corpus_hash` and `seed` set | No error |
| Train Unigram candidate | same fixture, `algorithm=unigram` | `metadata.algorithm == "unigram"` | No error |
| Unsupported algorithm | candidate `algorithm=superbpe` | -- | `train_candidate` raises `TrainerError` naming the candidate/algorithm; CLI catches it, prints a skip line, continues other candidates |
| All train corpora skipped/empty | every `train_corpora` source resolves 0 documents | -- | `TrainerError` (cannot train on zero documents) |
| Deterministic sampling | same recipe, `n_docs`, `seed` sampled twice | both calls' `corpus_hash` are identical | No error |
| Save/load round trip | a trained fake tokenizer + metadata | `save_artifact` writes `tokenizer.json` + `metadata.json`; `load_artifact` reconstructs a tokenizer and a `metadata.as_dict()` equal to the original | No error |
| Base install, no `train` extra | `uv sync --all-packages` (no extras) | `uv run task check` passes -- `trainers.py`/`artifact.py` import `tokenizers` only inside function bodies | No error at import/check time |

</intent-contract>

## Code Map

- `packages/tokenizer-core/src/tokenizer_core/corpora/sampling.py` --
  EXTEND: widen `sample_corpora`'s return type to `tuple[CorpusManifest,
  list[str]]`, returning the concatenated per-source sampled documents
  alongside the existing manifest.
- `packages/tokenizer-core/tests/test_sampling.py` -- update every
  `sample_corpora(...)` call site to unpack `(manifest, docs)`; add an
  assertion that `docs` matches the sampled content for at least one case.
- `packages/tokenizer-core/src/tokenizer_core/train/trainers.py` -- NEW:
  `TrainerError`, `train_candidate(recipe, candidate, *, n_docs, seed,
  data_dir=None) -> tuple[Tokenizer, TrainerMetadata]`, private
  `_train_bpe`/`_train_unigram`/`_build_normalizer` helpers.
- `packages/tokenizer-core/src/tokenizer_core/train/artifact.py` -- NEW:
  `TrainerMetadata` frozen dataclass (+ `as_dict`), `save_artifact(tokenizer,
  metadata, out_dir) -> tuple[Path, Path]`, `load_artifact(artifact_dir) ->
  tuple[Tokenizer, TrainerMetadata]`.
- `packages/tokenizer-core/src/tokenizer_core/train/__init__.py` -- add
  re-exports: `TrainerError`, `train_candidate`, `TrainerMetadata`,
  `save_artifact`, `load_artifact`.
- `packages/tokenizer-core/src/tokenizer_core/cli.py` -- add a `train`
  subparser (`--recipe` default `languages/pl/recipe.yaml`, `--data-dir`
  default `None`, `--n-docs` default `200000`, `--seed` default `42`) whose
  handler loads the recipe, trains each `bpe`/`unigram` candidate, saves it,
  skips unsupported algorithms with a printed message, and prints one
  summary line per candidate.
- `packages/tokenizer-core/tests/test_trainers.py` -- NEW: unit tests for
  the I/O Matrix rows above, using a fake `tokenizers` module injected via
  `sys.modules` (mirroring `test_sources.py`'s pattern) -- no real
  dependency required.
- `packages/tokenizer-core/tests/test_artifact.py` -- NEW: save/load
  round-trip tests using a minimal fake tokenizer object (`save`/
  `from_file`).
- `packages/tokenizer-core/tests/test_cli.py` -- EXTEND: add a `train`
  subcommand argparse-wiring smoke test (monkeypatched `train_candidate`/
  `save_artifact`).
- `Taskfile.yml` -- add `train:pl`: `scripts/op-run.sh uv run tokenizer
  train --recipe languages/pl/recipe.yaml`.
- `packages/tokenizer-core/src/tokenizer_core/corpora/sources.py`,
  `packages/tokenizer-core/src/tokenizer_core/train/config.py` -- READ-ONLY
  references for `resolve_source`/`_resolve_data_dir` and the
  `Recipe`/`Candidate` fields.
- `languages/pl/recipe.yaml`, `packages/tokenizer-core/pyproject.toml` --
  READ-ONLY; drive the live verification run and confirm the `train` extra
  already covers `tokenizers`.

## Tasks & Acceptance

**Execution:**
- [x] `packages/tokenizer-core/src/tokenizer_core/corpora/sampling.py` --
  widen `sample_corpora`'s return to `tuple[CorpusManifest, list[str]]`,
  returning the sampled documents (in source order) alongside the manifest
  -- rationale: training needs the exact text a run's `corpus_hash` covers,
  without re-deriving the weighted/seeded sampling logic.
- [x] `packages/tokenizer-core/tests/test_sampling.py` -- update the three
  existing `sample_corpora(...)` call sites to unpack the new tuple return
  -- rationale: keep Story 1.2's tests passing against the widened API.
  (Six call sites actually return successfully and needed unpacking, not
  three -- the other error-path tests raise before returning.)
- [x] `packages/tokenizer-core/src/tokenizer_core/train/artifact.py` -- add
  `TrainerMetadata(name, algorithm, vocab_size, byte_fallback, normalization,
  corpus_hash, seed, language_code)` + `as_dict()`; `save_artifact`/
  `load_artifact` writing/reading `tokenizer.json` (via the tokenizer's own
  `save`/`from_file`) and `metadata.json` under one directory -- rationale:
  the portable-artifact-plus-sidecar contract from the epic.
- [x] `packages/tokenizer-core/src/tokenizer_core/train/trainers.py` -- add
  `TrainerError(Exception)`; `train_candidate(recipe, candidate, *, n_docs,
  seed, data_dir=None) -> tuple[Tokenizer, TrainerMetadata]` that samples
  `recipe.train_corpora` via `corpora.sample_corpora`, dispatches to a BPE or
  Unigram build per `candidate.algorithm`, applies
  `candidate.normalization`/`candidate.byte_fallback`, trains via
  `train_from_iterator`, and returns the tokenizer plus its
  `TrainerMetadata` -- rationale: AC1, the core training entry point.
  (`sample_corpora` is imported inside the function body, not at module
  level, to break a real circular import: `corpora/sources.py` imports
  `tokenizer_core.train.config`, and `tokenizer_core.train/__init__.py`
  re-exports this module, so a module-level `tokenizer_core.corpora` import
  here closed the loop `train -> corpora -> train`.)
- [x] same file -- raise `TrainerError` for any `candidate.algorithm` other
  than `bpe`/`unigram`, and for a corpus sample with zero documents --
  rationale: keeps SuperBPE out of scope and fails clearly on unusable
  input.
- [x] `packages/tokenizer-core/src/tokenizer_core/train/__init__.py` --
  export `TrainerError, train_candidate, TrainerMetadata, save_artifact,
  load_artifact` -- rationale: documented import surface, matching
  `benchmark`'s re-export pattern.
- [x] `packages/tokenizer-core/src/tokenizer_core/cli.py` -- add a `train`
  subparser and `_run_train` handler: loads the recipe, iterates
  `recipe.candidates`, calls `train_candidate` for `bpe`/`unigram`
  candidates and `save_artifact` on the result, catches `TrainerError` per
  candidate to print a skip line and continue, prints one summary line
  (name, vocab size, corpus hash) per trained candidate -- rationale: AC1,
  a runnable one-command-per-language entry point.
- [x] `Taskfile.yml` -- add `train:pl` following the `benchmark:pl` pattern
  -- rationale: matches the project's one-task-per-language convention.
- [x] `packages/tokenizer-core/tests/test_trainers.py` -- unit tests for
  every I/O Matrix row involving `trainers.py`, using a fake `tokenizers`
  module in `sys.modules` -- rationale: proves training/dispatch logic
  without the real native dependency.
- [x] `packages/tokenizer-core/tests/test_artifact.py` -- unit tests for the
  save/load round trip -- rationale: proves the artifact contract
  independent of a real tokenizer.
- [x] `packages/tokenizer-core/tests/test_cli.py` -- test the `train`
  subcommand parses its arguments and calls `train_candidate`/
  `save_artifact` with monkeypatched functions, including one candidate that
  raises `TrainerError` -- rationale: proves the CLI wiring and the
  skip-and-continue behavior in isolation.
- [x] (not originally listed, required to keep `task check` green with no
  extras installed) `pyproject.toml` -- add
  `train/artifact.py`/`train/trainers.py` to the `[[tool.ty.overrides]]`
  list that already covers `corpora/sources.py`/`benchmark/compare.py`, so
  `ty check`'s `unresolved-import` doesn't fire on their lazy `tokenizers`
  imports in the base (no-`train`-extra) environment.

**Acceptance Criteria:**
- Given `languages/pl/recipe.yaml`'s `pl-bpe-32k` and `pl-unigram-32k`
  candidates, when `task train:pl` (or `uv run tokenizer train --recipe
  languages/pl/recipe.yaml`) runs with the `train` extra installed, then
  `tokenizer.json` and `metadata.json` are written under
  `$TOKENIZER_DATA_DIR/tokenizers/pl/pl-bpe-32k/` and
  `.../pl-unigram-32k/`, and the CLI prints a summary line for each
  including its vocab size and corpus hash.
- Given the same live run, when the `pl-superbpe-32k` candidate is reached,
  then it is skipped with a printed message and the run still completes
  (does not abort the other candidates).
- Given `uv run task check` (lint + format + types + tests) without the
  `train`/`corpora`/`adapters` extras installed, when run, then it passes --
  `trainers.py`/`artifact.py`'s only heavy import (`tokenizers`) stays
  inside function bodies, never at module level.
- Given a trained candidate's tokenizer object, when passed to
  `benchmark.fertility.fertility`/`aligned_fertility` (Story 1.3), then it
  works unmodified because a HF `tokenizers.Tokenizer` already satisfies the
  `Encoder` protocol (`.encode(text)` returning an object with `.ids`).

## Design Notes

HF `tokenizers`' byte-fallback story differs between algorithms:
`UnigramTrainer` takes a native `byte_fallback: bool` (mirrors sentencepiece
directly), but `BpeTrainer` has no such flag -- byte-complete coverage for
BPE is the GPT-2 way instead: a `ByteLevel` pre-tokenizer plus its full byte
alphabet as the trainer's initial alphabet, so every byte is representable
and no `<unk>` token is needed:

```python
tok = Tokenizer(models.BPE(unk_token=None))
tok.normalizer = normalizers.NFC()
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(
    vocab_size=candidate.vocab_size,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
)
tok.train_from_iterator(documents, trainer)
```

`sample_corpora`'s return widens because Story 1.2 only needed the manifest
(provenance/hash); Story 2.1 also needs the actual sampled text to train on,
and reusing the exact same seeded/weighted/reservoir logic (rather than
re-deriving it in `trainers.py`) keeps `corpus_hash` and "what a run saw"
single-sourced -- the recorded hash and the trained-on text can never
diverge.

## Verification

**Commands:**
- `uv run pytest packages/tokenizer-core/tests/test_trainers.py
  packages/tokenizer-core/tests/test_artifact.py
  packages/tokenizer-core/tests/test_sampling.py
  packages/tokenizer-core/tests/test_cli.py -q` -- expected: all pass.
- `uv run task check` -- expected: lint + format + types + full suite pass
  with no `train`/`corpora`/`adapters` extras installed.
- `task setup:all && task secrets:run -- uv run tokenizer train --recipe
  languages/pl/recipe.yaml` -- expected: `pl-bpe-32k` and `pl-unigram-32k`
  artifacts written under `$TOKENIZER_DATA_DIR/tokenizers/pl/...`,
  `pl-superbpe-32k` skipped; record vocab sizes, corpus hash, and rough
  timing in this section.

**Manual checks (if no CLI):**
- Inspect one written `metadata.json` for correct `algorithm`, `vocab_size`,
  `corpus_hash`, and `seed` fields before considering the story verified.

**Results (2026-07-04):**

- `uv run pytest packages/tokenizer-core/tests/test_trainers.py
  packages/tokenizer-core/tests/test_artifact.py
  packages/tokenizer-core/tests/test_sampling.py
  packages/tokenizer-core/tests/test_cli.py -q` -- **all pass**: 7 passed
  (`test_trainers.py`), 4 passed (`test_artifact.py`), 8 passed
  (`test_sampling.py`), 8 passed (`test_cli.py`) -- 27 total, 0 failed.
- `uv run task check` (base environment, `uv sync --all-packages`, no
  extras) -- **green**: `ruff check .` ("All checks passed!"), `ruff format
  --check .` ("23 files already formatted"), `ty check` ("All checks
  passed!"), `pytest` (89 passed across the whole workspace, up from 72
  after Story 1.4). Confirmed via `uv pip list` that `tokenizers`/
  `datasets`/`transformers`/`sentencepiece`/`tiktoken`/`huggingface-hub` are
  absent from the environment during this run; `train/artifact.py` and
  `train/trainers.py`'s only heavy import (`tokenizers`) stays inside
  function bodies (`_build_normalizer`, `_train_bpe`, `_train_unigram`,
  `load_artifact`), never at module level. Added both files to
  `pyproject.toml`'s `[[tool.ty.overrides]]` (alongside the existing
  `corpora/sources.py`/`benchmark/compare.py` entries) so `ty check` doesn't
  flag their lazy `tokenizers` import as `unresolved-import` when the
  `train` extra isn't installed.
- A real circular import was hit and fixed during implementation:
  `corpora/sources.py` imports `tokenizer_core.train.config`, and
  `tokenizer_core/train/__init__.py` re-exports `trainers.py` -- a
  module-level `from tokenizer_core.corpora.sampling import sample_corpora`
  in `trainers.py` closed the loop (`train` -> `corpora` -> `train`) and
  broke every import of `tokenizer_core.corpora.*`. Fixed by importing
  `sample_corpora` inside `train_candidate`'s function body instead (noted
  in both the module docstring and inline at the call site).

**Live run -- `task setup:all` then `task secrets:run -- uv run tokenizer
train --recipe languages/pl/recipe.yaml`:**

- `task setup:all` installed the `train`/`corpora`/`adapters` extras,
  pinning `tokenizers==0.22.2` (the version resolved by the repo's
  `tokenizers>=0.20` constraint). `task secrets:check` confirmed `HF_TOKEN`/
  `TOKENIZER_DATA_DIR` resolved from the `neuraffica` 1Password vault, and
  the pre-staged local corpora were present and readable:
  `$TOKENIZER_DATA_DIR/pl/train/fineweb2.txt` (22,166 docs) +
  `.../wiki.txt` (5,061 docs), 27,227 documents total, matching the spec's
  expectation.
- **API-version mismatch confirmed live, exactly as the spec's own "Block
  If" clause anticipated.** The installed `tokenizers==0.22.2`'s
  `trainers.UnigramTrainer.__init__` (confirmed via its `.pyi` stub, and
  independently at the floor of the pinned range, `tokenizers==0.20.0`) has
  **no `byte_fallback` parameter** at all -- it only accepts `vocab_size`,
  `show_progress`, `special_tokens`, `initial_alphabet`, `shrinking_factor`,
  `unk_token`, `max_piece_length`, `n_sub_iterations`. Passing
  `byte_fallback=candidate.byte_fallback` (as the spec's Boundaries
  explicitly require) does not raise -- `tokenizers` silently accepts and
  discards unknown trainer kwargs, printing `Ignored unknown kwargs option
  byte_fallback` to stderr -- so the live run **did** produce a
  `pl-unigram-32k/tokenizer.json`, but its saved model has
  `"byte_fallback": false` baked in (confirmed by inspecting the written
  JSON directly), directly contradicting its own `metadata.json` sidecar,
  which (per the spec's Design Notes) faithfully records
  `candidate.byte_fallback=true` from the recipe. This is a genuine,
  independently-reproduced instance of the documented "Block If" condition
  ("the installed `tokenizers` version's ... `UnigramTrainer` does not
  expose the parameters this spec assumes (e.g. no `byte_fallback` on
  `UnigramTrainer`) -- HALT with status `blocked`"), not a version-drift
  fluke: `byte_fallback` has never existed on `UnigramTrainer` in any
  release satisfying the repo's own `tokenizers>=0.20` pin. `models.BPE`'s
  own `byte_fallback` constructor kwarg is unrelated to and does not fix
  this (it's a different class); `models.Unigram(byte_fallback=True)` is
  the actual API surface that carries this flag in this library version,
  but wiring that in would be exactly the kind of "silent workaround" the
  spec instructs against, so `trainers.py` was left implementing the
  spec's literal (currently-broken) instruction rather than silently
  "fixing" it.
  - Per the spec's own Boundaries, this is a HALT condition requiring
    human re-verification of the intended `UnigramTrainer`/byte-fallback
    API surface (e.g. confirm whether `models.Unigram(byte_fallback=True)`
    plus a plain `UnigramTrainer(vocab_size=...)` is the intended fix, and
    whether that changes the metadata contract), not something this
    session should resolve unilaterally.
- **Separately, the live run also hit a sandbox resource ceiling** that is
  unrelated to the code and would not occur in a normal CI/dev machine:
  this environment's cgroup caps memory at `memory.max = 2147483648` (2
  GiB; confirmed via `/sys/fs/cgroup/memory.max`), and the default
  `--n-docs 200000` run against the real ~27k-document Polish corpus was
  OOM-killed (exit 137, `/sys/fs/cgroup/memory.events` showed
  `oom_kill` incrementing on each attempt) before either candidate finished
  training a 32k-vocab model. Reducing `--n-docs` to 500 completed the full
  pipeline successfully end to end (both `pl-bpe-32k` and `pl-unigram-32k`
  trained, saved, and printed a summary line; `pl-superbpe-32k` was
  correctly skipped with `candidate 'pl-superbpe-32k' has unsupported
  algorithm 'superbpe' ... 'superbpe' is Story 2.2`), but `--n-docs 2000`
  still OOM-killed and `--n-docs 5000` crashed with SIGSEGV (exit 139) in
  this constrained sandbox -- so the code path itself is confirmed
  functional at reduced scale, but a true `--n-docs 200000` run (the
  documented default, and what a real `task train:pl` invocation would do)
  could not be completed in this sandbox regardless of the `byte_fallback`
  issue above. This looks like a sandbox-specific ceiling, not a code
  defect: `tokenizers`' Rust-side BPE/Unigram trainers are known to hold
  significant transient memory for word-frequency tables at this vocab
  size, and 2 GiB is a very tight cap for a 32k-vocab trainer pass over
  even a moderate corpus.
  - The reduced-scale (`--n-docs 500`) run's artifacts were inspected then
    deleted (not left as stand-ins for the real, full-corpus artifacts):
    `pl-bpe-32k/tokenizer.json` (2,329,846 bytes, `model.type == "BPE"`,
    `pre_tokenizer.type == "ByteLevel"`, `decoder.type == "ByteLevel"`,
    vocab size 32000 -- matches the spec's GPT-2-style pipeline exactly)
    and `pl-unigram-32k/tokenizer.json` (2,122,991 bytes, `model.type ==
    "Unigram"`, vocab size 32000, `model.byte_fallback == false` --
    the mismatch above). Both directories' `metadata.json` correctly
    recorded `algorithm`, `vocab_size=32000`, `byte_fallback=true`
    (as given by the recipe, regardless of whether the underlying model
    actually honored it), a `corpus_hash`, `seed=42`, and
    `language_code="pl"`.
  - No fabricated timing/vocab numbers are recorded for the real
    `--n-docs 200000` run since it never completed in this sandbox; a
    human/CI environment without this cgroup's 2 GiB cap should be able to
    run `task train:pl` directly once the `byte_fallback` API mismatch
    above is resolved.
- The environment was returned to the base, no-extras state (`uv sync
  --all-packages`) after this live-run attempt, and `task check` was
  re-verified green in that state (see above, 89 passed) -- the
  `train`/`corpora`/`adapters` extras installed for this attempt are not
  left in the committed environment state. `uv pip list` after the restore
  showed no `tokenizers`/`datasets`/`transformers`/etc.

## Auto Run Result

Status: blocked

**Blocking condition:** the spec's own explicit "Block If" clause fired
during live verification: the installed `tokenizers==0.22.2`'s
`trainers.UnigramTrainer` (confirmed via its `.pyi` type stub, and
independently re-confirmed at `tokenizers==0.20.0`, the floor of this
repo's `tokenizers>=0.20` pin) has no `byte_fallback` parameter at all.
Passing it (as the spec's Boundaries require: "A `unigram` candidate passes
`byte_fallback` straight through to `tokenizers.trainers.UnigramTrainer`'s
native `byte_fallback` parameter") does not raise -- `tokenizers` silently
drops unknown trainer kwargs with a printed warning -- so the code runs
without crashing, but the resulting `pl-unigram-32k` tokenizer never
actually gets byte-fallback coverage: `tokenizer.json`'s
`model.byte_fallback` is `false` regardless of `candidate.byte_fallback`,
silently contradicting the `metadata.json` sidecar that (correctly, per
its own contract) records the recipe's requested value. This was
independently confirmed live (not just via static stub inspection): a
reduced-scale `tokenizer train` run produced exactly this
metadata/model-reality mismatch.

All code, tests, and the `task check` gate are complete and green (see
Tasks & Acceptance and Verification above); the implementation matches the
spec's literal instructions exactly (including the exact
`trainers.UnigramTrainer(vocab_size=..., byte_fallback=candidate.
byte_fallback)` call the spec's Boundaries mandate) -- this is not a coding
mistake, it is the documented HALT condition itself. Per the spec's
Boundaries ("HALT with status `blocked` ... not a silent workaround"), no
attempt was made to unilaterally switch to `models.Unigram(byte_fallback=
True)` (a different, model-level API in this same library that does carry
the flag) since that would change both the trained model's actual
behavior and arguably the metadata contract's meaning, and the spec
explicitly reserves that decision for human re-verification.

Separately (and independently of the above), the live run's default
`--n-docs 200000` could not complete in this session's sandbox due to a 2
GiB cgroup memory cap unrelated to the code; `--n-docs 500` proved the
full pipeline (both candidates train + save, `pl-superbpe-32k` skips
correctly) works end to end at reduced scale.

**To unblock:** a human should decide how `train_candidate`'s Unigram path
should honor `byte_fallback` given the installed `tokenizers` version's
real API surface -- e.g. (a) switch to `models.Unigram(byte_fallback=
candidate.byte_fallback)` at construction time (paired with a plain
`UnigramTrainer(vocab_size=...)`, no `byte_fallback` kwarg) if that
achieves equivalent behavior, and update the spec's Design Notes/Boundaries
to match; or (b) pin/require a different `tokenizers` version if one
genuinely exposes `UnigramTrainer(..., byte_fallback=...)` (none was found
in this session's search); or (c) accept `byte_fallback` is BPE-only in
practice for this library version and adjust the recipe/spec's semantics
accordingly. Then re-run `task setup:all && task secrets:run -- uv run
tokenizer train --recipe languages/pl/recipe.yaml` (ideally outside this
2 GiB-capped sandbox, or with a smaller `--n-docs` override) to produce and
commit-reference the real artifacts.
