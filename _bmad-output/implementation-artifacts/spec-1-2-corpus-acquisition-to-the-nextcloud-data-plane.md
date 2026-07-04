---
title: 'Corpus acquisition to the Nextcloud data plane'
type: 'feature'
created: '2026-07-04'
status: 'done'
baseline_revision: d249f08228654f89ae2da875e6626adfb4867201
final_revision: 98a984d
review_loop_iteration: 0
followup_review_recommended: false
context: []
warnings: [oversized]
---

<intent-contract>

## Intent

**Problem:** Story 1.1 gives us a typed, validated `Recipe`, but nothing yet
turns a recipe's declared `train_corpora`/`eval_corpora` (`CorpusSpec` entries)
into actual text — pulled from Hugging Face / a URL / local disk, mixed by
seeded weighted sampling, cached off-git under `$TOKENIZER_DATA_DIR`, and
recorded in a reproducible manifest. Without this, Stories 1.3/1.4 (fertility,
baselines) have no corpus to measure against.

**Approach:** Add `corpora.sources.resolve_source` to turn one `CorpusSpec`
into a stream of text documents (hf via `datasets` streaming, url via stdlib
`urllib`, local via direct file read), caching url downloads under the data
plane. Add `corpora.sampling.sample_corpora` to draw a deterministic, seeded,
weighted mix of documents across a recipe's corpora and write a JSON manifest
(per-source counts + content hashes) under the data plane.

## Boundaries & Constraints

**Always:**
- `resolve_source(spec: CorpusSpec, *, data_dir: str | os.PathLike) ->
  tuple[Iterator[str], str | None]` in `corpora/sources.py`. Returns
  `(documents, skip_reason)`; `skip_reason` is `None` unless the source was
  gracefully skipped. A "document" is one non-empty, stripped line for
  `url`/`local` sources, or one example's text field for `hf`.
- `hf`: lazy `import datasets` inside the function body (never at module top
  level); call `datasets.load_dataset(spec.id, name=spec.config,
  split=spec.split, streaming=True)`. Extract each example's `"text"` field,
  falling back to the first `str`-valued column if `"text"` is absent.
- `hf` gating: if `spec.split is None`, raise `SourceError` (ambiguous —
  out of scope). Otherwise, if loading/iterating raises and neither
  `HF_TOKEN` nor `HUGGINGFACE_HUB_TOKEN` is set in the environment, treat it
  as a gated/inaccessible source: return `(iter(()), skip_reason)` where
  `skip_reason` names the source id and the missing env var — never raise.
  If a token IS set and it still fails, raise `SourceError` chaining the
  original exception (do not mask a real failure once credentials exist).
- `url`: fetch via stdlib `urllib.request.urlopen(spec.id)` (no new
  dependency; supports `http(s)://` and, for tests, `file://`). Cache the raw
  bytes to `{data_dir}/corpora/url/<sha256(spec.id)[:16]>.txt` before
  streaming lines from the cached file, so repeated calls don't re-fetch.
- `local`: `spec.id` is a filesystem path to a UTF-8 text file; stream its
  non-empty lines directly. Do not copy it into `data_dir`.
- `corpora/sampling.py`: `sample_corpora(corpora: list[CorpusSpec], *,
  n_docs: int, seed: int, language_code: str, data_dir: str | os.PathLike) ->
  CorpusManifest`. Weight shares are normalized across all sources passed in
  (including skipped ones); `k_i = floor(share_i * n_docs)`, then the
  remainder `n_docs - sum(k_i)` is handed out one-by-one to sources in
  descending-weight order (ties broken by input order) so totals sum exactly
  to `n_docs` when nothing is skipped. Use reservoir sampling (`random.Random
  (seed)`) to draw `k_i` documents from each source's stream — deterministic
  for a given seed and source content, and correct for unbounded/unknown-
  length streams. A skipped source contributes 0 docs; its slice is **not**
  redistributed to other sources (`n_docs_sampled` in the manifest may be <
  `n_docs_requested` when sources are skipped).
- `CorpusManifest`/`SourceManifestEntry` are frozen dataclasses (mirrors
  `benchmark.fertility.FertilityReport`) with an `as_dict()` for JSON. Each
  entry records `source, id, config, split, domain, weight, n_docs_sampled,
  skipped, skip_reason, content_sha256` (sha256 hex digest of that source's
  sampled docs joined by `\n`). `sample_corpora` writes the manifest to
  `{data_dir}/corpora/{language_code}/manifest.json` (creating parent dirs)
  and also returns it.
- Both `data_dir` params accept an explicit path (tests always pass
  `tmp_path`); if omitted, default to `os.environ["TOKENIZER_DATA_DIR"]` and
  raise a clear `SourceError`/`ValueError` if that env var is also unset.
- Only base deps + the already-declared `corpora` extra (`datasets`,
  `huggingface-hub`); stdlib `urllib`/`hashlib`/`random` for the rest. Google
  docstrings, `from __future__ import annotations`, double quotes, 80-col.
- Unit tests must pass with **no** `corpora` extra installed (CI only runs
  `uv sync --all-packages`, no extras): fake the `hf` path by injecting a
  fake module via `monkeypatch.setitem(sys.modules, "datasets", fake)` so the
  lazy `import datasets` picks up the fake, never the real package.

**Block If:** No blocking decision identified — the gated-skip heuristic,
no-redistribution-on-skip, and `split=None`-rejection choices above resolve
every ambiguity found during investigation without needing a human call.

**Never:** Do not implement `train.trainers`, fertility, or baseline
comparison (Stories 1.3/1.4+). Do not add a CLI subcommand. Do not make real
network calls in tests (`file://` URLs / fakes / `tmp_path` only). Do not
non-streaming-download full HF datasets. Do not run acquisition against the
real `languages/pl/recipe.yaml` in tests (needs network + `HF_TOKEN`).

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Valid hf spec | fake `datasets.load_dataset` yields 5 examples with `"text"` | `resolve_source` yields the 5 text values, `skip_reason=None` | No error |
| hf example missing `"text"` | fake example is `{"sentence": "..."}` only | Falls back to the first `str` column (`"sentence"`) | No error |
| hf spec, `split=None` | `CorpusSpec(source="hf", split=None, ...)` | Raises `SourceError` naming the missing split | Raised, not skipped |
| hf gated, no token | fake `load_dataset` raises; `HF_TOKEN`/`HUGGINGFACE_HUB_TOKEN` unset | `(iter(()), skip_reason)` naming id + env var | No exception |
| hf gated, token set | fake `load_dataset` raises; `HF_TOKEN` set | Raises `SourceError` chaining original | Raised, not skipped |
| Valid url spec | `file://{tmp_path}/c.txt` with 3 non-empty + 1 blank line | Yields the 3 non-empty lines; cache file created under `data_dir/corpora/url/` | No error |
| Repeated url fetch | same spec resolved twice | Second call reads the cache file, does not re-`urlopen` | No error |
| Valid local spec | `CorpusSpec(source="local", id=str(tmp_file))` | Yields non-empty lines from the file, no copy under `data_dir` | No error |
| Local path missing | `id` points at a non-existent file | Raises `SourceError` naming the path | Raised |
| Weighted seeded mix | 3 local `CorpusSpec`s, weights `0.5/0.3/0.2`, `n_docs=10`, `seed=7` | `k_i = [5,3,2]`; manifest `n_docs_sampled == 10`; identical hashes on a second call with the same seed | No error |
| Mix with one skipped | 3 sources, one hf-gated-and-skipped | Skipped entry `n_docs_sampled=0, skipped=True`; manifest `n_docs_sampled < n_docs_requested` (no redistribution) | No error |
| Different seed | same 3 sources, `seed=8` vs `seed=7` | At least one source's sampled doc set/hash differs | No error |

</intent-contract>

## Code Map

- `packages/tokenizer-core/src/tokenizer_core/corpora/sources.py` -- NEW:
  `SourceError`, `resolve_source`, private `_resolve_hf`/`_resolve_url`/
  `_resolve_local`, `_cache_path_for_url`.
- `packages/tokenizer-core/src/tokenizer_core/corpora/sampling.py` -- NEW:
  `SourceManifestEntry`, `CorpusManifest`, `sample_corpora`.
- `packages/tokenizer-core/src/tokenizer_core/corpora/__init__.py` -- replace
  the placeholder docstring with re-exports (`resolve_source`, `SourceError`,
  `sample_corpora`, `CorpusManifest`, `SourceManifestEntry`) + `__all__`,
  mirroring `train/__init__.py`'s pattern.
- `packages/tokenizer-core/tests/test_sources.py` -- NEW: I/O matrix rows
  1-8, hf via `sys.modules` fake injection, url via `file://` fixtures.
- `packages/tokenizer-core/tests/test_sampling.py` -- NEW: I/O matrix rows
  9-11, local `CorpusSpec` fixtures pointing at tiny `tmp_path` text files.
- `packages/tokenizer-core/src/tokenizer_core/train/config.py` -- READ-ONLY
  reference: `CorpusSpec` fields (`source`, `id`, `config`, `split`, `weight`,
  `domain`) this story consumes.
- `packages/tokenizer-core/src/tokenizer_core/benchmark/fertility.py` --
  style + `FertilityReport`-style frozen-dataclass reference to mirror.
- `AGENT.md` -- data-plane (`$TOKENIZER_DATA_DIR`) and lazy-extra-import
  conventions this story must follow.

## Tasks & Acceptance

**Execution:**
- [x] `packages/tokenizer-core/src/tokenizer_core/corpora/sources.py` -- Add
  `SourceError(RuntimeError)`; `resolve_source(spec, *, data_dir=None) ->
  tuple[Iterator[str], str | None]` dispatching on `spec.source` to
  `_resolve_hf`/`_resolve_url`/`_resolve_local` per the Boundaries above --
  rationale: single typed entry point Story 1.3+ and `sample_corpora` build
  on.
- [x] `packages/tokenizer-core/src/tokenizer_core/corpora/sampling.py` -- Add
  `SourceManifestEntry`/`CorpusManifest` frozen dataclasses (+ `as_dict`) and
  `sample_corpora(corpora, *, n_docs, seed, language_code, data_dir=None) ->
  CorpusManifest` implementing weighted-share + reservoir sampling and
  writing `{data_dir}/corpora/{language_code}/manifest.json` -- rationale:
  the seeded, reproducible mix + provenance manifest AC2 requires.
- [x] `packages/tokenizer-core/src/tokenizer_core/corpora/__init__.py` --
  Replace the placeholder with re-exports and `__all__` -- rationale:
  `tokenizer_core.corpora` becomes the documented import surface.
- [x] `packages/tokenizer-core/tests/test_sources.py` -- Cover every I/O
  Matrix row for `resolve_source` (hf fake-module injection via
  `monkeypatch.setitem(sys.modules, ...)`, url via `file://` + cache-reuse
  assertion, local success/missing-path) -- rationale: proves AC1/AC3 with
  no network and no `corpora` extra installed.
- [x] `packages/tokenizer-core/tests/test_sampling.py` -- Cover the weighted-
  mix, skipped-source, and different-seed rows using local `CorpusSpec`
  fixtures; assert exact `k_i` counts, manifest totals, and hash
  reproducibility/divergence -- rationale: proves AC2 determinism.

**Acceptance Criteria:**
- Given a recipe's `train_corpora`/`eval_corpora` (`hf`/`url`/`local`
  `CorpusSpec`s), when each is passed to `resolve_source`, then it returns a
  document stream resolved per its source type, caching `url` downloads
  under `$TOKENIZER_DATA_DIR`.
- Given a list of `CorpusSpec`s with weights and a seed, when
  `sample_corpora` runs twice with the same seed, then both runs produce
  identical per-source document counts and `content_sha256` hashes, and the
  written manifest reflects them.
- Given an `hf` `CorpusSpec` whose repo is gated and no `HF_TOKEN`/
  `HUGGINGFACE_HUB_TOKEN` is set, when it is resolved, then `resolve_source`
  returns an empty stream with a `skip_reason` message (no exception raised)
  and `sample_corpora` records it as `skipped` in the manifest.
- Given `task check` (ruff lint + format + ty + pytest), when run without
  the `corpora` extra installed, then it passes.

## Spec Change Log

## Review Triage Log

### 2026-07-04 — Review pass
- intent_gap: 0
- bad_spec: 0
- patch: 8: (high 0, medium 3, low 5)
- defer: 6: (high 0, medium 2, low 4)
- reject: 2: (high 0, medium 0, low 2)
- addressed_findings:
  - `[medium]` `[patch]` `sample_corpora` raised an unhandled `ZeroDivisionError` on an empty `corpora` list (bypasses `Recipe`'s `min_length=1` since `sample_corpora` is directly callable) — added an explicit `ValueError` guard.
  - `[medium]` `[patch]` Negative `n_docs` silently corrupted the remainder-distribution slice (`order_by_weight[:remainder]`) — added a `ValueError` guard.
  - `[medium]` `[patch]` `_resolve_hf`'s try/except only wrapped `datasets.load_dataset(...)`, so a streaming dataset that fails lazily on first fetch (the common case for auth/gating errors) escaped the graceful-skip contract entirely — now eagerly probes the first example inside the guarded block via `itertools.islice`/`itertools.chain`, added tests for both the no-token-skip and token-set-raise paths under lazy failure.
  - `[low]` `[patch]` `content_sha256` hashed `"\n".join(docs)`, so two distinct document lists could collide if a document itself contained an embedded newline (plausible for `hf` `text` fields) — switched to a length-prefixed, collision-safe hash.
  - `[low]` `[patch]` `_reservoir` fully drained a source's stream even when its computed quota was `0` (weight rounds down), wastefully iterating a full streaming `hf` dataset for zero benefit — added a `k <= 0` short-circuit, with a direct unit test proving the stream is never consumed.
  - `[low]` `[patch]` `_iter_lines` let a raw `UnicodeDecodeError`/`OSError` escape for non-UTF-8 or unreadable files instead of the module's single `SourceError` contract — wrapped the read and added a non-UTF-8 fixture test.
  - `[low]` `[patch]` `SourceManifestEntry` recorded `n_docs_sampled` but never the per-source quota it was measured against, so a short/exhausted (non-skipped) source was indistinguishable from one that simply had a small quota — added `n_docs_quota`.
  - `[low]` `[patch]` `resolve_source`'s docstring didn't note that only `url` sources are cached under `data_dir` (an asymmetry with `hf`, which relies on `datasets`'s own `$HF_HOME` cache) — added a clarifying docstring note.
  - deferred to `deferred-work.md`: shared `random.Random(seed)` reused across every corpus rather than a per-source derived seed (medium); `sample_corpora` non-atomic across sources on mid-loop failure (medium); `url` cache has no invalidation/ETag/TTL strategy (low); `_iter_hf_examples`'s "first str column" fallback can pick a metadata column over real text (low; spec-directed); `_resolve_url`'s cache write is not atomic (low); `resolve_source`'s graceful-skip contract exists only for `hf`, not `url`/`local` (low).
  - rejected as noise: "skipped-source weight not redistributed" is the spec's explicit, documented Boundaries behavior, not a defect; "zero-quota-due-to-rounding indistinguishable from a gated skip" is already differentiated by the `skipped` boolean field.

### 2026-07-04 — Review pass
- intent_gap: 0
- bad_spec: 0
- patch: 4: (high 0, medium 2, low 2)
- defer: 2: (high 0, medium 1, low 1)
- reject: 10: (high 0, medium 2, low 8)
- addressed_findings:
  - `[medium]` `[patch]` `resolve_source` unconditionally required a resolvable `data_dir` even for `hf`/`local` specs that never use one (only `url` caches under `data_dir`, per the function's own docstring note) — an `hf`- or `local`-only recipe run with no `data_dir`/`$TOKENIZER_DATA_DIR` configured would fail with a spurious `SourceError`. Moved `data_dir` resolution into the `url`-only branch; added tests proving `hf`/`local` resolve without `data_dir`/env set, and repointed the old "missing data_dir raises" test at a `url` spec (the only source type it still applies to).
  - `[medium]` `[patch]` `_resolve_hf`'s graceful-skip/raise contract (eager first-fetch probe) only covered failures at-or-before the first fetched item; a stream failing on item 2+ escaped the `try`/`except` entirely and propagated as a raw non-`SourceError` exception straight out of `sample_corpora`, undermining the module's single-typed-exception contract for a plausible real failure mode (mid-stream auth/connection failures). Wrapped the remainder of iteration (`_wrap_hf_iteration_errors`) so any later failure raises `SourceError`; added a test for a failure on the second item.
  - `[low]` `[patch]` `resolve_source`'s docstring `Raises` section omitted the `local`/cached-`url` non-UTF-8 decode failure and didn't note it surfaces lazily on iteration rather than at call time (a caller wrapping only the `resolve_source(...)` call, not the iteration, would miss it) — updated the docstring.
  - `[low]` `[patch]` `sample_corpora`'s manifest write (`mkdir`/`write_text`) could raise a raw `OSError`/`PermissionError` instead of the module's typed `SourceError` if `data_dir` were unwritable, breaking the stated single-typed-entry-point exception contract — wrapped the write in `try`/`except OSError`, added a test using a pre-existing file in place of the manifest's parent directory.
  - deferred to `deferred-work.md`: `sample_corpora` doesn't detect/reject duplicate `CorpusSpec` entries (same `source`+`id`) within one call, so a copy-paste recipe mistake silently double-counts that source's weight share (medium); `sample_corpora`'s manifest write is not atomic/lock-protected, so a crash mid-write or concurrent same-`language_code` calls can leave `manifest.json` truncated or interleaved (low; distinct from the already-deferred `url`-cache write atomicity).
  - rejected as noise or already tracked: `sample_corpora` not exception-safe across sources on mid-loop failure, shared `random.Random(seed)` per corpus, `_iter_hf_examples`'s fragile first-str-column fallback, `_resolve_url`'s no-invalidation cache and non-atomic cache write are all already open entries in `deferred-work.md` from the prior pass — re-raised by this pass's independent reviewers but not re-logged as duplicates; `_content_hash`'s docstring wording ("collision-safe") read as overclaiming SHA-256's cryptographic collision resistance — not a real defect; a negative remainder in `sample_corpora`'s quota rounding is mathematically unreachable given `CorpusSpec.weight`'s `gt=0, allow_inf_nan=False` validation and realistic `n_docs` values; `_iter_hf_examples` silently yielding nothing for an example with zero string-valued columns matches the existing skip-on-empty/malformed convention used elsewhere (e.g. blank lines in `_iter_lines`) and causes no manifest inaccuracy; commentary that the spec's own `followup_review_recommended: true` was itself evidence of a defect is not a grounded code finding.

## Design Notes

Reservoir sampling (algorithm R) draws `k` items from a stream of unknown
length deterministically for a given seed:

```python
def _reservoir(stream: Iterator[str], k: int, rng: random.Random) -> list[str]:
    sample: list[str] = []
    for i, doc in enumerate(stream):
        if i < k:
            sample.append(doc)
        else:
            j = rng.randint(0, i)
            if j < k:
                sample[j] = doc
    return sample
```

Gated-skip pattern in `_resolve_hf` -- broad catch is intentional: without
credentials, many distinct failures (403 gated, 401 auth, private-repo 404)
all mean "can't get this corpus," so treat any failure as skip-worthy when no
token is configured, but never swallow a failure once a token is present:

```python
try:
    ds = datasets.load_dataset(spec.id, name=spec.config,
                                split=spec.split, streaming=True)
except Exception as err:
    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")):
        return iter(()), f"skipping gated/inaccessible hf source {spec.id!r}: {err} (set HF_TOKEN to retry)"
    raise SourceError(f"failed to load hf source {spec.id!r}") from err
```

## Verification

**Commands:**
- `uv run pytest packages/tokenizer-core/tests/test_sources.py packages/tokenizer-core/tests/test_sampling.py -q` -- expected: all tests pass, no `corpora` extra required.
- `uv run ruff check . && uv run ruff format --check .` -- expected: clean.
- `uv run ty check` -- expected: no errors.
- `task check` -- expected: full local gate green.

## Auto Run Result

Status: done

**Summary:** Added corpus acquisition and seeded weighted sampling for the
tokenizer library. `tokenizer_core.corpora.sources.resolve_source` turns a
`CorpusSpec` (`hf`/`url`/`local`) into a stream of text documents: `hf` via
lazily-imported streaming `datasets` (with a gated/inaccessible-source
graceful-skip when no `HF_TOKEN`/`HUGGINGFACE_HUB_TOKEN` is set, covering
load-time, lazy first-iteration, and now mid-stream failures), `url` via
stdlib `urllib` with caching under `$TOKENIZER_DATA_DIR`, `local` via
direct file read — and `data_dir` is only ever required for `url` sources.
`tokenizer_core.corpora.sampling.sample_corpora` draws a seeded, weighted,
reproducible document mix across a recipe's corpora via reservoir sampling
and writes a JSON provenance manifest (per-source quotas, sampled counts,
skip status, and collision-safe content hashes), with manifest-write
failures surfaced through the module's own `SourceError` type.

This is a follow-up independent review pass over the already-implemented
and committed story (baseline unchanged; no new files).

**Files changed (this pass):**
- `packages/tokenizer-core/src/tokenizer_core/corpora/sources.py` —
  `resolve_source` no longer resolves `data_dir` for `hf`/`local` specs
  (only `url` needs it); `_resolve_hf` now wraps mid-stream iteration
  failures (past the first item) as `SourceError` via a new
  `_wrap_hf_iteration_errors` helper instead of letting them escape raw;
  docstring `Raises` section updated for the non-UTF-8 lazy-raise case.
- `packages/tokenizer-core/src/tokenizer_core/corpora/sampling.py` —
  `sample_corpora`'s manifest write is now wrapped in `try`/`except OSError`
  and re-raised as `SourceError`.
- `packages/tokenizer-core/tests/test_sources.py`,
  `test_sampling.py` — added tests for `hf`/`local` resolving without
  `data_dir`/`$TOKENIZER_DATA_DIR` set, a mid-stream `hf` iteration failure
  raising `SourceError`, and a manifest-write failure raising `SourceError`;
  repointed the old "missing data_dir" test at a `url` spec (the only
  source type that still needs one).
- `_bmad-output/implementation-artifacts/deferred-work.md` (modified) — 2
  new deferred review findings appended.

**Review findings breakdown:** 4 patches applied (2 medium: `resolve_source`
spuriously requiring `data_dir` for `hf`/`local` specs that never use it,
and `_resolve_hf`'s graceful-skip/raise contract not covering mid-stream
iteration failures past the first item; 2 low: an incomplete `Raises`
docstring, and a raw `OSError` escaping `sample_corpora`'s manifest write
instead of the module's typed `SourceError`), 2 deferred (duplicate
`CorpusSpec` entries silently double-counting a source's weight share;
`sample_corpora`'s manifest write not atomic/lock-protected), 10 rejected
(6 already tracked in `deferred-work.md` from the prior pass and re-surfaced
by this pass's independent reviewers without prior context; 4 judged not
real defects on inspection — a docstring wording nit, a mathematically
unreachable negative-remainder edge case given existing weight validation,
a silent-skip path consistent with the codebase's existing empty/malformed
convention, and meta-commentary not grounded in code). 0 intent_gap, 0
bad_spec — no repair loopback.

**Verification:** `task check` green — ruff lint + `ruff format --check`
clean, `ty check` clean, `pytest` 50 passed (25 in the two corpora test
modules), all without the `corpora` extra installed (matching CI's `uv
sync --all-packages`, no extras). Committed on branch `romantic-dolphin`;
not pushed.

**Residual risks:** The 2 newly deferred findings and the 6 previously
deferred findings are real but out-of-scope robustness/design refinements
(duplicate-source detection, manifest/cache write atomicity, RNG
independence across sources, url-cache invalidation, hf text-field
detection) rather than correctness gaps against this story's acceptance
criteria. `followup_review_recommended: false` — this pass's patches were
narrower in scope and volume than the prior pass (4 vs. 8), self-contained
to `resolve_source`/`_resolve_hf`/the manifest write, fully covered by new
targeted tests, and did not touch the sampling math, the manifest schema,
or any security-relevant surface.
