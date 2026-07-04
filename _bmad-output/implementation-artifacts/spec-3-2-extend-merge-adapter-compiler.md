---
title: 'Extend/merge adapter (compiler)'
type: 'feature'
created: '2026-07-04'
status: 'done'
review_loop_iteration: 0
followup_review_recommended: true
context: []
warnings: ['oversized']
baseline_revision: '413537c0e5dc0f38b58c7419c2e1e5180b209082'
---

<intent-contract>

## Intent

**Problem:** A trained Polish tokenizer candidate (Epic 2) has no way to be grafted onto an existing target model's tokenizer — `tokenizer_core.adapters` only has a registry (Story 3.1) that resolves a target's format/vocab/loader, but nothing that actually merges Polish vocabulary into it while preserving the target's own tokens.

**Approach:** Add `tokenizer_core.adapters.extend.extend(family, polish_artifact_dir)`, which loads a target's backend BPE tokenizer via the Story 3.1 registry, loads a trained Polish BPE/SuperBPE artifact (Story 2.1/2.2), and appends the Polish vocab/merges after the target's own — new tokens get ids starting at `max(target_id) + 1`, existing target ids are never reassigned. Also add `adapters.base.AdapterResult`, the shared `(tokenizer, added_tokens, base_vocab_size)` contract Story 3.3's `replace` adapter will reuse.

## Boundaries & Constraints

**Always:**
- `extend(family: str, polish_artifact_dir: str | os.PathLike[str]) -> AdapterResult` lives in `tokenizer_core/adapters/extend.py`; `tokenizers` is imported lazily inside the function body (`# noqa: PLC0415`), matching `train/artifact.py::load_artifact`'s guard convention.
- Only `TargetModelEntry.format == "hf_tokenizers"` targets are supported (the merge operates on the target's raw `tokenizers.Tokenizer` BPE model dict). For any other format, raise `AdapterError` naming the family and its actual format, before any tokenizer load is attempted.
- Only a Polish artifact whose `TrainerMetadata.algorithm` is `"bpe"` or `"superbpe"` is supported (both are merge-based; `"unigram"` has no merge list to append). Check this immediately after loading the artifact, before touching the target.
- The target tokenizer is obtained via `load_target_tokenizer(family)` (Story 3.1); its raw BPE tokenizer is its `.backend_tokenizer` attribute (present when `.is_fast` is `True`, per `transformers` fast-tokenizer convention). If `.is_fast` is falsy or `.backend_tokenizer` is missing, raise `AdapterError`.
- Merge algorithm: read both tokenizers' JSON state via `json.loads(tok.to_str())`. Build the extended vocab by copying the target's `model.vocab` unchanged, then appending each Polish vocab entry whose token string is not already a target vocab key, assigning it the next sequential id after the target's current max id. Build the extended merges by keeping the target's `model.merges` unchanged (same order) then appending the Polish tokenizer's `model.merges` entries that are not already present, in their original relative order. Leave every other key in the target's tokenizer JSON (`normalizer`, `pre_tokenizer`, `post_processor`, `decoder`, `added_tokens`, and all `model` keys besides `vocab`/`merges`) untouched, then reconstruct via `tokenizers.Tokenizer.from_str(json.dumps(data))`.
- After construction, assert every target vocab entry's `(token, id)` pair resolves identically via `merged.token_to_id(token) == id` (existing-id-preservation invariant, AC2). If any mismatch, raise `AdapterError` — this must never trigger given the append-only merge above, but stands as a hard guard, not a silent skip.
- `AdapterResult` (new, in `adapters/base.py`) is a frozen dataclass: `tokenizer: Tokenizer`, `added_tokens: tuple[str, ...]` (new Polish token strings, in ascending id order), `base_vocab_size: int` (the id boundary — target vocab size before extension). This is the shared contract Story 3.3 (`replace`) will also return.
- `adapters/__init__.py` re-exports `AdapterResult` and `extend` alongside the existing registry API.
- Add `tokenizers>=0.20` to the `adapters` extra in `packages/tokenizer-core/pyproject.toml` (extend.py needs the raw `tokenizers` API directly, not just what `transformers` re-exposes).
- Add `packages/tokenizer-core/src/tokenizer_core/adapters/extend.py` to the root `pyproject.toml`'s `[[tool.ty.overrides]]` `include` list (same `unresolved-import = "ignore"` treatment as `registry.py`, since it lazily imports `tokenizers`).
- Any failure loading the Polish artifact (missing files, bad metadata) is wrapped into `AdapterError` chained `from err` — callers never see a raw `FileNotFoundError` from this module.

**Block If:** none identified — target format and Polish algorithm scoping are concrete, registry-driven decisions, not open questions.

**Never:**
- Never implement the `replace` adapter (Story 3.3) or OMP embedding init (Story 3.4) here — this story is `extend` (+ the shared `AdapterResult` contract) only.
- Never reassign or mutate any of the target's original token ids, special tokens, normalizer, or pre/post-processing config.
- Never silently fall back to a different merge strategy for sentencepiece/tiktoken targets — raise `AdapterError` instead (no format handler for those in this story).

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Happy path | `family="gpt2"` (hf_tokenizers), tiny real Polish BPE artifact with novel diacritic tokens | `AdapterResult` whose `tokenizer` preserves every target id, appends Polish-only tokens at ids ≥ target vocab size, and `added_tokens` lists exactly those new strings | No error expected |
| Unknown family | `family="falcon"` | Raised before any load | `AdapterError` naming the unknown family |
| Non-`hf_tokenizers` target | `family="mistral"` (sentencepiece) or `family="kimi"` (tiktoken) | No load attempted | `AdapterError` naming the family and its actual format |
| Unigram Polish artifact | Artifact `metadata.algorithm == "unigram"` | No target load attempted | `AdapterError` stating merge-based algorithms only |
| Overlapping vocab | Polish vocab contains a token string already in the target's vocab (e.g. common ASCII subword) | Target's existing id for that string is kept; no duplicate entry or id collision | No error expected |
| Missing/corrupt Polish artifact | `polish_artifact_dir` has no `tokenizer.json`/`metadata.json` | No target load attempted | `AdapterError` chaining the underlying `FileNotFoundError` |
| Target has no fast backend | Stand-in `AutoTokenizer` result with `is_fast=False` (no `backend_tokenizer`) | No merge attempted | `AdapterError` naming the family |

</intent-contract>

## Code Map

- `packages/tokenizer-core/src/tokenizer_core/adapters/base.py` -- NEW: `AdapterResult` frozen dataclass (`tokenizer`, `added_tokens`, `base_vocab_size`) — shared extend/replace return contract.
- `packages/tokenizer-core/src/tokenizer_core/adapters/extend.py` -- NEW: `extend(family, polish_artifact_dir) -> AdapterResult`, implementing the merge algorithm above.
- `packages/tokenizer-core/src/tokenizer_core/adapters/__init__.py` -- add re-exports: `AdapterResult`, `extend`.
- `packages/tokenizer-core/src/tokenizer_core/adapters/registry.py` -- reference only (no changes): `REGISTRY`, `TargetModelEntry.format`, `load_target_tokenizer`, `AdapterError`.
- `packages/tokenizer-core/src/tokenizer_core/train/artifact.py` -- reference only (no changes): `load_artifact` returns `(Tokenizer, TrainerMetadata)` for the Polish candidate.
- `packages/tokenizer-core/src/tokenizer_core/benchmark/fertility.py` -- reference only (no changes): `fertility()` used by tests to verify AC3 (fertility improves after extension).
- `packages/tokenizer-core/tests/test_extend.py` -- NEW: unit tests using a real tiny `tokenizers.Tokenizer` as the fake `transformers` stand-in's `backend_tokenizer` (mirrors `test_trainers.py`'s "real library, tiny corpus" philosophy — a faked merge algorithm proves little) plus `train_candidate`/`save_artifact` to produce a genuine tiny Polish artifact.
- `packages/tokenizer-core/pyproject.toml` -- add `tokenizers>=0.20` to the `adapters` extra.
- `pyproject.toml` (root) -- add `adapters/extend.py`, `adapters/base.py`, and `tests/test_extend.py` to the existing `[[tool.ty.overrides]]` include list (all three lazily/conditionally reference `tokenizers` and trip `unresolved-import` when the extra isn't installed, matching CI's default `uv sync --all-packages`).

## Tasks & Acceptance

**Execution:**
- [x] `packages/tokenizer-core/src/tokenizer_core/adapters/base.py` -- Add `AdapterResult` frozen dataclass -- gives `extend` (and later `replace`) a shared, typed return contract for Story 3.4's OMP init to consume (`added_tokens`, `base_vocab_size`).
- [x] `packages/tokenizer-core/src/tokenizer_core/adapters/extend.py` -- Implement `extend(family, polish_artifact_dir)`: registry/format/algorithm checks, load target + Polish artifact, JSON-level vocab/merge append, reconstruct via `Tokenizer.from_str`, assert id-preservation invariant, return `AdapterResult` -- delivers AC1/AC2.
- [x] `packages/tokenizer-core/src/tokenizer_core/adapters/__init__.py` -- Re-export `AdapterResult`, `extend` -- keeps `tokenizer_core.adapters` the stable import surface.
- [x] `packages/tokenizer-core/pyproject.toml` -- Add `tokenizers>=0.20` to `adapters` extra -- `extend.py`'s direct `tokenizers` import needs it declared, not just transitively available via `transformers`.
- [x] `pyproject.toml` (root) -- Add `extend.py` (plus `base.py` and `tests/test_extend.py`, discovered necessary during CI-parity verification) to `[[tool.ty.overrides]]` include -- keeps `ty check` green for the new lazy imports whether or not extras are installed.
- [x] `packages/tokenizer-core/tests/test_extend.py` -- Unit-test every I/O matrix row above, plus an AC3 fertility check: `fertility(target_backend, polish_corpus)` vs `fertility(result.tokenizer, polish_corpus)` on a tiny aligned corpus, asserting the extended tokenizer's fertility is strictly lower -- covers AC1-AC3 and the full I/O matrix.

**Acceptance Criteria:**
- Given a `family` registered as `hf_tokenizers` format and a merge-based (`bpe`/`superbpe`) Polish artifact, when `extend(family, polish_artifact_dir)` is called, then the returned `AdapterResult.tokenizer` accepts every target vocab string at its original id and every Polish-only token string at a new id ≥ the target's original vocab size.
- Given the same inputs, when `AdapterResult.added_tokens` is inspected, then it lists exactly the Polish token strings absent from the target's original vocab, in ascending id order, and none of them collide with a pre-existing target token id.
- Given a small Polish-language corpus, when scored via `benchmark.fertility.fertility` on the unextended target tokenizer versus `AdapterResult.tokenizer`, then the extended tokenizer's fertility (tokens/word) is strictly lower.
- Given `packages/tokenizer-core/tests/test_extend.py`, when run via `uv run pytest`, then all tests pass without network access (the target stand-in is a real, tiny, locally-built `tokenizers.Tokenizer`; only `AutoTokenizer.from_pretrained` is faked).

## Spec Change Log

## Review Triage Log

### 2026-07-04 — Review pass
- intent_gap: 0
- bad_spec: 0
- patch: 3 (high 1, medium 1, low 1)
- defer: 2 (medium 1, low 1)
- reject: 12 (low 12)
- addressed_findings:
  - `[high]` `[patch]` `base_vocab_size` was computed from `model.vocab` alone, ignoring the top-level `added_tokens` array where special tokens (bos/eos/pad/...) can live at ids beyond `model.vocab`'s max -- both reviewers independently found a newly-appended Polish token could collide with a pre-existing special token's id. Deeper investigation (reproduced against a real `tokenizers` install) showed the true root cause: `Tokenizer.from_str` does not honor a recorded `added_tokens` id when it leaves a gap above `model.vocab`'s size -- it silently renumbers special tokens to sit immediately after the (grown) model vocab on *every* reload, including future save/reload cycles of the returned tokenizer. Fixed by baking pre-existing `added_tokens` into `model.vocab` at their existing id first (closing the gap, matching how real gpt2 already stores `<|endoftext|>`), then computing `base_vocab_size` from the now-complete `model.vocab` alone. Added `test_extend_special_token_beyond_vocab_max_is_preserved_no_collision`, which also asserts stability across a second round-trip.
  - `[medium]` `[patch]` A `hf_tokenizers`-format target whose model type is Unigram/WordPiece/WordLevel (not BPE) would hit a raw `KeyError` on `model["merges"]` instead of a clean error. Added a `model.type == "BPE"` guard raised as `AdapterError` before any vocab/merge access. Added `test_extend_non_bpe_target_raises_clean_error`.
  - `[low]` `[patch]` `tokenizers.Tokenizer.from_str` was uncaught; a malformed merged JSON would surface a raw exception instead of the documented `AdapterError` contract. Wrapped in `try`/`except` chained `from err`, consistent with every other external call in the function.
  - Also refactored `extend()` into smaller private helpers (`_check_family_format`, `_load_polish_artifact`, `_get_target_backend_tokenizer`, `_bake_added_tokens`, `_append_polish_vocab`, `_append_polish_merges`, `_reconstruct`, `_verify_id_preservation`) after the patches pushed it over `ruff`'s branch/statement-count limits.

## Design Notes

Because merges are appended after the target's own (never reordered or removed), any input the target already tokenizes a certain way keeps tokenizing exactly that way — the target's BPE merge-rank priority is untouched. New Polish merges only fire on byte/token sequences (typically UTF-8 multi-byte diacritics) the target's own merge list never matches, which is exactly the coverage gap this adapter exists to close; it is not a bug that Polish merges are lower-priority.

Both the target's and the Polish candidate's `model.merges` must come from the same `tokenizers` library version for the JSON round-trip (`to_str()`/`from_str()`) to stay format-compatible; both are produced by this repo's own `train` extra install, so this holds today.

`tokenizers.Tokenizer.from_str` does not treat a serialized `added_tokens` entry's `id` field as authoritative when it leaves a gap above `model.vocab`'s size: it silently renumbers such tokens to sit immediately after the model vocab on every reload (verified empirically, not documented behavior). Any pre-existing special token must therefore be baked into `model.vocab` at its existing id *before* appending new tokens and reconstructing -- this is the only way its id survives both this merge and any later save/reload of the result. Story 3.3 (`replace`) should account for this same behavior if it also round-trips through `to_str()`/`from_str()`.

Example vocab-append shape (illustrative, not literal code):

```python
target_vocab = target_data["model"]["vocab"]          # {"Ġthe": 262, ...}
next_id = max(target_vocab.values()) + 1
added = []
for tok, _ in sorted(polish_vocab.items(), key=lambda kv: kv[1]):
    if tok not in target_vocab:
        target_vocab[tok] = next_id
        added.append(tok)
        next_id += 1
```

## Verification

**Commands:**
- `uv run pytest packages/tokenizer-core/tests/test_extend.py -q` -- expected: all new tests pass.
- `task check` -- expected: ruff lint + format check + `ty check` + full pytest suite pass (no regressions).

**Manual checks (if no CLI):**
- Confirm `from tokenizer_core.adapters import extend, AdapterResult` succeeds and `tokenizer_core.adapters.__all__` includes both names.

## Auto Run Result

Status: done

**Summary:** Implemented the extend/merge adapter for Epic 3: `tokenizer_core.adapters.extend.extend(family, polish_artifact_dir)` grafts a trained Polish BPE/SuperBPE candidate onto an `hf_tokenizers`-format target's BPE tokenizer, appending Polish vocab/merges after the target's own so every pre-existing target id (including special/added tokens) is preserved and new Polish tokens get sequential ids starting at `base_vocab_size`. Added the shared `AdapterResult` contract in `adapters/base.py` for Story 3.3 (`replace`) to reuse.

**Files changed:**
- `packages/tokenizer-core/src/tokenizer_core/adapters/base.py` (new) — `AdapterResult` frozen dataclass (`tokenizer`, `added_tokens`, `base_vocab_size`).
- `packages/tokenizer-core/src/tokenizer_core/adapters/extend.py` (new) — `extend()`, decomposed into `_check_family_format`, `_load_polish_artifact`, `_get_target_backend_tokenizer`, `_bake_added_tokens`, `_append_polish_vocab`, `_append_polish_merges`, `_reconstruct`, `_verify_id_preservation`.
- `packages/tokenizer-core/src/tokenizer_core/adapters/__init__.py` — re-exports `AdapterResult`, `extend`.
- `packages/tokenizer-core/tests/test_extend.py` (new) — 12 unit tests covering the full I/O matrix, AC1-AC3, plus two review-driven regression tests (special-token-beyond-vocab-max, non-BPE target).
- `packages/tokenizer-core/pyproject.toml` — added `tokenizers>=0.20` to the `adapters` extra.
- `pyproject.toml` (root) — added `extend.py`, `base.py`, and `tests/test_extend.py` to `[[tool.ty.overrides]]` (all lazily/conditionally reference `tokenizers`).
- `_bmad-output/implementation-artifacts/deferred-work.md` — 2 new deferred entries (pretokenizer/normalizer compatibility check; `Encoder` protocol param-name mismatch in `benchmark/fertility.py`).

**Review findings breakdown** (Blind Hunter + Edge Case Hunter, deduplicated):
- 3 patches applied: `[high]` `base_vocab_size` ignored the top-level `added_tokens` array, and (root cause, found via reproduction against a real `tokenizers` install) `Tokenizer.from_str` silently renumbers `added_tokens` ids that leave a gap above `model.vocab`'s size on every reload — fixed by baking pre-existing added/special tokens into `model.vocab` before computing the id boundary, closing the gap so ids survive both this merge and any future save/reload. `[medium]` a non-BPE `hf_tokenizers`-format target (Unigram/WordPiece/WordLevel) now raises a clean `AdapterError` instead of a raw `KeyError`. `[low]` `Tokenizer.from_str` failures are now wrapped into `AdapterError`. `extend()` was also refactored into 8 small helpers after the patches pushed it over `ruff`'s complexity limits.
- 2 deferred to `deferred-work.md` (1 medium, 1 low): no compatibility check between target/Polish pretokenization schemes; a pre-existing `Encoder` protocol param-name mismatch (`text` vs. `sequence`) in `benchmark/fertility.py` that this story's tests had to route around.
- 12 rejected as noise or already-intentional-by-spec: malformed multi-element merge shapes (documented same-library-version assumption), `validated_today` not gating loads (already resolved as intentional in Story 3.1's review), `import tokenizers` placement (matches existing lazy-import convention), redundant `AdapterError` message text (matches `registry.py` precedent exactly), unjustified `tokenizers>=0.20` floor (matches the `train` extra's identical floor), pathological empty-vocab/non-contiguous-id/sort-tie edge cases (unreachable given this repo's own well-formed trainer output), merge-rank interaction untested beyond docstring (inherent to the documented append-only design), duplicate `REGISTRY.get` lookup (harmless, matches the spec's own fail-fast-before-any-load ordering requirement), and a missing zero-added-tokens test (low value, already covered by the overlap test).

**Verification performed:** `uv run pytest packages/tokenizer-core/tests/test_extend.py -q` (12 passed); full suite `uv run pytest packages -q` under both `uv sync --all-packages --all-extras` (113 passed) and CI-standard `uv sync --all-packages` (95 passed, 7 skipped for the uninstalled `tokenizers` extra); `task check` passes cleanly in the CI-standard (no-extras) sync — the only remaining `ty` diagnostics under a full-extras sync are 4 pre-existing, unrelated issues in `adapters/registry.py`/`tests/test_artifact.py` (confirmed via `git stash` against baseline, present before this story).

**Residual risks:** The two deferred items above; the `added_tokens`-gap behavior discovered here is an unrelease-documented `tokenizers` library quirk (verified empirically against the installed version, not a documented guarantee) that Story 3.3 will also need to account for if it round-trips through `to_str()`/`from_str()`.

Follow-up review recommendation: true — the high-severity fix required a real root-cause investigation (not a mechanical patch) and changed the core merge algorithm's internal structure (added a baking step, a new invariant input, and a full decomposition into 8 helper functions), which is enough behavioral/structural churn from the reviewed version to warrant an independent second look.
