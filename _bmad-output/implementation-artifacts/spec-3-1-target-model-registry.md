---
title: 'Target-model registry'
type: 'feature'
created: '2026-07-04'
status: 'done'
review_loop_iteration: 0
followup_review_recommended: false
context: []
warnings: ['oversized']
baseline_revision: 'ae7a23aeee57e64323e31a611150e5a647777c24'
final_revision: 'b5d0685'
---

<intent-contract>

## Intent

**Problem:** Epic 3's adapters (extend/replace/OMP embedding init) need to know each target model's tokenizer format, vocab size, and special tokens before they can graft the Polish tokenizer onto it — today `tokenizer_core.adapters` is an empty placeholder package with no such lookup.

**Approach:** Add a static, dependency-light target-model registry (`tokenizer_core.adapters.registry`) recording format/vocab/special-tokens/validated-today per model family, plus a single-call loader that resolves a family to a live tokenizer object via the existing lazy-import-behind-guard convention.

## Boundaries & Constraints

**Always:**
- `REGISTRY` is a static, module-level `dict[str, TargetModelEntry]` keyed by family name — no I/O or network access at import time.
- `TargetModelEntry` is a Pydantic v2 `BaseModel` with `model_config = ConfigDict(extra="forbid")`, matching the style in `train/config.py`.
- Registry entries exist for exactly these families: `gpt2`, `llama`, `mistral`, `qwen`, `gemma`, `deepseek` (each `validated_today=True`), and `glm`, `nemotron`, `kimi`, `mimo` (each `validated_today=False`, i.e. format-ready).
- `load_target_tokenizer(family)` imports `transformers` lazily inside the function body (`# noqa: PLC0415`), mirroring `benchmark/compare.py::_load_baseline_encoder` — the `adapters` extra already depends on `transformers`.
- Failures raise `AdapterError(ValueError)` with the family name and underlying cause in the message, chained via `from err` — mirroring `RecipeError`'s style in `train/config.py`.
- `packages/tokenizer-core/src/tokenizer_core/adapters/__init__.py` re-exports the public registry API (`TargetModelEntry`, `TokenizerFormat`, `REGISTRY`, `AdapterError`, `load_target_tokenizer`) so `tokenizer_core.adapters` is the stable import surface for Stories 3.2–3.4.
- Tests use an in-file fake `transformers` module injected via `monkeypatch.setitem(sys.modules, "transformers", ...)`, following `tests/test_compare.py`'s pattern — no network, no on-disk fixture files.

**Block If:** none identified — this story only adds new, self-contained metadata and a loader function; it does not touch `train/config.py`'s `Recipe` schema or any existing module's behavior.

**Never:**
- Never implement the extend, replace, or OMP embedding-init adapters themselves (Stories 3.2–3.4) — this story is registry + loader only.
- Never make `load_target_tokenizer` silently swallow failures or return `None`/a skip-reason tuple — that graceful-skip pattern belongs to `benchmark.compare` (which scores many baselines tolerantly); this registry's single-target loader raises so callers can decide how to handle an unavailable target.
- Never fetch data over the network in tests — all tokenizer loading in tests goes through a fake `transformers` stand-in.
- Never invent a precise `vocab_size` or `special_tokens` value for a format-ready family without a real public precedent to anchor it (e.g. use the most recent real released model in that family); if genuinely unknown, leave the field `None` rather than fabricating a number.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Known, validated family loads fine | `family="gpt2"`, fake `transformers.AutoTokenizer.from_pretrained` returns a stand-in tokenizer | `load_target_tokenizer("gpt2")` returns that tokenizer object | No error expected |
| Unknown family | `family="falcon"` (not a `REGISTRY` key) | Raises before any import/network attempt | `AdapterError` naming the unknown family |
| Registered family, underlying load fails | `family="gemma"`, fake loader raises `RuntimeError` (simulating gated repo / network) | No tokenizer returned | `AdapterError` chaining the original exception (`from err`), message includes family and `hf_id` |
| Format-ready family, no checkpoint available | `family="kimi"` (`validated_today=False`), fake loader raises | Same as any other family — no special-cased skip logic in this story | `AdapterError` chaining the original exception |
| Registry entry validation | Constructing a `TargetModelEntry` with an unrecognized field | Rejected at construction | Pydantic `ValidationError` (via `extra="forbid"`) |

</intent-contract>

## Code Map

- `packages/tokenizer-core/src/tokenizer_core/adapters/registry.py` -- NEW: `TokenizerFormat` literal, `TargetModelEntry` model, `AdapterError`, `REGISTRY` constant, `load_target_tokenizer()`.
- `packages/tokenizer-core/src/tokenizer_core/adapters/__init__.py` -- currently a one-line placeholder docstring; replace with re-exports of the registry's public API.
- `packages/tokenizer-core/tests/test_registry.py` -- NEW: unit tests using a fake `transformers` module (pattern from `tests/test_compare.py`).
- `packages/tokenizer-core/src/tokenizer_core/train/config.py` -- reference only (no changes): `RecipeError`/`Recipe.load` is the error-handling and Pydantic-style precedent to follow.
- `packages/tokenizer-core/src/tokenizer_core/benchmark/compare.py` -- reference only (no changes): `_load_baseline_encoder` is the lazy-import-behind-guard precedent to follow.
- `packages/tokenizer-core/pyproject.toml` -- reference only (no changes expected): `adapters` extra already includes `transformers`.

## Tasks & Acceptance

**Execution:**
- [x] `packages/tokenizer-core/src/tokenizer_core/adapters/registry.py` -- Define `TokenizerFormat = Literal["hf_tokenizers", "sentencepiece", "tiktoken"]`, `class AdapterError(ValueError)`, `class TargetModelEntry(BaseModel)` (`model_config = ConfigDict(extra="forbid")`; fields `family: str`, `hf_id: str`, `format: TokenizerFormat`, `vocab_size: int | None`, `special_tokens: dict[str, str] | None`, `validated_today: bool`), and the `REGISTRY: dict[str, TargetModelEntry]` constant covering `gpt2`, `llama`, `mistral`, `qwen`, `gemma`, `deepseek` (`validated_today=True`) and `glm`, `nemotron`, `kimi`, `mimo` (`validated_today=False`) -- gives adapters a single typed source of target-model metadata (AC1, AC2).
- [x] `packages/tokenizer-core/src/tokenizer_core/adapters/registry.py` -- Add `def load_target_tokenizer(family: str) -> Encoder` that looks up `family` in `REGISTRY` (raising `AdapterError` for an unknown family before any import), then lazily imports `transformers` and calls `transformers.AutoTokenizer.from_pretrained(entry.hf_id, token=os.environ.get("HF_TOKEN"))`, wrapping any exception into `AdapterError` chained with `from err` -- gives adapters (Stories 3.2/3.3) a one-call way to get a live target tokenizer (AC3).
- [x] `packages/tokenizer-core/src/tokenizer_core/adapters/__init__.py` -- Replace the placeholder docstring with re-exports: `TokenizerFormat`, `TargetModelEntry`, `AdapterError`, `REGISTRY`, `load_target_tokenizer` -- makes `tokenizer_core.adapters` the stable import surface for downstream stories.
- [x] `packages/tokenizer-core/tests/test_registry.py` -- Unit-test: (1) `REGISTRY` has exactly the 10 expected family keys with the expected `validated_today` split; (2) `TargetModelEntry(**data, unknown_field=1)` raises `ValidationError`; (3) `load_target_tokenizer` success path via a fake `transformers` module (mirroring `test_compare.py`'s `_fake_transformers_module`); (4) `load_target_tokenizer("falcon")` raises `AdapterError` without touching `transformers`; (5) `load_target_tokenizer` on a registered family whose fake loader raises surfaces `AdapterError` with the original exception chained -- covers the I/O matrix and AC3's "unit-tested with a local stand-in".

**Acceptance Criteria:**
- Given the `REGISTRY` constant, when inspected, then it contains an entry for each of `gpt2`, `llama`, `mistral`, `qwen`, `gemma`, `deepseek`, `glm`, `nemotron`, `kimi`, `mimo`, with the first six `validated_today=True` and the last four `validated_today=False`.
- Given any `REGISTRY` entry, when read, then it exposes `format`, `vocab_size` (or `None`), `special_tokens` (or `None`), and `validated_today`.
- Given a family present in `REGISTRY` and a working (possibly fake, in tests) `transformers.AutoTokenizer`, when `load_target_tokenizer(family)` is called, then it returns a usable tokenizer object via a single function call — no multi-step setup required by the caller.
- Given a family absent from `REGISTRY`, when `load_target_tokenizer(family)` is called, then it raises `AdapterError` naming the unknown family, without attempting to import `transformers`.
- Given `packages/tokenizer-core/tests/test_registry.py`, when run via `uv run pytest`, then all tests pass without network access.

## Spec Change Log

## Review Triage Log

### 2026-07-04 — Review pass
- intent_gap: 0
- bad_spec: 0
- patch: 2 (medium 2, low 0)
- defer: 3 (medium 1, low 2)
- reject: 10 (low 10)
- addressed_findings:
  - `[medium]` `[patch]` `load_target_tokenizer` left `import transformers` outside the `try`/`except`, so a missing `adapters` extra raised a raw `ImportError`/`ModuleNotFoundError` instead of the documented `AdapterError` — wrapped the import in its own `try`/`except ImportError` and added `test_load_target_tokenizer_missing_extra_raises_adapter_error`.
  - `[medium]` `[patch]` No test asserted each `REGISTRY` entry's `.family` matches its own dict key, so a copy-paste typo diverging the two would go undetected — added `test_registry_entry_family_matches_its_own_dict_key`.

## Design Notes

`REGISTRY` is intentionally independent of any `languages/<iso>/recipe.yaml` — it is cross-language, cross-recipe model metadata (which tokenizer format a family uses), whereas a recipe's `baselines: list[str]` names specific HF checkpoints to benchmark against for one language. The two lists overlap in spirit but serve different callers; this story does not merge them.

For the four format-ready families (`glm`, `nemotron`, `kimi`, `mimo`), anchor `hf_id`/`vocab_size`/`special_tokens` to the most recent real, publicly documented checkpoint in that family (e.g. an existing GLM-4, Nemotron, Kimi-K2, or MiMo release) even though the specific future version named in planning docs may not exist yet — `validated_today=False` communicates that distinction, not the metadata's absence. Use `WebSearch`/`WebFetch` to confirm current public vocab sizes/special tokens if uncertain; do not guess.

Example shape for one entry:

```python
"gpt2": TargetModelEntry(
    family="gpt2",
    hf_id="gpt2",
    format="hf_tokenizers",
    vocab_size=50257,
    special_tokens={"bos": "<|endoftext|>", "eos": "<|endoftext|>"},
    validated_today=True,
),
```

## Verification

**Commands:**
- `uv run pytest packages/tokenizer-core/tests/test_registry.py -q` -- expected: all new tests pass.
- `task check` -- expected: ruff lint + format check + ty + full pytest suite all pass (no regressions in existing tests).

**Manual checks (if no CLI):**
- Confirm `packages/tokenizer-core/src/tokenizer_core/adapters/__init__.py` no longer contains only the placeholder docstring, and that `from tokenizer_core.adapters import load_target_tokenizer, REGISTRY` succeeds.

## Auto Run Result

Status: done

**Summary:** Implemented the target-model registry for Epic 3's adapters: a static `REGISTRY` of 10 model families (gpt2/llama/mistral/qwen/gemma/deepseek validated-today; glm/nemotron/kimi/mimo format-ready) with format/vocab/special-tokens metadata, plus `load_target_tokenizer()`, a single-call loader following the codebase's existing lazy-import-behind-guard and chained-exception conventions.

**Files changed:**
- `packages/tokenizer-core/src/tokenizer_core/adapters/registry.py` (new) — `TokenizerFormat`, `AdapterError`, `TargetModelEntry`, `REGISTRY`, `load_target_tokenizer()`.
- `packages/tokenizer-core/src/tokenizer_core/adapters/__init__.py` — replaced placeholder docstring with re-exports of the registry's public API.
- `packages/tokenizer-core/tests/test_registry.py` (new) — 10 unit tests covering registry shape, validation, and the loader's success/unknown-family/load-failure/missing-extra paths, all via an in-`sys.modules` fake `transformers` stand-in (no network).
- `pyproject.toml` (root) — added `adapters/registry.py` to the existing `[[tool.ty.overrides]]` list that suppresses `unresolved-import` for files that lazily import optional-extra deps.
- `_bmad-output/implementation-artifacts/epic-3-context.md` (new) — compiled Epic 3 planning context (goal, stories, requirements, technical decisions, cross-story dependencies).

**Review findings breakdown** (Blind Hunter + Edge Case Hunter, deduplicated):
- 2 patches applied (both medium): `import transformers` now wrapped in its own `try/except ImportError` so a missing `adapters` extra raises the documented `AdapterError` instead of a raw `ModuleNotFoundError`; added a test asserting every `REGISTRY` entry's `.family` matches its own dict key.
- 3 deferred to `deferred-work.md` (1 medium, 2 low): `HF_TOKEN`-only/empty-string token handling, missing `HUGGINGFACE_HUB_TOKEN` fallback, and unredacted exception text in `AdapterError` messages — all pre-existing patterns this story mirrored from `benchmark/compare.py`/`train/config.py` rather than introduced.
- 10 rejected as noise or already-intentional-by-spec: case/whitespace family-key normalization, empty-string family input (already behaves correctly), `BaseException` handling (correct as-is), exhaustive per-family loader tests beyond the spec's I/O matrix, unpinned/undated fact-verification claims, gpt2's shared bos/eos token (standard community convention), `validated_today` being advisory-only (explicit spec design), epic's aspirational future-version naming vs. registry's real anchors (already reconciled in Design Notes), `Encoder` Protocol not runtime-checked (standard structural typing), and the `ty` override's file-wide scope (matches established precedent exactly).

**Verification performed:** `uv run pytest packages/tokenizer-core/tests/test_registry.py -q` (10 passed); `task check` — ruff check, ruff format --check, `ty check`, full `uv run pytest` (95 passed, 6 pre-existing skips for the uninstalled `tokenizers` extra, unrelated to this story).

**Residual risks:** The four format-ready families' vocab/special-token facts are anchored to real, currently-public checkpoints but are inherently perishable (no version pin or last-verified date recorded); the three deferred HF-token/error-message findings apply codebase-wide, not just to this new module.

Follow-up review recommendation: false — the two patches were small, mechanical, and fully localized to the new module; no behavioral surface beyond this story's own scope was touched.
