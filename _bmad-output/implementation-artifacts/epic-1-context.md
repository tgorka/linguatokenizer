# Epic 1 Context: Foundations — corpora, fertility benchmark, baselines

<!-- Generated from planning artifacts. Regenerate with compile-epic-context if planning docs change. -->

## Goal

This epic builds the measurement spine that every later epic is judged against. It quantifies the Polish token penalty on real, sentence-aligned data — establishing that universal tokenizers over-segment Polish (~1.5–3.2 tokens/word vs ~1.3 for English) because they never learned the language's morphology. Concretely, it delivers the typed recipe layer, corpus acquisition to the off-git data plane, the fertility/compression metrics, and a committed baseline comparison table. Without a trustworthy, reproducible benchmark first, no downstream claim about tokenizer efficiency or quality parity can be validated.

## Stories

- Story 1.1: Recipe schema and loader
- Story 1.2: Corpus acquisition to the Nextcloud data plane
- Story 1.3: Fertility & compression metrics
- Story 1.4: Baseline comparison table

## Requirements & Constraints

- The recipe file is the single source of truth for a language: it fully declares corpora, candidates, targets, and baselines, and everything else must be reproducible from it. Malformed recipes must fail fast with clear errors and reject unknown fields.
- Corpus sources may be Hugging Face, URL, or local; acquisition must stream and cache large data to the off-git data plane (`$TOKENIZER_DATA_DIR`), never bloating the repo or local disk. Sampling is seeded, weighted per domain, and produces a documented corpus manifest with content hashes for reproducibility.
- Gated sources with no available HF token must skip gracefully with a clear message rather than fail hard.
- Fertility (tokens/word) is the North Star metric and must be measured on sentence-aligned PL/EN content (FLORES) so "Polish costs N× English" is a strict apples-to-apples comparison. Also report chars/token and bytes/token.
- Metrics must work over any tokenizer — baselines and trained candidates alike — and results serialize to JSON under `results/` (committed) plus a markdown summary in the language README.
- The baseline table must score gpt2, Llama, Mistral, Qwen, Gemma, and Bielik-v3, and should reproduce the ~1.5–3.2 tokens/word Polish range documented in prior research.
- No-GPU constraint: all of acquisition, benchmarking, and metrics run CPU-only.
- Licensing/provenance: use only permissively licensed, documented corpora (prefer OSCAR / C4 / Wikipedia / SpeakLeash Polish sets); record provenance in the manifest.
- Public-ready hygiene: English-only committed docs and artifacts; presubmit and CI must stay green.

## Technical Decisions

- Recipe models are Pydantic v2 (`Recipe`, `Candidate`, `CorpusSpec`, ...). The loader is exposed as `Recipe.load(path)` and must parse both the Polish recipe and the `_template` recipe.
- Heavy dependencies (`tokenizers`, `datasets`, `transformers`, `numpy`) are optional extras imported lazily behind guards; the base install stays light. Do not import them at module top level in code paths that must run without the extra.
- Benchmark metrics are decoupled from native deps: they accept any `encode`-capable object or callable, so baseline model tokenizers and in-repo candidates are scored by the same code without coupling.
- Component layout for this epic: `corpora/sources.py` (source resolver, streaming, caching), `corpora/sampling.py` (seeded weighted mixing + manifest/hash), `train/config.py` (Pydantic recipe models), `benchmark/fertility.py` (already seeded/done — extend it), and `benchmark/compare.py` (baseline-vs-candidate table → `results/`).
- Data flow: recipe → resolved+sampled documents (streamed) → fertility/compare over aligned PL/EN → JSON in `results/`.
- Determinism is mandatory: seeded sampling and pinned corpus metadata so the same recipe reproduces the same mix and numbers.

## Cross-Story Dependencies

- Stories 1.2 (acquisition) and 1.4 (baseline table) both consume the recipe models from Story 1.1, so the schema/loader should land first.
- Story 1.4's aligned PL/EN comparison builds on the metrics from Story 1.3 (and the already-present `benchmark.fertility`).
- This epic's fertility and quality-proxy machinery is the foundation later epics depend on: Epic 2 uses it to score and iterate trained candidates, and Epics 3–4 reuse the quality proxy to gate adapters.
