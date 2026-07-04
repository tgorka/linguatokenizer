# Epics & Stories: tokenizer

_Author: BMAD · Date: 2026-07-04 · Language: English_

Derived from `PRD.md` and `architecture.md`. Story keys follow BMAD sprint
convention `epic-story-title` (kebab-case).

---

## Epic 1: Foundations — corpora, fertility benchmark, baselines

Quantify the Polish token penalty on real data and build the measurement
spine everything else is judged against.

### Story 1.1: Recipe schema and loader

As a developer, I want typed recipe models, so that a language's corpora,
candidates, and targets are validated and reproducible.

Acceptance Criteria:
1. Pydantic models parse `languages/pl/recipe.yaml` and `_template/recipe.yaml`.
2. Invalid recipes fail fast with clear errors; unknown fields rejected.
3. `tokenizer_core.train.config` exposes `Recipe.load(path)`; unit-tested.

### Story 1.2: Corpus acquisition to the Nextcloud data plane

As a developer, I want to fetch declared corpora to `$TOKENIZER_DATA_DIR`, so
that large data stays off git and local disk.

Acceptance Criteria:
1. `corpora.sources` resolves hf/url/local specs and streams to the data plane.
2. Seeded, weighted sampling across domains; a corpus manifest with hashes.
3. Graceful skip + clear message when a source is gated and no HF token.
4. Unit tests use tiny local fixtures (no network in CI).

### Story 1.3: Fertility & compression metrics

As a developer, I want tokens/word, chars/token, bytes/token over any
tokenizer, so that efficiency is measurable. (Seeded — extend the existing
`benchmark.fertility`.)

Acceptance Criteria:
1. Metrics accept any `encode`-capable object or callable. [done]
2. Aligned-corpus PL-vs-EN ratio helper added; unit-tested.
3. Results serialize to JSON for `results/`.

### Story 1.4: Baseline comparison table

As a model builder, I want a table of Polish vs English fertility across
baseline tokenizers, so that the penalty is documented on real data.

Acceptance Criteria:
1. `benchmark.compare` scores gpt2/Llama/Mistral/Qwen/Gemma/Bielik-v3 on
   aligned FLORES PL/EN.
2. Emits a committed `results/baselines-pl-en.json` + markdown summary.
3. Reproduces the ~1.5–3.2 tokens/word Polish range from the research.

---

## Epic 2: Polish tokenizer training & iteration

Train and iterate candidates to Polish ≤ English fertility without quality
loss.

### Story 2.1: BPE and Unigram trainers

As a developer, I want to train BPE and Unigram candidates from a recipe, so
that I have baseline candidates to benchmark.

Acceptance Criteria:
1. `train.trainers` trains both at configurable vocab size, byte fallback, NFC.
2. Emits portable `tokenizer.json` + metadata sidecar (algo, vocab, corpus
   hash, seed).
3. Round-trip load/encode tested on a tiny corpus.

### Story 2.2: SuperBPE (superword) trainer

As a developer, I want a SuperBPE candidate, so that cross-word tokens can push
fertility lower.

Acceptance Criteria:
1. Two-stage pretokenization-curriculum training implemented.
2. Produces a valid candidate scored by the same benchmark.
3. Documented trade-offs vs BPE/Unigram.

### Story 2.3: Iterate to target fertility

As a developer, I want an iteration loop over vocab size/algorithm/
normalization, so that I converge on Polish ≤ English fertility.

Acceptance Criteria:
1. Sweep runner records each candidate's fertility + quality proxy to results/.
2. Best candidate meets `targets.max_fertility_vs_english` with no quality-
   proxy regression beyond budget.
3. Winner recorded in `languages/pl/README.md` with the results table.

### Story 2.4: Technical-vs-general routing experiment

As a researcher, I want to test a technical/general tokenizer router, so that
we know whether routing beats a single tokenizer.

Acceptance Criteria:
1. Train a technical-domain and a general candidate; a thin router selects one.
2. Benchmark router vs single-tokenizer on mixed content.
3. Keep-or-drop decision documented with numbers.

---

## Epic 3: Model adapters & compilers

Graft the Polish tokenizer onto existing model tokenizers, training-free where
possible.

### Story 3.1: Target-model registry

As a developer, I want a registry of target models' tokenizer formats, so that
adapters degrade gracefully across available and future models.

Acceptance Criteria:
1. Registry entries for mistral/llama/qwen/gemma/deepseek/gpt2 (+ glm/nemotron/
   kimi/MiMo as format-ready).
2. Each entry records format, vocab, special tokens, and validated-today flag.
3. Loading a target tokenizer is one call; unit-tested with a local stand-in.

### Story 3.2: Extend/merge adapter (compiler)

As a model builder, I want to inject the Polish vocabulary into a target
tokenizer preserving its tokens, so that I extend rather than replace.

Acceptance Criteria:
1. `adapters.extend` merges Polish tokens/merges into a target tokenizer.
2. Existing token ids preserved; new tokens appended; round-trip verified.
3. Fertility on Polish improves vs the unextended target.

### Story 3.3: Replace adapter

As a model builder, I want to swap in the Polish tokenizer wholesale, so that I
get maximum efficiency where a full swap is acceptable.

Acceptance Criteria:
1. `adapters.replace` produces a model-compatible tokenizer + init plan.
2. Special tokens / chat template preserved or re-mapped.
3. Documented when to prefer replace vs extend.

### Story 3.4: Training-free OMP embedding init

As a model builder, I want new-token embeddings initialized without retraining,
so that I can adopt the tokenizer cheaply.

Acceptance Criteria:
1. `adapters.embed_omp` reconstructs new embeddings from shared anchors (OMP),
   with FOCUS-mean and plain-mean fallbacks.
2. OMP math unit-tested on small synthetic matrices.
3. Produces an updated embedding matrix consumable by `transformers`.

---

## Epic 4: Quality validation & gpt2 PoC

Prove no quality regression; demonstrate the full path end to end.

### Story 4.1: Quality regression gate

As a developer, I want a CPU-cheap quality proxy, so that adapters are gated on
≤1% drop.

Acceptance Criteria:
1. `benchmark.quality` computes bits-per-byte / perplexity parity on held-out
   text for a (model, tokenizer) pair.
2. A gate function returns pass/fail against the budget; unit-tested.

### Story 4.2: gpt2 transplant proof-of-concept

As a researcher, I want an end-to-end gpt2 transplant (+ optional short
retrain), so that the no-retrain claim is validated and the retrain path shown.

Acceptance Criteria:
1. Extend/replace + OMP applied to gpt2; quality gate measured before/after.
2. If training-free is insufficient, a short retrain (remote GPU) recovers
   quality; results recorded.
3. Reproducible via `task` / CLI; artifacts to Nextcloud, summary to results/.

---

## Epic 5: Extensibility & release

Solidify the multi-language pipeline and prepare for public release.

### Story 5.1: Add-a-language pipeline hardening

As a maintainer, I want the `_template` path validated end to end, so that a
new language is a copy-and-fill.

Acceptance Criteria:
1. A second language stub builds through acquire→train→benchmark on tiny data.
2. Docs describe the exact steps; CLI supports `--lang <iso>`.

### Story 5.2: Release readiness

As a maintainer, I want docs, licenses, and CI solid, so that the repo can go
public.

Acceptance Criteria:
1. README/AGENT/CLAUDE accurate; LICENSE + corpus provenance documented.
2. CI green; next-language backlog (fr, zh, cs, ru, uk, lt, es, de, hi, he, ar,
   th, pt) listed as epics.

---

## Epic 6: Product packages — datacenter optimizer + ei-models integration

Turn the tokenizer + adapters into things a user can actually consume: a
datacenter inference-cost optimizer and a drop-in for the ei-models experiments
to verify the tokenizer penalty is gone.

### Story 6.1: Inference-cost economics

As a datacenter operator, I want to quantify the cost/throughput/context wins of
a lower-fertility tokenizer, so that I can decide whether to adopt it.

Acceptance Criteria:
1. `tokenizer_core.economics` turns (baseline fertility, optimized fertility,
   workload, pricing) into token-reduction %, $ savings, throughput gain, and
   effective-context multiplier, grounded in cited token economics.
2. A committed Polish savings report + CLI `tokenizer economics`.
3. Pure-python, unit-tested; no model required.

### Story 6.2: Adapt bundle export

As a serving engineer, I want a single portable artifact bundle, so that I can
load the adapted tokenizer + embeddings into a serving stack.

Acceptance Criteria:
1. `export_bundle` writes tokenizer.json + the transplanted embedding matrix
   (safetensors) + a manifest (base model, method, vocab boundary, fertility).
2. Round-trip loadable; documented layout. Large matrices go to the data plane.

### Story 6.3: ei-models integration

As an ei-models researcher, I want a portable penalty report, so that I can
re-run the PL-vs-EN token spike with our tokenizer and confirm the ~2× penalty
is removed.

Acceptance Criteria:
1. `tokenizer_core.integrations.ei_models` reproduces the fertility/penalty
   report for any tokenizer over aligned pairs, matching ei-models' framing.
2. A standalone script + usage doc the ei-models repo can call.
3. A committed before/after report: baseline vs our tokenizer on Polish.
