# PRD: tokenizer

_Author: BMAD (PM) · Date: 2026-07-04 · Language: English_

Builds on `product-brief.md`. Defines functional/non-functional requirements
and the epic breakdown for a token-efficient Polish tokenizer plus model
adapters, structured for multi-language extension.

## 1. Product summary

A recipe-driven toolkit that (a) trains and benchmarks morphology-aware
tokenizers for a target language, (b) proves efficiency gains via a fertility
benchmark against baseline model tokenizers, and (c) adapts a chosen tokenizer
onto existing LLM tokenizers with quality-preserving embedding initialization —
without full model retraining where possible.

## 2. Users & jobs-to-be-done

- *As a model builder*, I want a Polish tokenizer that halves token counts so
  my Polish serving cost and latency drop and context effectively doubles.
- *As a platform operator*, I want evidence (a benchmark) that the new
  tokenizer preserves quality before I adopt it.
- *As a researcher*, I want a reproducible pipeline I can point at a new
  language.

## 3. Functional requirements

### FR1 — Corpus acquisition & preparation
- FR1.1 Load training/eval corpora declared in `languages/<iso>/recipe.yaml`
  from Hugging Face / URL / local, with sampling weights per domain.
- FR1.2 Stream and cache large corpora to the Nextcloud data plane
  (`$TOKENIZER_DATA_DIR`), never bloating the repo or local disk.
- FR1.3 Provide sentence-aligned PL/EN eval data (FLORES) for fair,
  same-content fertility comparison.
- FR1.4 Deterministic sampling (seeded) and a documented corpus manifest.

### FR2 — Tokenizer training
- FR2.1 Train BPE, Unigram, and SuperBPE (superword) candidates at a
  configurable vocab size with byte fallback and NFC normalization.
- FR2.2 Emit a portable tokenizer artifact (HF `tokenizers` JSON /
  SentencePiece) plus a metadata sidecar (algo, vocab, corpus hash, seed).
- FR2.3 Candidates and hyperparameters are declared in the recipe; training is
  one command per language.

### FR3 — Benchmarking
- FR3.1 Fertility (tokens/word), chars/token, and bytes/token over any corpus
  for any tokenizer (baselines + candidates), producing a comparison table.
- FR3.2 Compare against declared baselines (gpt2, Llama, Mistral, Qwen, Gemma,
  Bielik-v3, …) on identical content; report Polish-vs-English ratios.
- FR3.3 A downstream **quality** proxy that detects regression when a tokenizer
  is transplanted (see FR5) — measured on the gpt2 PoC and/or intrinsic
  metrics (e.g. bits-per-byte / perplexity parity).
- FR3.4 Results written to `results/` (committed) and summarized in the
  language README.

### FR4 — Iteration
- FR4.1 Support multiple candidates and an optional **router** (e.g. technical
  vs general-language tokenizer) and measure whether it helps.
- FR4.2 An iteration loop that tunes vocab size / algorithm / normalization and
  keeps the best candidate **without quality regression** (guarded by a gate).

### FR5 — Model adapters / compilers
- FR5.1 A **model registry** describing each target model's tokenizer family
  (format, vocab, special tokens): mistral 2/3/4, llama, glm 5.2, nemotron 3,
  qwen 3.5, kimi k2.7, deepseek v4, MiMo 2.5, gpt2, gemma 4.
- FR5.2 **Extend/merge** adapter: inject the Polish vocabulary into a target
  tokenizer, preserving existing tokens (compiler that *extends* the default).
- FR5.3 **Replace** adapter: swap in the Polish tokenizer wholesale.
- FR5.4 **Training-free embedding init** (OMP; mean/FOCUS fallbacks) so new
  tokens get sensible embeddings without retraining.
- FR5.5 gpt2 **retrain proof-of-concept**: when training-free is insufficient,
  demonstrate the full path (transplant + short retrain) end to end.

### FR6 — Reproducibility & extensibility
- FR6.1 Everything driven by `recipe.yaml`; adding a language = copy template.
- FR6.2 CLI (`tokenizer …`) exposing acquire / train / benchmark / adapt.
- FR6.3 Deterministic, seeded, with pinned corpus + tokenizer metadata.

## 4. Non-functional requirements

- NFR1 **No-GPU core**: acquisition, training, benchmarking, and OMP transplant
  run CPU-only. GPU only for the optional gpt2 retrain (remote Hyperstack).
- NFR2 **Quality gate**: adapters must pass a ≤1% downstream-quality-drop check
  before being marked done.
- NFR3 **Licensing**: only permissively licensed / documented corpora; record
  provenance.
- NFR4 **Data hygiene**: large data on Nextcloud, not git; repo stays small.
- NFR5 **Public-ready**: English-only docs/artifacts; presubmit + CI green.
- NFR6 **Portability**: adapters target tokenizer *formats*, degrading
  gracefully when a specific model is unavailable.

## 5. Epics

1. **Epic 1 — Foundations: corpora, fertility benchmark, baselines.** Data
   acquisition to Nextcloud, the fertility/quality metrics, and a baseline
   comparison table quantifying the Polish penalty on real data.
2. **Epic 2 — Polish tokenizer training & iteration.** Train BPE/Unigram/
   SuperBPE candidates, benchmark, iterate to Polish ≤ English fertility, pick
   a winner; explore technical-vs-general routing.
3. **Epic 3 — Model adapters & compilers.** Registry + extend/merge + replace
   adapters with training-free OMP embedding init; validate on ≥5 real models.
4. **Epic 4 — Quality validation & gpt2 PoC.** Prove no quality regression;
   gpt2 transplant (+short retrain) as the end-to-end proof-of-concept.
5. **Epic 5 — Extensibility & release.** Solidify the multi-language pipeline,
   docs, and the "add a language" path; stage next languages as backlog.

## 6. Out of scope / next steps

Other languages (fr, zh, cs, ru, uk, lt, es, de, hi, he, ar, th, pt) — each a
future epic reusing the Epic 1–3 machinery. Production-scale model retraining
and serving are out of scope.

## 7. Acceptance (product-level)

- A committed benchmark table showing a trained Polish tokenizer at fertility
  ≤ the English baseline on aligned content.
- At least one adapter demonstrating quality parity (≤1% drop) on the gpt2 PoC.
- A documented, one-command path to reproduce, and to add a new language.
