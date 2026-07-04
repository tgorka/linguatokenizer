# Epic 2 Context: Polish tokenizer training & iteration

<!-- Generated from planning artifacts. Regenerate with compile-epic-context if planning docs change. -->

## Goal

This epic trains and iterates actual Polish tokenizer candidates and converges on a winner. Building on Epic 1's recipe schema and fertility benchmark, it trains BPE, Unigram, and SuperBPE (superword) candidates, sweeps vocab size/algorithm/normalization to drive Polish fertility to at or below the English baseline with no unbudgeted quality regression, and explores whether a technical-vs-general tokenizer router beats a single tokenizer. The outcome is the trained candidate (and recipe) that Epic 3's adapters graft onto real model tokenizers, so getting the fertility/quality trade-off right here is load-bearing for everything downstream.

## Stories

- Story 2.1: BPE and Unigram trainers
- Story 2.2: SuperBPE (superword) trainer
- Story 2.3: Iterate to target fertility
- Story 2.4: Technical-vs-general routing experiment

## Requirements & Constraints

- Train BPE, Unigram, and SuperBPE (superword) candidates at a configurable vocab size, with byte fallback and NFC normalization; all candidates and their hyperparameters are declared in the recipe, and training is one command per language.
- Each trained candidate must emit a portable tokenizer artifact (HF `tokenizers` JSON, or SentencePiece where needed) plus a metadata sidecar recording algorithm, vocab size, corpus hash, and seed — so results are reproducible and traceable to the exact corpus/config that produced them.
- The iteration loop must tune vocab size / algorithm / normalization and keep only the best candidate, gated so it does not regress the quality proxy beyond a defined budget (ties to the ≤1% quality-drop budget used across the project).
- The target to converge on is Polish fertility ≤ English fertility on the same aligned benchmark used in Epic 1; the winning candidate and its results table must be recorded in `languages/pl/README.md`.
- The optional technical-vs-general router is an experiment, not a commitment: train a technical-domain and a general candidate, benchmark the router against a single tokenizer on mixed content, and record a keep-or-drop decision backed by numbers.
- No-GPU constraint: training, benchmarking, and iteration all run CPU-only (GPU is reserved for the Epic 4 gpt2 retrain only).
- Public-ready hygiene applies: English-only committed docs/results; trained tokenizer binaries are never committed — only the recipe and results/metadata are.

## Technical Decisions

- Tokenizer engine: Hugging Face `tokenizers` (Rust) for BPE and Unigram. SuperBPE is layered on top via its two-stage pretokenization-curriculum recipe, not a separate engine.
- Component layout for this epic: `train/trainers.py` (BPE/Unigram/SuperBPE trainers over HF `tokenizers`) and `train/artifact.py` (save/load `tokenizer.json` + metadata sidecar). Both consume `train/config.py`'s Pydantic recipe models from Epic 1.
- Data flow: recipe (corpora + candidate configs) → streamed documents → `train.trainers` → tokenizer artifact + metadata → scored by `benchmark.fertility` / `benchmark.compare` (Epic 1) → results written to `results/*.json`.
- Heavy deps (`tokenizers`, `datasets`, `numpy`) remain optional extras imported lazily behind guards; base install stays light.
- The router (Story 2.4), if kept, is a thin classifier choosing between a technical and a general tokenizer — it is explicitly optional/experimental and must be justified by benchmark numbers before being adopted.
- Known risk to design around: SuperBPE's cross-word tokens can hurt some downstream tasks, so the quality gate must be checked for SuperBPE candidates specifically, and BPE/Unigram should remain the fallback default if SuperBPE regresses quality.

## Cross-Story Dependencies

- Story 2.1 (BPE/Unigram trainers) lands first; Story 2.2 (SuperBPE) and Story 2.3 (iteration sweep) both depend on having trainable candidates and the artifact format from 2.1.
- Story 2.3's sweep runner scores every candidate (from 2.1 and 2.2) using Epic 1's fertility/compare benchmark machinery, and its winner selection is gated by a quality proxy — the same proxy Epic 4 later formalizes as a hard gate for adapters.
- Story 2.4's router experiment is independent and optional; it reuses candidates/benchmark machinery from 2.1–2.3 but does not block the epic's core convergence goal.
- This epic's winning candidate and recipe are the direct input to Epic 3 (adapters graft it onto target model tokenizers) and Epic 4 (gpt2 PoC transplants it end to end).
