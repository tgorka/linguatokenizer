# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

First end-to-end milestone: a token-efficient, morphology-aware Polish
tokenizer, adapters that graft it onto existing model tokenizers, the same
pipeline generalized to 13 more languages, and two product surfaces
(datacenter cost optimizer, ei-models penalty check).

### Added

- **Core library (`tokenizer_core`).** Corpus staging, tokenizer training
  (byte-level BPE, Unigram, SuperBPE), and a fertility/quality benchmark suite.
  First-party import root `tokenizer_core`; `tokenizer` CLI.
- **Polish tokenizer.** Recipe-driven BPE at 32k/48k/64k reaching **1.56 /
  1.48 / 1.42 tokens/word** (≈1.14× English at 64k) — beats every universal
  baseline and matches the Polish-tuned Bielik v3 reference. SuperBPE
  (cross-word) explored as the path below parity.
- **Adapters.** `extend` (graft the Polish vocabulary onto a model's tokenizer,
  base ids preserved — English stays 99.8% token-identical) and `replace`
  (wholesale swap to the standalone tokenizer). Training-free **FVT** embedding
  transplant (`adapters.embed`), deployable bundle export (`adapters.bundle`),
  and a model registry.
- **gpt2 retrain PoC (`scripts/gpt2_retrain.py`).** Optional short fine-tune of
  only the transplanted rows (base frozen → English preserved), runnable on
  **CPU**; recovers ~78% of the transplant's bits-per-byte penalty while keeping
  −23.5% Polish tokens and passing the English quality gate.
- **Quality gate (`benchmark.quality`).** Bits-per-byte and model-free
  base-language-preservation checks so adapter changes can be gated on no
  regression.
- **Multi-language rollout.** The pipeline generalized to any FLORES-200 script
  and applied to **14 languages across 7 scripts** (pl, de, ru, fr, es, pt, cs,
  uk, lt, zh, hi, he, ar, th). Most reach ~1.0–1.3× English (Hebrew, Chinese,
  Thai below parity), cutting tokens 23–77%. Per-language recipes under
  `languages/<iso>/`.
- **Domain router (`tokenizer_core.routing`).** Technical-vs-general routing
  heuristic + experiment; measured **drop** decision (a single mixed tokenizer
  beat the routed pair).
- **Products.** `tokenizer_core.economics` (datacenter inference cost /
  throughput / context / KV-memory savings, all linear in fertility) and
  `tokenizer_core.integrations.ei_models` (portable Polish/English penalty
  report for the ei-models repo).
- **Benchmarks (`results/`).** Baselines, tokenizer candidates, adapters,
  economics, ei-models penalty, multi-language penalty landscape + our results,
  cross-model old-vs-new token counts and encode timing, routing experiment,
  gpt2 transplant + retrain. Consolidated in `docs/benchmark-report.md`.
- **Docs.** `docs/usage.md`, `docs/add-a-language.md`,
  `docs/datacenter-optimizer.md`, `docs/ei-models-integration.md`,
  `docs/benchmark-report.md`, and `docs/data-provenance.md` (corpus sources and
  licenses).
- **Project infrastructure.** uv workspace, Taskfile, ruff (Google style) +
  ty + pytest gate, prek pre-commit, 1Password-backed secrets, and a remote
  GPU run harness. Managed with BMAD (`_bmad-output/`).

### Notes

- Corpora and large build artifacts (trained tokenizers, transplanted models,
  bundles) live on the Nextcloud data plane (`$TOKENIZER_DATA_DIR`), not in git.
  Recipes are the committed source of truth. See `docs/data-provenance.md`.
- Dataset licenses differ from this repository's code; verify a corpus's terms
  before redistributing any trained artifact derived from it.

[Unreleased]: https://github.com/Neuraffica/tokenizer/commits/main
