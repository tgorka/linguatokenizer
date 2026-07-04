# Architecture: tokenizer

_Author: BMAD (Architect) · Date: 2026-07-04 · Language: English_

The lean, load-bearing decisions that keep everything built from this repo
consistent. Favors boring, portable technology.

## 1. Principles / invariants

1. **Recipe is the source of truth.** A language's tokenizer is fully described
   by `languages/<iso>/recipe.yaml` (corpora, candidates, targets, baselines).
   Everything else is reproducible from it. Trained binaries are build
   artifacts, not committed.
2. **Fertility is the North Star**, measured on **sentence-aligned** content so
   "Polish costs N× English" is an apples-to-apples statement.
3. **No-GPU core.** Acquisition, training, benchmarking, and training-free
   transplant run CPU-only. GPU is confined to the optional gpt2 retrain.
4. **Quality is gated.** No adapter is "done" until it passes the
   quality-drop budget (≤1%).
5. **Adapters target formats, not model instances.** HF `tokenizers`,
   SentencePiece, and tiktoken cover the target list; a registry maps each
   model to its format so unavailable/future models degrade gracefully.
6. **Data lives on Nextcloud, code lives in git.** `$TOKENIZER_DATA_DIR`.
7. **Lazy, guarded heavy deps.** Base install is light; `tokenizers`,
   `datasets`, `transformers`, `numpy` are extras imported behind guards.

## 2. Component map

```
tokenizer_core/
  corpora/     acquire + sample + mix corpora -> iterable of documents
    sources.py     Source resolver (hf|url|local), streaming, caching to Nextcloud
    sampling.py    Seeded weighted mixing across domains; corpus manifest + hash
  train/       train candidate tokenizers -> portable artifact + metadata
    config.py      Pydantic recipe models (Recipe, Candidate, CorpusSpec, ...)
    trainers.py    BPE / Unigram / SuperBPE trainers over HF `tokenizers`
    artifact.py    Save/load tokenizer.json + metadata sidecar
  benchmark/   measure efficiency + quality
    fertility.py   tokens/word, chars/token, bytes/token  [DONE]
    compare.py     baseline-vs-candidate table over aligned corpora -> results/
    quality.py     regression proxy (bits-per-byte / perplexity parity; PoC hook)
  adapters/    graft a tokenizer onto a model tokenizer
    registry.py    Target-model registry (family, format, vocab, specials)
    base.py        Adapter protocol: extend|replace -> new tokenizer + init plan
    extend.py      Merge Polish vocab into a target tokenizer (preserve base)
    replace.py     Wholesale swap
    embed_omp.py   Training-free OMP embedding init (mean/FOCUS fallbacks)
    gpt2_poc.py    End-to-end transplant (+ optional short retrain) on gpt2
  cli.py       `tokenizer acquire|train|benchmark|adapt|report`
```

## 3. Data flow

```
recipe.yaml
   │  corpora.sources + sampling
   ▼
documents (streamed) ──► train.trainers ──► tokenizer artifact + metadata
   │                                             │
   │  benchmark.compare (aligned PL/EN)          │
   ▼                                             ▼
results/*.json  ◄──────── fertility + quality ◄── candidate + baselines
                                                  │
                                    adapters.(extend|replace) + embed_omp
                                                  │
                                    target model tokenizer + embeddings
                                                  │
                                    benchmark.quality gate (≤1% drop)
```

## 4. Key technical choices

- **Tokenizer engine:** Hugging Face `tokenizers` (Rust) for BPE/Unigram;
  SuperBPE via its two-stage pretokenization-curriculum recipe on top. Export
  `tokenizer.json` (portable) and, where a target needs it, SentencePiece.
- **Recipe validation:** Pydantic v2 models — a malformed recipe fails fast.
- **Benchmark decoupling:** metrics accept *any* `encode`-capable object or
  callable (already true for `fertility`), so baselines and candidates are
  scored by the same code without native-dep coupling.
- **Embedding transplant:** OMP (arXiv:2506.06607) — reconstruct each new
  token's embedding as a sparse combination of shared "anchor" token
  embeddings; no training. Fallbacks: FOCUS-style overlap-weighted mean, then
  plain mean. `numpy` only; `transformers` to read/write embedding matrices.
- **Quality gate:** intrinsic bits-per-byte / perplexity parity on a held-out
  set (CPU-cheap); the gpt2 PoC provides a downstream check.
- **Router (optional):** a thin classifier picking a technical vs general
  tokenizer; measured against a single-tokenizer baseline before adoption.

## 5. Model registry (adapters)

Each target maps to a tokenizer **format** and load path:

| Model (as requested) | Format | Notes |
|----------------------|--------|-------|
| gpt2 | HF BPE (byte-level) | retrain PoC target |
| llama (3.x) | SentencePiece/BPE | gated; HF token |
| mistral 2/3/4 | SentencePiece/BPE | v0.1/0.2 available today; newer versions when released |
| qwen 3.5 | tiktoken/BPE | use current Qwen2.5 tokenizer as stand-in |
| gemma 4 | SentencePiece | current Gemma-2 as stand-in |
| deepseek v4 | BPE | current DeepSeek as stand-in |
| glm 5.2, nemotron 3, kimi k2.7, MiMo 2.5 | BPE/SP | future/less-common; registry entry + format handler, validate when weights exist |

The registry records what's *validated today* vs *format-ready* so coverage is
honest. Adapters are exercised on the models actually available at build time.

## 6. Repo & workspace

uv workspace; one member `packages/tokenizer-core`. ruff/ty/pytest + prek + CI
(see `AGENT.md`). Per-language assets under `languages/<iso>/`. Secrets via
1Password `op.env`. bmad-loop drives dev/review.

## 7. Testing strategy

- Unit tests for metrics (done), recipe parsing, sampling determinism, artifact
  round-trip, registry, and OMP math (small synthetic matrices).
- Golden-file benchmark fixtures (tiny corpora + a toy tokenizer) so the
  comparison table is regression-tested without large data.
- The gpt2 PoC is an integration test behind the `adapters` extra / GPU, not a
  presubmit gate.

## 8. Risks & mitigations

- OMP under-performs on very different vocabularies → keep FOCUS/mean
  fallbacks and the retrain PoC as the escape hatch.
- SuperBPE cross-word tokens can hurt some downstream tasks → gate on quality,
  keep BPE/Unigram winners as defaults if SuperBPE regresses.
- Corpus licensing/quality → manifest + provenance; prefer OSCAR/C4/Wikipedia/
  SpeakLeash Polish sets with clear licenses.
