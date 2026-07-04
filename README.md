# tokenizer

Token-efficient, morphology-aware tokenizers for under-served languages —
**Polish first** — plus adapters that graft them onto existing LLM tokenizers.

## Why

Research in [`Neuraffica/ei-models`](https://github.com/Neuraffica/ei-models)
found that across every model measured, **Polish costs ~2.05× the tokens of
English** for the same task. About three-quarters of that penalty is the
*tokenizer*: universal tokenizers (Mistral, Llama, …) never learned Polish
morphology, so they over-segment it — **~1.5–3.2 tokens/word for Polish vs
~1.3 for English**. That inflates cost, latency, and shrinks the effective
context window. Bielik v3 proved the fix works: a Polish-tuned tokenizer cut
fertility from **3.22 → 1.62** tokens/word (arXiv:2604.10799).

## Goal

A Polish tokenizer whose **fertility approaches or beats English**, with **no
downstream quality regression** — ideally achieved **without retraining** the
target model (via training-free embedding transplantation), and reproducible
for other languages.

## What's here

| Path | Purpose |
|------|---------|
| `packages/tokenizer-core/` | Core library: `corpora`, `train`, `benchmark`, `adapters`. |
| `languages/<iso>/recipe.yaml` | Per-language source of truth (corpora, candidates, targets). Polish is `languages/pl/`. |
| `languages/_template/` | Copy this to add a new language. |
| `results/` | Benchmark artifacts (fertility tables, quality parity) — committed. |
| `data/` | Local scratch; real corpora + large artifacts live on the Nextcloud data store (`$TOKENIZER_DATA_DIR`). |
| `_bmad-output/` | BMAD planning + implementation artifacts. |

The **North Star metric** is *fertility* (tokens/word) on held-out, sentence-
aligned text — see `tokenizer_core.benchmark.fertility`.

## Results so far (Polish, measured on FLORES-200)

Fertility = tokens per word on 1012 sentence-aligned PL/EN pairs. Lower is
better; the baselines show the ~2× Polish tax, our tokenizers cut it — **with
no model retraining**.

| Tokenizer | PL tok/word | PL vs EN |
|-----------|------------:|---------:|
| gpt2 (baseline) | 3.71 | 3.02× |
| Mistral-7B / Bielik-v2.3 (baseline) | 2.97 | 2.11× |
| Llama-3.1-8B (baseline) | 2.68 | 2.09× |
| DeepSeek-V3 (baseline) | 2.33 | 1.90× |
| Bielik-v3 APT4 (Polish-optimized ref, paper) | ~1.62 | — |
| **ours: BPE 32k** | **1.56** | 1.25× |
| **ours: BPE 64k** | **1.42** | 1.14× |
| **ours: SuperBPE 34k** | **1.49** (beats BPE-34k 1.58) | 1.19× |

Our purpose-built Polish tokenizer cuts Polish token counts by **33–58%**
versus the universal baselines and matches/beats the Polish-optimized Bielik-v3
reference. SuperBPE's cross-word tokens push further toward English parity.
Details + reproduction in `results/` and `languages/pl/README.md`.

## Adapters

For each target model (Mistral 2/3/4, Llama, GLM, Nemotron, Qwen, Kimi,
DeepSeek, MiMo, GPT-2, Gemma, …) an adapter either **extends/merges** the
Polish vocabulary into the model's existing tokenizer or **replaces** it, then
initializes embeddings for new tokens with a quality-preserving method
(training-free OMP where possible; gpt2 is the from-scratch retrain PoC).

## Roadmap

Polish ships first. Other languages (fr, zh, cs, ru, uk, lt, es, de, hi, he,
ar, th, pt) follow using the same recipe-driven pipeline — tracked as BMAD next
steps.

## Development

```sh
task setup          # uv sync the workspace (.venv + all members + dev deps)
task check          # ruff lint + format check + ty types + pytest
task precommit      # run all pre-commit hooks (prek)
```

Secrets come from the `neuraffica` 1Password vault via `op.env`
(`task secrets:run -- <cmd>`). See [`AGENT.md`](AGENT.md) for the full
developer guide.
