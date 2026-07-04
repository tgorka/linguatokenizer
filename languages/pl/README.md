# Language: Polish (pl)

The first target language. Motivation, from the `ei-models` research: across
every model measured, Polish costs **~2.05× the output tokens of English**
(fertility 1.47× × verbosity 1.20×) — and ~¾ of that is the **tokenizer**
over-segmenting Polish, not the model being more verbose.

## Goal

Bring Polish **fertility** (tokens/word) to at or below the English baseline on
sentence-aligned content, **without downstream quality regression**.

## Reference playbook (Bielik v3, arXiv:2604.10799)

The Bielik team replaced Mistral's tokenizer with a Polish-optimized one
("APT4", ~32k vocab): fertility **3.22 → 1.62** tokens/word, chars/token
**2.40 → 4.78** — ~2× effective Polish context. They transplanted it with
**FOCUS** embedding init + staged continued-pretraining (20B tokens, partial
freezing) and validated quality held on Polish benchmarks (Open PL LLM
Leaderboard, EQ-Bench). We reproduce the *tokenizer* efficiency win and pursue
the **training-free** transplant path (OMP, arXiv:2506.06607) where possible.

## Approach

1. Train BPE / Unigram / SuperBPE candidates on Polish corpora (`recipe.yaml`).
2. Benchmark fertility on aligned FLORES PL/EN + held-out Wikipedia.
3. Iterate vocab size / algorithm / normalization; keep quality flat.
4. Adapt the winner onto target model tokenizers (extend/merge or replace) with
   quality-preserving embedding init; gpt2 is the retrain proof-of-concept.

Results land in `results/` and are summarized here as they come in.

## Baseline penalty (Epic 1 — real data)

Measured with `task benchmark:pl` on the 1012 sentence-aligned FLORES-200
devtest pairs (`task data:flores` fetches the ungated set). Fertility =
tokens/word; `pl/en` is the Polish token premium on identical content —
reproducing the ei-models ~2× finding, and gpt2's ~3×.

| Baseline | PL tok/word | EN tok/word | PL/EN |
|----------|------------:|------------:|------:|
| gpt2 | 3.713 | 1.230 | **3.02** |
| mistralai/Mistral-7B-v0.1 | 2.971 | 1.411 | 2.11 |
| speakleash/Bielik-11B-v2.3-Instruct | 2.971 | 1.411 | 2.11 |
| meta-llama/Llama-3.1-8B | 2.683 | 1.282 | 2.09 |
| Qwen/Qwen2.5-7B | 2.497 | 1.257 | 1.99 |
| deepseek-ai/DeepSeek-V3 | 2.332 | 1.230 | **1.90** |

Gated (skipped unless the vault `HF_TOKEN` account accepts terms):
`google/gemma-2-9b`, `speakleash/Bielik-11B-v3.0-Instruct` (the APT4 reference,
reported at ~1.62 tok/word in arXiv:2604.10799 — the target to beat). Full
data in `results/baselines-pl-en.json`.

**Target:** a trained Polish tokenizer at **PL/EN ≤ 1.0** (Epic 2).
