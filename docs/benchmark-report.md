# Benchmark report (consolidated)

All numbers measured on **FLORES-200 devtest**, 1012 sentence-aligned Polish/
English pairs. Fertility = tokens/word (lower is better). Reproduce with
`task benchmark:pl`, `task train:pl`, and the scripts under `scripts/`.

## 1. The penalty (baselines)

| Tokenizer | PL tok/word | EN tok/word | PL/EN |
|-----------|------------:|------------:|------:|
| gpt2 | 3.71 | 1.23 | 3.02× |
| Mistral-7B-v0.1 / Bielik-v2.3 | 2.97 | 1.41 | 2.11× |
| Llama-3.1-8B | 2.68 | 1.28 | 2.09× |
| Qwen2.5-7B | 2.50 | 1.26 | 1.99× |
| DeepSeek-V3 | 2.33 | 1.23 | 1.90× |

Confirms the ei-models finding: Polish costs ~2× (gpt2 ~3×) the tokens of
English. Source: [`../results/baselines-pl-en.md`](../results/baselines-pl-en.md).

## 2. Our tokenizer

| Candidate | PL tok/word | vs baselines | vs EN parity (~1.25) |
|-----------|------------:|-------------|----------------------|
| BPE 32k | 1.56 | −33 to −58% | 1.25× |
| BPE 48k | 1.48 | | 1.18× |
| **BPE 64k** | **1.42** | | **1.14×** |
| SuperBPE 34k | 1.49 (beats BPE-34k 1.58) | | 1.19× |

Beats every baseline and the Polish-optimized Bielik-v3 reference (~1.62).
Vocab scaling plateaus ~1.14×; SuperBPE's cross-word tokens push further.
Source: [`../results/pl-tokenizer-candidates.md`](../results/pl-tokenizer-candidates.md).

## 3. Adapters (real gpt2, no retraining)

| Path | PL tok/word | English |
|------|------------:|---------|
| gpt2 baseline | 3.71 | — |
| **extend** (gpt2 + Polish vocab) | 2.77 (−25%) | 99.8% token-identical, embeddings preserved |
| **replace** (wholesale) | 1.56 (full) | new table, FVT warm-started, wants short retrain |

Embeddings transplanted training-free (FVT), torch-free. Source:
[`../results/gpt2-transplant-poc.md`](../results/gpt2-transplant-poc.md),
[`../results/adapters-extend.md`](../results/adapters-extend.md).

### 3b. Optional retrain — recover the transplant's quality (CPU, no GPU)

The FVT transplant is a warm start: the merged Polish tokens are new to the
model, so Polish bits-per-byte temporarily rises. A short fine-tune of **only
the new token rows** (base model frozen, so English is untouched) recovers most
of it — on CPU, no GPU:

| Polish bits-per-byte | stock | transplant | retrain |
|----------------------|------:|-----------:|--------:|
| gpt2 | 3.62 | 4.74 | **3.86** |

That recovers **78%** of the transplant penalty in 30 steps, while English
stays within **+0.2%** (quality gate PASS) and Polish keeps its **−23.5%** token
saving. The CPU path trains a compact delta over just the new rows (base frozen),
so it never materialises a full-vocab gradient and fits a small (~2 GiB) box;
more steps on a less memory-constrained host recover more. Source:
[`../results/gpt2-retrain.md`](../results/gpt2-retrain.md).

## 4. Datacenter economics

Polish workload, Llama baseline (2.68) → our BPE-64k (1.42) = **47% fewer
tokens** → 47% lower cost, ×1.89 throughput/context, −47% KV memory (linear in
tokens). Source: [`../results/economics-pl.md`](../results/economics-pl.md).

## 5. ei-models penalty removal

Baselines ~1.9–3.0× PL/EN penalty → extend keeps English + removes 38% of gpt2's
excess; replace reaches ~parity. Source:
[`../results/ei-models-penalty-pl.md`](../results/ei-models-penalty-pl.md).

## 6. Multi-language (generalization)

Baseline penalty landscape across the target languages (tokens/sentence vs
English, avg of 5 baselines) — non-Latin scripts are hit hardest:

| worst → | hi 4.34× · th 3.94× · he 2.93× · uk 2.79× · ar 2.54× · ru 2.50× |
|---------|----------------------------------------------------------------|
| milder  | lt 2.11× · pl 1.98× · cs 1.95× · de 1.68× · fr 1.66× · es/zh/pt ~1.6× |

Our recipe-driven pipeline, applied end to end to **14 languages** across 7
scripts (32k BPE, no retraining) — token reduction vs baselines and the
resulting penalty vs English:

| Lang | Script | Reduction | Penalty (baseline → ours) |
|------|--------|----------:|---------------------------|
| Hebrew | Hebrew | −68% | 2.93× → **0.96×** |
| Ukrainian | Cyrillic | −59% | 2.77× → **1.13×** |
| Arabic | Arabic | −59% | 2.56× → **1.05×** |
| Russian | Cyrillic | −54% | 2.48× → **1.14×** |
| Lithuanian | Latin | −52% | 2.11× → **1.01×** |
| Chinese | Han | −40% | 1.59× → **0.95×** |
| Czech / Polish | Latin | −45% | ~1.95× → **1.08–1.10×** |
| Portuguese | Latin | −30% | 1.56× → **1.09×** |
| German | Latin | −32% | 1.67× → **1.14×** |
| Spanish / French | Latin | −27 / −23% | ~1.6× → **1.19 / 1.28×** |
| Thai | Thai | −77% | 3.94× → **0.91×** |
| Hindi | Devanagari | −76% | 4.34× → **1.04×** |

Most languages reach ~1.0–1.3× English (Hebrew, Chinese & Thai **below**
parity), cutting tokens 23–77%. The byte-dense scripts (Devanagari, Thai) need
`split_words: false` in the recipe: the GPT-2 word-split regex over-fragments
them (~1.5 chars/token), and disabling it lets BPE merge those sequences
(~4 chars/token) — Hindi 3.05×→**1.04×**, Thai 1.99×→**0.91×**. Source: [`../results/multilang-penalty.md`](../results/multilang-penalty.md)
(baseline landscape), [`../results/multilang-ours.md`](../results/multilang-ours.md)
(ours). Add a language: [`add-a-language.md`](add-a-language.md).

## 7. Cross-model (old vs new tokenizer, per model)

Polish tokens/sentence for each model's **stock** tokenizer vs ours applied by
extend/replace, plus encode speed. Reduction % = cost-savings % (linear in
tokens, both input and output legs).

| Model | Stock | Extend | Replace | Ours enc tok/s |
|-------|------:|-------:|--------:|---------------:|
| gpt2 | 71.6 | −25% | **−58%** | 1.1M |
| Mistral-7B | 56.3 | n/a (SP) | **−47%** | 1.1M |
| Llama-3.1 | 50.8 | −29% | **−41%** | 1.1M |
| Qwen2.5 | 48.2 | −9% | **−38%** | 1.1M |
| DeepSeek-V3 | 45.0 | −9% | **−33%** | 1.1M |

Our tokenizer encodes at ~1.1M tok/s — on par with or faster than the stock
tokenizers (Llama 0.5M, Qwen 0.9M, DeepSeek 1.0M) — so the token win costs no
tokenization throughput. Full inference wall-clock (prefill/decode) needs a
served model on GPU; `tokenizer_core.economics` projects it. Source:
[`../results/cross-model-benchmark.md`](../results/cross-model-benchmark.md).

## Method notes

- Fertility uses a Unicode-aware word regex + NFC normalization; token counts
  use `.ids` for HF encodings. Baseline/ei-models penalties encode with
  `add_special_tokens=False` so BOS/EOS don't inflate counts.
- Training memory: byte-level BPE OOMs on the full 165 MB corpus; candidates are
  trained on a bounded ~75 MB sample (a 32k vocab needs only tens of MB).
  Unigram (HF `tokenizers` 0.22) is slow/memory-heavy on large data.
- SuperBPE stage-2 is `O(merges × corpus)` in pure Python; the full
  16k-superword run is compute-bound in this environment (2k-superword result
  shown, direction confirmed).
