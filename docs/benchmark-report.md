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

## 4. Datacenter economics

Polish workload, Llama baseline (2.68) → our BPE-64k (1.42) = **47% fewer
tokens** → 47% lower cost, ×1.89 throughput/context, −47% KV memory (linear in
tokens). Source: [`../results/economics-pl.md`](../results/economics-pl.md).

## 5. ei-models penalty removal

Baselines ~1.9–3.0× PL/EN penalty → extend keeps English + removes 38% of gpt2's
excess; replace reaches ~parity. Source:
[`../results/ei-models-penalty-pl.md`](../results/ei-models-penalty-pl.md).

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
