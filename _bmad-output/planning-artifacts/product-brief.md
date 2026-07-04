# Product Brief: tokenizer

_Author: BMAD (Analyst) · Date: 2026-07-04 · Language: English_

## Problem

Large language models tokenize under-served languages far less efficiently
than English. Research in `Neuraffica/ei-models` measured that **Polish costs
~2.05× the tokens of English** for the same task (95% CI 1.94–2.34). The
penalty decomposes as **fertility 1.47× × verbosity 1.20×** — about
three-quarters of it is the *tokenizer*, which over-segments Polish because it
never learned the language's morphology. Universal tokenizers emit **~1.5–3.2
tokens/word for Polish vs ~1.3 for English**. This inflates inference cost and
latency (~1.4× wall-clock), shrinks the effective context window, and is a
fairness issue (arXiv:2305.15425).

The fix is proven: Bielik v3 replaced Mistral's tokenizer with a
Polish-optimized one and cut fertility from **3.22 → 1.62 tokens/word**
(arXiv:2604.10799) — roughly doubling effective Polish context — while holding
Polish task quality flat.

## Vision

An open, reproducible toolkit that produces **token-efficient,
morphology-aware tokenizers** for under-served languages (Polish first) and
**adapters** that graft them onto existing LLM tokenizers. Bring Polish
fertility to **at or below the English baseline** with **no downstream quality
regression**, ideally **without retraining** the target model.

## Target users

- **Model builders / fine-tuners** who serve Polish (and other) users and want
  lower cost-per-request and larger effective context.
- **Inference platform operators** paying the ~2× Polish token premium on
  throughput and margin.
- **Researchers** studying tokenizer fairness and morphology-aware
  segmentation.

## Goals & success metrics

| Goal | Metric | Target |
|------|--------|--------|
| Efficient Polish tokenizer | Fertility (tokens/word) on aligned FLORES PL vs EN | Polish ≤ English baseline (stretch: below) |
| No quality loss | Downstream task quality vs baseline tokenizer | ≤ 1% drop (parity) |
| Model coverage | Adapters/compilers for target model tokenizers | Framework + ≥5 concretely validated |
| No-retrain path | Training-free transplant (OMP) quality retention | Within quality budget on gpt2 PoC |
| Reproducibility | Recipe-driven, one-command benchmark | `task` runs end to end |
| Extensibility | Add a new language | Copy `languages/_template`, fill recipe |

## Scope

**In scope (v1):** Polish tokenizer candidates (BPE / Unigram / SuperBPE),
fertility + quality benchmark harness, baseline comparison, iterative
improvement, model-tokenizer adapters (extend/merge + replace) with
training-free embedding init, gpt2 retrain proof-of-concept, multi-language
repo structure.

**Out of scope (v1):** Full production model retraining; languages beyond
Polish (tracked as next steps: fr, zh, cs, ru, uk, lt, es, de, hi, he, ar, th,
pt); serving infrastructure.

## Key risks

- **Quality regression** when a tokenizer is swapped/extended → mitigate with
  training-free OMP init + a strict quality gate; gpt2 PoC to measure.
- **Multilingual trade-off** (Bielik saw European-benchmark drops) → measure
  and document; keep a merge/extend option that preserves the base vocabulary.
- **Data quality/licensing** for corpora → prefer permissively licensed,
  well-documented sources; store on the Nextcloud data plane.
- **Future/unavailable target models** (some in the list are hypothetical
  versions) → design adapters against tokenizer *formats* (HF tokenizers,
  SentencePiece, tiktoken), validate on what exists today.

## References

- ei-models findings (Polish ~2.05× tokens; ¾ tokenizer).
- Bielik v3 tokenizer optimization — arXiv:2604.10799.
- Tokenizer fairness — arXiv:2305.15425.
- Training-free transplant (OMP) — arXiv:2506.06607.
- Zero-Shot Tokenizer Transfer (ZeTT) — arXiv:2405.07883.
- SuperBPE — arXiv:2503.13423.
