# Data provenance & licenses

This project trains and benchmarks tokenizers on third-party corpora. The
corpora themselves are **not** committed to git — they are staged onto the
Nextcloud data plane (`$TOKENIZER_DATA_DIR`) by the scripts below, and the
recipes reference them as local files. This page records where each dataset
comes from and under what terms, so any trained artifact can be traced back to
its sources.

> **Licensing note.** These dataset licenses are independent of this
> repository's own license. A tokenizer's vocabulary is a statistical summary
> of its training text, but before **redistributing** any artifact (trained
> tokenizer, transplanted model, or bundle) derived from a corpus, check that
> corpus's terms — some (e.g. CC-BY-SA, GFDL) carry attribution and
> share-alike obligations.

## Training corpora

| Dataset | Source (Hugging Face / URL) | Used for | License |
|---------|-----------------------------|----------|---------|
| FineWeb-2 | `HuggingFaceFW/fineweb-2` (per-language configs, e.g. `pol_Latn`) | Primary training text (filtered web), all languages | ODC-BY 1.0 (underlying Common Crawl text subject to its terms) |
| Wikipedia | `wikimedia/wikipedia` (e.g. `20231101.pl`) | Secondary training text (encyclopedic) | CC-BY-SA 4.0 + GFDL |
| CodeParrot (clean) | `codeparrot/codeparrot-clean-valid` | Technical/code corpus for the routing experiment | Derived from public GitHub repositories under their respective open-source licenses; see the dataset card |

Staged by:

- `scripts/stage_pl_corpus.py` — Polish (FineWeb-2 `pol_Latn` + Wikipedia
  `20231101.pl`), bounded and ungated.
- `scripts/stage_corpus.py` — the generic per-language stager used for the
  other 13 languages (FineWeb-2 `<iso639-3>_<Script>` + Wikipedia
  `<YYYYMMDD>.<iso639-1>`).

Sizes are bounded (≈100 MB FineWeb-2 + ≈50 MB Wikipedia per language) so
training is fast and offline; a 32k vocabulary needs only tens of MB.

## Evaluation corpus

| Dataset | Source | Used for | License |
|---------|--------|----------|---------|
| FLORES-200 | `https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz` (the public, ungated NLLB tarball; the HF mirror `facebook/flores` is gated) | Sentence-aligned fertility + quality benchmarking (identical content across languages) | CC-BY-SA 4.0 |

Fetched by `scripts/fetch-flores.sh` into
`$TOKENIZER_DATA_DIR/flores200`. FLORES-200 provides the same sentences in
every language, which is what makes "Polish costs N× the tokens of English for
the same sentence" a fair, apples-to-apples measurement.

## Base models & tokenizers (benchmarking / adapters)

The baseline and cross-model benchmarks load stock tokenizers (and, for the
gpt2 PoC, the model weights) from the Hugging Face Hub: `gpt2`,
`mistralai/Mistral-7B-v0.1`, `speakleash/Bielik-*`, `meta-llama/Llama-3.1-8B`,
`Qwen/Qwen2.5-7B`, `deepseek-ai/DeepSeek-V3`, and others named in the recipes
and `results/`. These are used only to **measure** stock behavior and, for
gpt2, to demonstrate the transplant/retrain adapters. Each model is governed by
its own license/usage terms on the Hub; some are gated and require accepting
those terms (and a `HF_TOKEN`). No third-party model weights are committed to
this repository.

## What lives where

- **In git:** code, recipes (`languages/<iso>/recipe.yaml`), benchmark result
  summaries (`results/*.{json,md}`), and docs.
- **On the Nextcloud data plane (`$TOKENIZER_DATA_DIR`, off git):** the staged
  corpora, trained tokenizer binaries, transplanted/retrained models, and
  deployable bundles. Recipes are the reproducible source of truth for
  rebuilding them.
