# Language: Portuguese (pt)

FLORES `por_Latn`. Our 32k byte-level BPE (FineWeb-2 + Wikipedia, no model
retraining) cuts Portuguese tokens **-30.1%** vs
baselines, from a 1.564x English penalty to
**1.093x**. See `results/multilang-ours.md`. Reproduce
via `scripts/stage_corpus.py` + `scripts/train_benchmark_lang.py` (see
`docs/add-a-language.md`).
