# Language: Czech (cs)

FLORES `ces_Latn`. Our 32k byte-level BPE (FineWeb-2 + Wikipedia, no model
retraining) cuts Czech tokens **-44.6%** vs
baselines, from a 1.949x English penalty to
**1.08x**. See `results/multilang-ours.md`. Reproduce
via `scripts/stage_corpus.py` + `scripts/train_benchmark_lang.py` (see
`docs/add-a-language.md`).
