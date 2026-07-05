# Language: Spanish (es)

FLORES `spa_Latn`. Our 32k byte-level BPE (FineWeb-2 + Wikipedia, no model
retraining) cuts Spanish tokens **-26.5%** vs
baselines, from a 1.62x English penalty to
**1.19x**. See `results/multilang-ours.md`. Reproduce
via `scripts/stage_corpus.py` + `scripts/train_benchmark_lang.py` (see
`docs/add-a-language.md`).
