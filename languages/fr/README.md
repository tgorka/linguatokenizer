# Language: French (fr)

FLORES `fra_Latn`. Our 32k byte-level BPE (FineWeb-2 + Wikipedia, no model
retraining) cuts French tokens **-23.1%** vs
baselines, from a 1.663x English penalty to
**1.279x**. See `results/multilang-ours.md`. Reproduce
via `scripts/stage_corpus.py` + `scripts/train_benchmark_lang.py` (see
`docs/add-a-language.md`).
