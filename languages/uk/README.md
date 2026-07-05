# Language: Ukrainian (uk)

FLORES `ukr_Cyrl`. Our 32k byte-level BPE (FineWeb-2 + Wikipedia, no model
retraining) cuts Ukrainian tokens **-59.3%** vs
baselines, from a 2.768x English penalty to
**1.127x**. See `results/multilang-ours.md`. Reproduce
via `scripts/stage_corpus.py` + `scripts/train_benchmark_lang.py` (see
`docs/add-a-language.md`).
