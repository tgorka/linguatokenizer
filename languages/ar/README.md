# Language: Arabic (ar)

FLORES `arb_Arab`. Our 32k byte-level BPE (FineWeb-2 + Wikipedia, no model
retraining) cuts Arabic tokens **-59.2%** vs
baselines, from a 2.561x English penalty to
**1.045x**. See `results/multilang-ours.md`. Reproduce
via `scripts/stage_corpus.py` + `scripts/train_benchmark_lang.py` (see
`docs/add-a-language.md`).
