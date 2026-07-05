# Language: Hindi (hi)

FLORES `hin_Deva`. Our 32k byte-level BPE (FineWeb-2 + Wikipedia, no model
retraining) cuts Hindi tokens **-29.7%** vs
baselines, from a 4.343x English penalty to
**3.051x**. See `results/multilang-ours.md`. Reproduce
via `scripts/stage_corpus.py` + `scripts/train_benchmark_lang.py` (see
`docs/add-a-language.md`).
