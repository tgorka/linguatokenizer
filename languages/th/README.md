# Language: Thai (th)

FLORES `tha_Thai`. Our 32k byte-level BPE (FineWeb-2 + Wikipedia, no model
retraining) cuts Thai tokens **-49.6%** vs
baselines, from a 3.938x English penalty to
**1.987x**. See `results/multilang-ours.md`. Reproduce
via `scripts/stage_corpus.py` + `scripts/train_benchmark_lang.py` (see
`docs/add-a-language.md`).
