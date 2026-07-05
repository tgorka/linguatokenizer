# Language: Chinese (zh)

FLORES `zho_Hans`. Our 32k byte-level BPE (FineWeb-2 + Wikipedia, no model
retraining) cuts Chinese tokens **-40.3%** vs
baselines, from a 1.594x English penalty to
**0.952x**. See `results/multilang-ours.md`. Reproduce
via `scripts/stage_corpus.py` + `scripts/train_benchmark_lang.py` (see
`docs/add-a-language.md`).
