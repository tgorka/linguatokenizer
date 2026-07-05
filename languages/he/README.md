# Language: Hebrew (he)

FLORES `heb_Hebr`. Our 32k byte-level BPE (FineWeb-2 + Wikipedia, no model
retraining) cuts Hebrew tokens **-67.5%** vs
baselines, from a 2.934x English penalty to
**0.955x**. See `results/multilang-ours.md`. Reproduce
via `scripts/stage_corpus.py` + `scripts/train_benchmark_lang.py` (see
`docs/add-a-language.md`).
