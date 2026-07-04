# Language: German (de)

Latin script. Baseline tokenizer penalty ~1.67× English (FLORES-200). Our 32k
byte-level BPE (FineWeb-2 + Wikipedia, no model retraining) cuts German tokens
**−32%** vs baselines, bringing German to **~1.14× English** (near parity). See
`results/multilang-ours.md`. Reproduce:

```bash
scripts/op-run.sh uv run --no-sync python scripts/stage_corpus.py \
  --lang de --fineweb deu_Latn --wiki 20231101.de --mb 60
scripts/op-run.sh uv run --no-sync python scripts/train_benchmark_lang.py \
  --lang de --candidate de-bpe-32k
```
