# Language: Russian (ru)

Cyrillic script — proves the pipeline generalizes beyond Latin. Baseline
tokenizer penalty ~2.48× English (FLORES-200); Cyrillic is over-segmented by
English-centric tokenizers. Our 32k byte-level BPE cuts Russian tokens **−54%**
vs baselines, to **~1.14× English**. See `results/multilang-ours.md`. Reproduce:

```bash
scripts/op-run.sh uv run --no-sync python scripts/stage_corpus.py \
  --lang ru --fineweb rus_Cyrl --wiki 20231101.ru --mb 60
scripts/op-run.sh uv run --no-sync python scripts/train_benchmark_lang.py \
  --lang ru --candidate ru-bpe-32k
```
