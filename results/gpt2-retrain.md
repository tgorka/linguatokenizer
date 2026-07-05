# gpt2 retrain PoC — adapt the transplanted Polish tokens (Epic 4.2)

Extends **real gpt2** with the trained Polish vocabulary, FVT-initialises the new
tokens (training-free, see [`gpt2-transplant-poc.md`](gpt2-transplant-poc.md)),
then runs a short fine-tune of **only the new token rows** (the base model is
frozen, so English is preserved by construction). Quality is measured in
**bits-per-byte** — a tokenizer-independent, per-byte metric, so the stock and
extended tokenizers compare fairly. Ran on **cpu**, 30
steps. Reproduce: `MALLOC_ARENA_MAX=1 uv run --no-sync python scripts/gpt2_retrain.py`.

| Polish bits-per-byte | value |
|----------------------|-------|
| stock gpt2 | 3.619 |
| + transplant (FVT, no retrain) | 4.735 |
| + short retrain (new rows only) | **3.863** |

| efficiency / safety | value |
|---------------------|-------|
| Polish tokens (extended vs stock) | 5683 vs 7429 (**-23.5%**) |
| transplant penalty recovered | **78%** |
| English bits-per-byte (stock -> retrain) | 1.055 -> 1.057 (+0.2%) |
| English quality gate (<=2%) | **PASS** |

**Read:** the training-free transplant cuts Polish tokens by
23.5% but temporarily raises Polish bits-per-byte (the model
has never seen the merged tokens); a short fine-tune of just the new rows
recovers 78% of that penalty while English stays within
+0.2% (gate PASS). This confirms the optional retrain step
is cheap — it touches only the new embeddings, needs no GPU, and does not
regress the base language. The full recipe scales the step count / eval set on a
GPU (`task gpu:run`).
