# Usage guide

End-to-end: from a language recipe to an adapted, deployable tokenizer.

## Install

```bash
task setup:all          # uv sync with all extras (train/corpora/adapters)
task check              # ruff + ty + pytest gate
```

Secrets (HF token, etc.) come from the `neuraffica` 1Password vault; prefix
network commands with `scripts/op-run.sh` or `task secrets:run --`.

## 1. Data

```bash
task data:flores                 # ungated FLORES-200 aligned PL/EN eval set
uv run python scripts/stage_pl_corpus.py   # stage Polish training text
```
Corpora + trained tokenizers live on the Nextcloud data plane
(`$TOKENIZER_DATA_DIR`), not git.

## 2. Baseline & train

```bash
task benchmark:pl                # baseline tokenizer PL/EN fertility table
task train:pl                    # train the recipe's bpe/unigram candidates
```
Candidates, corpora, and targets are declared in
[`../languages/pl/recipe.yaml`](../languages/pl/recipe.yaml). Add a language by
copying [`../languages/_template/`](../languages/_template/).

## 3. Adapt onto a model (no retraining)

```python
from tokenizer_core.adapters import extend, replace, read_token_embeddings, \
    transplant_embeddings, export_bundle
from transformers import AutoTokenizer

art = "$TOKENIZER_DATA_DIR/tokenizers/pl/pl-bpe-32k"
res = extend("gpt2", art)                    # or replace("gpt2", art)
base_tok = AutoTokenizer.from_pretrained("gpt2")
emb = transplant_embeddings(res, read_token_embeddings("gpt2"), base_tok)
export_bundle(res, emb, "$TOKENIZER_DATA_DIR/bundles/gpt2-pl",
              base_model="gpt2", method="extend+fvt")
```

- `extend` preserves all base token ids (English unchanged); `replace` swaps
  wholesale (max efficiency, wants a short retrain).
- `transplant_embeddings` (FVT) initializes new-token embeddings training-free.
- `export_bundle` writes `tokenizer.json` + `embeddings.safetensors` +
  `manifest.json` for a serving stack.

## 4. Quantify the win

```bash
tokenizer economics --baseline-fertility 2.68 --optimized-fertility 1.42 \
  --pricing claude-sonnet
```
See [`datacenter-optimizer.md`](datacenter-optimizer.md).

## 5. Gate quality

```python
from tokenizer_core.benchmark.quality import quality_gate, base_language_preservation
quality_gate(before_bpb, after_bpb, budget_pct=1.0).passed   # bits-per-byte
base_language_preservation(base_ids, adapted_ids, base_vocab_size=50257)  # model-free
```

## Reference

- Adapters + transplant: [`../results/gpt2-transplant-poc.md`](../results/gpt2-transplant-poc.md)
- Benchmarks: [`benchmark-report.md`](benchmark-report.md)
- ei-models integration: [`ei-models-integration.md`](ei-models-integration.md)
- Canonical dev guide: [`../AGENT.md`](../AGENT.md)
