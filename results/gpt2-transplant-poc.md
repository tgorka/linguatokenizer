# gpt2 transplant PoC — training-free, no retraining (Epic 3.4 / 4.2)

Grafts the trained Polish tokenizer onto **real gpt2** and initializes the new
tokens' embeddings with **FVT** (mean of the base model's embeddings of each new
token's sub-tokens) — **no retraining, no torch** (gpt2's embedding matrix is
read from safetensors as numpy). Reproduce: `scripts/op-run.sh uv run --no-sync
python scripts/gpt2_transplant_poc.py`.

| metric | value |
|--------|-------|
| vocab | 50257 → 76663 (+26406 Polish tokens) |
| **English preserved** | 26991 vs 27044 base tokens (+0.2%); only **0.2%** of English tokens use new ids |
| **Polish efficiency** | 3.713 → 2.773 tok/word (**−25%**) |
| new-token embeddings | finite, norm 2.58 (base 3.96) — sane FVT init |

**Read:** the transplant is **quality-preserving on English essentially for
free** — base token ids and their embeddings are untouched, and the appended
Polish vocabulary re-tokenizes only 0.2% of English — while Polish immediately
costs 25% fewer tokens, each new token carrying a sensible FVT embedding.

The FVT embeddings are a *starting point*: a short continued-pretraining pass on
Polish (torch/GPU, the classic APT4/Bielik-v3 recipe) adapts them for strong
generative Polish output. This PoC delivers the no-retrain baseline that makes
that step cheap and optional; the full replace path (`adapters.replace`) reaches
the standalone Polish fertility (~1.56) at the cost of re-mapping more of the
embedding table.
