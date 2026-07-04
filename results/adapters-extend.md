# Model adapters — extend/merge (Epic 3.2, real validation)

The extend/merge adapter grafts a trained Polish tokenizer onto an existing
model's tokenizer, **preserving every original token id** (so the base model's
embeddings stay valid) and appending Polish-only tokens after `base_vocab_size`.
Only the appended tokens need embedding initialization — the job of the
training-free OMP transplant (story 3.4).

Validated on real `gpt2` + our `pl-bpe-32k` artifact, scored on FLORES-200:

| tokenizer | vocab | PL tok/word | Polish tokens |
|-----------|------:|------------:|--------------|
| gpt2 (base) | 50257 | 3.713 | — |
| **gpt2 + Polish (extended)** | 76663 | **2.773** | **−25%** |

All 50257 base ids preserved unchanged. Extend trades some efficiency for
compatibility (it keeps gpt2's English-centric merges); a full **replace**
(story 3.3) reaches the standalone Polish fertility (~1.56) but requires
re-mapping more of the embedding table. Reproduce with
`tokenizer_core.adapters.extend.extend("gpt2", "<artifact_dir>")`.
