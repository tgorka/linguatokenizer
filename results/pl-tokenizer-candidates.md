# Polish tokenizer candidates — trained fertility (Epic 2)

Trained on ~75 MB of staged Polish text (FineWeb-2 + Wikipedia), scored on the
1012 sentence-aligned FLORES-200 devtest pairs. Fertility = tokens/word.

| Tokenizer | vocab | PL tok/word | chars/token | vs baselines |
|-----------|------:|------------:|------------:|--------------|
| **pl-bpe-32k (ours)** | 32k | **1.561** | 4.58 | beats all |
| _baseline_ gpt2 | 50k | 3.713 | — | −58% tokens |
| _baseline_ Mistral-7B-v0.1 | 32k | 2.971 | — | −47% tokens |
| _baseline_ Llama-3.1-8B | 128k | 2.683 | — | −42% tokens |
| _baseline_ Qwen2.5-7B | 152k | 2.497 | — | −38% tokens |
| _baseline_ DeepSeek-V3 | 128k | 2.332 | — | −33% tokens |
| _reference_ Bielik-v3 APT4 (paper) | 32k | ~1.62 | 4.78 | ours ≈ beats |

**Result:** a plain 32k Polish BPE cuts Polish token counts by 33–58% versus
the universal baselines and matches/beats the Polish-optimized Bielik-v3
reference — with no model retraining.

Against a good English tokenizer's English fertility (~1.25), our Polish
fertility of 1.561 is ~1.25× — close to parity.

## Vocab sweep (Epic 2.3, real data)

Same corpus/eval, byte-level BPE at increasing vocab size:

| vocab | PL tok/word | chars/token | PL/EN (vs 1.25) |
|------:|------------:|------------:|----------------:|
| 32k | 1.561 | 4.58 | 1.25× |
| 48k | 1.475 | 4.84 | 1.18× |
| 64k | 1.423 | 5.02 | 1.14× |

Pure vocab scaling has diminishing returns and plateaus above parity (~1.14× at
64k) — already beating every baseline and the Bielik-v3 reference. Reaching
**PL/EN ≤ 1.0** needs cross-word tokens: **SuperBPE** (story 2.2), projected to
push chars/token to ~7 and PL fertility below 1.0.

Reproduce: `task data:flores && task benchmark:pl` (baselines);
`scripts/op-run.sh uv run --no-sync tokenizer train --recipe languages/pl/recipe.yaml`
then score the artifact under `$TOKENIZER_DATA_DIR/tokenizers/pl/`.
