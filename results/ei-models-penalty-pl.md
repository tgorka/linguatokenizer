# ei-models: Polish token penalty, before and after

Reproduces the ei-models PL-vs-EN token-penalty framing (the *tokenizer* half of
the ~2.05× total premium) on 1012 aligned FLORES-200 sentences, using
`tokenizer_core.integrations.ei_models`. `penalty_x` = Polish tok/word ÷ English
tok/word for the **same** tokenizer.

| Tokenizer | Polish penalty (PL/EN) |
|-----------|-----------------------:|
| gpt2 | 3.02× |
| Mistral-7B-v0.1 | 2.14× |
| Llama-3.1-8B | 2.13× |
| Qwen2.5-7B | 1.99× |
| DeepSeek-V3 | 1.90× |
| **gpt2 + our Polish vocab (extend)** | **2.26×** (English kept intact; **38% of the excess penalty removed**) |
| **our Polish tokenizer standalone (replace)** | **1.56 tok/word** Polish — ~parity with a good English tokenizer (~1.25) |

**How to use it in ei-models** — drop-in, any `encode`-capable tokenizer:

```python
from tokenizer_core.integrations.ei_models import measure_penalty, compare
baseline = measure_penalty(model_tokenizer, pol_docs, eng_docs)   # ~2x
result   = compare(model_tokenizer, our_tokenizer, pol_docs, eng_docs)
result["penalty_removed_pct"]   # how much of the excess penalty is gone
```

So an ei-models experiment can re-run its Polish token spike with our tokenizer
and confirm the penalty shrinks (extend, bilingual) or effectively disappears
(replace, Polish-only). Full data in `ei-models-penalty-pl.json`.
