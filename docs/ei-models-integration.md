# ei-models integration

Lets the [`ei-models`](https://github.com/Neuraffica/ei-models) experiments
re-run their Polish-vs-English token-penalty measurements **with our tokenizer**
and confirm the ~2× tokenizer penalty is removed.

## Why

ei-models measured that Polish costs **~2.05× the tokens of English**
(fertility 1.47× × verbosity 1.20× — ~¾ of the premium is the *tokenizer*). Our
tokenizer targets exactly that tokenizer half. This integration reproduces the
fertility-penalty framing so the two are directly comparable.

## API

`tokenizer_core.integrations.ei_models` — works with any `encode(text) -> ids`
object (HF fast tokenizers, `tokenizers.Tokenizer`, tiktoken, a callable):

```python
from tokenizer_core.integrations.ei_models import measure_penalty, compare

# one tokenizer's PL/EN fertility penalty on aligned docs
p = measure_penalty(model_tokenizer, pol_docs, eng_docs)
p.penalty_x            # e.g. 2.13  ("Polish costs 2.13x English")

# baseline vs our optimized tokenizer
result = compare(model_tokenizer, our_tokenizer, pol_docs, eng_docs)
result["baseline_penalty_x"]     # ~2.1
result["optimized_penalty_x"]    # lower
result["penalty_removed_pct"]    # % of the *excess* penalty eliminated
```

`pol_docs`/`eng_docs` must be **sentence-aligned** (same length, same content) —
FLORES-200 pol_Latn/eng_Latn devtest is the reference set; fetch it with
`task data:flores`.

## Measured result (FLORES-200, 1012 aligned pairs)

| Tokenizer | PL/EN penalty |
|-----------|--------------:|
| gpt2 | 3.02× |
| Mistral-7B-v0.1 | 2.14× |
| Llama-3.1-8B | 2.13× |
| Qwen2.5-7B | 1.99× |
| DeepSeek-V3 | 1.90× |
| **gpt2 + our Polish vocab (extend)** | **2.26×** — English intact, **38% of excess removed** |
| **our tokenizer standalone (replace)** | **1.56 tok/word** — ~English parity |

Full data: [`../results/ei-models-penalty-pl.md`](../results/ei-models-penalty-pl.md)
/ `.json`. Regenerate with `scripts/generate_product_reports.py`.

## Drop-in for an ei-models spike

The ei-models `token_usage` / Sp10 fertility spike compares model tokenizers on
PL vs EN. To add our tokenizer as a contender, load it and pass it to
`measure_penalty` alongside the model tokenizers — no other changes. To test a
model *served with* our tokenizer, build the adapted tokenizer
(`adapters.extend`/`replace`) and measure that.
