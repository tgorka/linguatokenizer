# Datacenter inference-cost optimizer

Reduce the cost of serving under-served languages by fixing the **tokenizer**,
not the model. Polish (and most non-English languages) cost ~2× the tokens of
English for the same text; almost every inference cost scales with token count,
so cutting tokens cuts cost, raises throughput, and enlarges effective context —
**with no model change**.

## The opportunity, quantified

Our Polish tokenizer cuts Polish fertility from ~2.68 tok/word (Llama-3.1
baseline) to **1.42 tok/word** (BPE-64k) — a **47% token reduction** on Polish
workloads. Because inference billing is linear in tokens, that is a ~47% cost
cut. See [`../results/economics-pl.md`](../results/economics-pl.md).

| Effect | Scales with tokens | 47% fewer tokens → |
|--------|--------------------|--------------------|
| API / self-host **cost** | linear | **−47%** (input + output legs) |
| **KV-cache memory**/request | linear | **−47%** → bigger batches |
| **Throughput** (req/GPU) | inverse | **×1.89** |
| **Effective context** (words/window) | inverse | **×1.89** |
| **Prefill** compute/latency | ≥ linear (attention O(n²)) | **≥ −47%**, more at long context |
| Decode | linear per token | fewer prompt + output tokens |

Output tokens cost ~3–5× input (decode is sequential/memory-bound), and batch
packing is the dominant real-world cost lever — smaller sequences pack better,
compounding the win.

## Using it

```bash
# Estimate savings for your workload + provider pricing:
tokenizer economics \
  --baseline-fertility 2.68 --optimized-fertility 1.42 \
  --pricing claude-sonnet --input-words 1e9 --output-words 3e8
```

Or from Python:

```python
from tokenizer_core.economics import estimate_savings, PRICING_PRESETS
r = estimate_savings(2.68, 1.42, pricing=PRICING_PRESETS["selfhost-70b-h100"],
                     input_words_per_month=1_000_000_000,
                     output_words_per_month=300_000_000)
print(r.savings_pct, r.monthly_savings, r.throughput_gain_x)
```

Measure your *own* baseline/optimized fertility on your traffic with
`tokenizer_core.benchmark.fertility.fertility(tokenizer, docs)`.

## Deploying the optimized tokenizer

1. **Pick an adapter** (`tokenizer_core.adapters`):
   - `extend(family, artifact)` — append the Polish vocabulary, **preserve all
     base token ids** → English stays intact, Polish gets cheaper. Best when you
     serve multiple languages and want zero base-language regression.
   - `replace(family, artifact)` — swap the tokenizer wholesale → maximum Polish
     efficiency (1.56 tok/word), needs a short retrain.
2. **Transplant embeddings training-free** (`transplant_embeddings`, FVT): new
   tokens get sane embeddings with no training. English preservation is measured
   (only ~0.2% of English tokens shift under `extend`). See
   [`../results/gpt2-transplant-poc.md`](../results/gpt2-transplant-poc.md).
3. **Export a serving bundle** (`export_bundle`): `tokenizer.json` +
   `embeddings.safetensors` + `manifest.json`, loadable into your serving stack.
   Large matrices go to the data plane (`$TOKENIZER_DATA_DIR`), not git.
4. **Optional short retrain** to adapt the new embeddings for generation
   (the classic APT4/Bielik-v3 recipe) — cheap because it starts from the FVT
   warm start, not random.

## Pricing presets

`tokenizer_core.economics.PRICING_PRESETS` ships indicative 2025/2026 numbers
(GPT-4o, Claude Sonnet/Haiku, Llama-70B hosted, self-host H100). **Pin to your
provider's live pricing page at runtime** — frontier prices move monthly. Pass a
custom `Pricing(name, input_usd_per_1m, output_usd_per_1m)`.

## Honest caveats

- The savings are on the **target-language** portion of traffic. Mixed traffic
  saves proportionally to its non-English share.
- `extend` keeps English identical but only partially reduces the Polish penalty
  (it keeps the base merges); `replace` reaches full efficiency but wants a
  retrain. Choose per your quality/effort budget.
- Quality is gated: `tokenizer_core.benchmark.quality` provides a bits-per-byte
  regression gate + a model-free base-language-preservation proxy.

## Sources

Token pricing and scaling: OpenAI/Anthropic pricing pages; Spheron & GMICloud
GPU cost-per-token benchmarks (2026); DigitalOcean KV-cache cost analysis;
vLLM/SGLang throughput reports. Tokenizer inefficiency: "LM Tokenizers Introduce
Unfairness Between Languages" (arXiv:2305.15425); Bielik v3 / APT4
(arXiv:2604.10799, −49.7% tok/word, ~2× effective context); SuperBPE
(arXiv:2503.13423, 33% fewer tokens, 27% less inference compute). Full URL list
in the project research brief.
