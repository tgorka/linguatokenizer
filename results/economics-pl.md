# Polish inference-cost savings (datacenter optimizer)

Cost/throughput/context wins of moving a Polish workload from a baseline
tokenizer (Llama-3.1, **2.68 tok/word** on Polish) to our **BPE-64k**
(**1.42 tok/word**) — a **47% token cut**. Cost scales linearly with tokens, so
these are dollar-for-dollar. Per **1M input + 300k output Polish words/month**;
savings scale linearly with volume (×1000 for 1B words/mo, etc.).

| Pricing | $/mo before | $/mo after | Save | Throughput | Eff. context | KV mem |
|---------|------------:|-----------:|-----:|-----------:|-------------:|-------:|
| Claude Sonnet ($3/$15) | $20.10 | $10.65 | **47%** | ×1.89 | ×1.89 | −47% |
| GPT-4o ($2.50/$10) | — | — | 47% | ×1.89 | ×1.89 | −47% |
| Llama-70B (Together $0.88) | — | — | 47% | ×1.89 | ×1.89 | −47% |
| Self-host 70B/H100 ($2.30) | — | — | 47% | ×1.89 | ×1.89 | −47% |

(The savings **percentage** is identical across providers — it's a pure token
ratio; only the absolute $ differ. Full numbers in `economics-pl.json`.)

**What scales how** (from the research brief, `docs/datacenter-optimizer.md`):
API cost and KV-cache memory scale **linearly** with tokens (−47%); throughput
and effective context scale **inversely** (×1.89); prefill compute is ≥linear
(the attention term is O(n²), so long-context wins exceed 47%). Reproduce with
`tokenizer economics --baseline-fertility 2.68 --optimized-fertility 1.42
--pricing <preset>`.
