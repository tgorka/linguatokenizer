# Language: Polish (pl)

The first target language. Motivation, from the `ei-models` research: across
every model measured, Polish costs **~2.05× the output tokens of English**
(fertility 1.47× × verbosity 1.20×) — and ~¾ of that is the **tokenizer**
over-segmenting Polish, not the model being more verbose.

## Goal

Bring Polish **fertility** (tokens/word) to at or below the English baseline on
sentence-aligned content, **without downstream quality regression**.

## Reference playbook (Bielik v3, arXiv:2604.10799)

The Bielik team replaced Mistral's tokenizer with a Polish-optimized one
("APT4", ~32k vocab): fertility **3.22 → 1.62** tokens/word, chars/token
**2.40 → 4.78** — ~2× effective Polish context. They transplanted it with
**FOCUS** embedding init + staged continued-pretraining (20B tokens, partial
freezing) and validated quality held on Polish benchmarks (Open PL LLM
Leaderboard, EQ-Bench). We reproduce the *tokenizer* efficiency win and pursue
the **training-free** transplant path (OMP, arXiv:2506.06607) where possible.

## Approach

1. Train BPE / Unigram / SuperBPE candidates on Polish corpora (`recipe.yaml`).
2. Benchmark fertility on aligned FLORES PL/EN + held-out Wikipedia.
3. Iterate vocab size / algorithm / normalization; keep quality flat.
4. Adapt the winner onto target model tokenizers (extend/merge or replace) with
   quality-preserving embedding init; gpt2 is the retrain proof-of-concept.

Results land in `results/` and are summarized here as they come in.
