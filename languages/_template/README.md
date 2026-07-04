# Language: &lt;name&gt;

Copy this folder to `languages/<iso-639-1>/` to add a language.

## Files

- `recipe.yaml` — the source of truth: corpora, candidate tokenizers,
  fertility/quality targets, and baselines to beat. See the template comments.
- `README.md` — this file: language-specific notes (morphology, script,
  known tokenizer pitfalls, chosen corpora rationale, results summary).

## Workflow

1. Fill in `recipe.yaml` (corpora + candidates + targets).
2. Acquire corpora to the Nextcloud data store (`$TOKENIZER_DATA_DIR`).
3. Train candidates and benchmark fertility vs the English baseline.
4. Iterate on vocab size / algorithm / normalization without quality loss.
5. Pick the winner; adapt it onto target model tokenizers (see `packages/
   tokenizer-core/src/tokenizer_core/adapters`).

Results (fertility tables, quality parity) are written to `results/` and
summarized back here.
