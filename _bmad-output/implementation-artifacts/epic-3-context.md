# Epic 3 Context: Model adapters & compilers

<!-- Generated from planning artifacts. Regenerate with compile-epic-context if planning docs change. -->

## Goal

Graft the Polish tokenizer produced in Epic 2 onto existing model tokenizers,
training-free where possible. This turns a trained tokenizer artifact into
something a model builder can actually adopt: a registry that knows the
tokenizer format of each target model, two grafting strategies (extend the
existing vocabulary vs. replace it wholesale), and a training-free embedding
initializer so new tokens get usable embeddings without retraining. Together
these de-risk adoption — they let a model builder try the Polish tokenizer on
a real target model and get a concrete quality/efficiency trade-off, without
committing to a full retrain up front.

## Stories

- Story 3.1: Target-model registry
- Story 3.2: Extend/merge adapter (compiler)
- Story 3.3: Replace adapter
- Story 3.4: Training-free OMP embedding init

## Requirements & Constraints

- Adapters must cover a registry of target tokenizer families: mistral 2/3/4,
  llama, glm 5.2, nemotron 3, qwen 3.5, kimi k2.7, deepseek v4, MiMo 2.5, gpt2,
  gemma 4. Each registry entry records format, vocab, and special tokens, plus
  whether it's validated today (weights exist) vs. format-ready (future/
  hypothetical model version).
- Adapters target tokenizer *formats* (HF `tokenizers`, SentencePiece,
  tiktoken/BPE), not specific model instances, so unavailable or future models
  degrade gracefully rather than failing outright.
- Extend/merge must preserve the target's existing token ids and vocabulary,
  appending new (Polish) tokens rather than disturbing the base — a round-trip
  encode/decode check is required, and Polish fertility must improve versus
  the unextended target.
- Replace must produce a fully model-compatible tokenizer plus an
  initialization plan, preserving or correctly re-mapping special tokens and
  chat templates. When to prefer replace vs. extend must be documented.
- Embedding init (OMP) must work without any retraining or GPU: reconstruct
  each new token's embedding as a sparse combination of shared "anchor" token
  embeddings from the target model. Provide FOCUS-style overlap-weighted mean
  and plain-mean as fallbacks when OMP is unsuitable. Output must be an
  embedding matrix directly consumable by `transformers`.
- Framework must concretely validate adapters against at least 5 real,
  currently-available models (not just format handlers).
- No adapter is considered "done" until it passes the quality-drop gate
  (≤1% downstream quality drop) — the gate itself is built in Epic 4, but
  Epic 3 adapters are the thing being gated.
- Everything in this epic runs CPU-only (no-GPU core invariant); GPU is
  reserved for the optional gpt2 retrain in Epic 4.

## Technical Decisions

- Component home: `tokenizer_core.adapters`, with `registry.py` (target-model
  registry), `base.py` (shared Adapter protocol: extend|replace -> new
  tokenizer + init plan), `extend.py`, `replace.py`, `embed_omp.py`. (A fifth
  module, `gpt2_poc.py`, wires these into the Epic 4 end-to-end PoC — it
  consumes this epic's adapters but is not itself part of Epic 3.)
- Data flow: a trained candidate + baseline target tokenizer feed
  `adapters.(extend|replace)`, producing a target-model tokenizer; `embed_omp`
  then produces the matching embedding matrix; the result flows into the
  Epic 4 quality gate.
- OMP reference: arXiv:2506.06607 (training-free transplant). Related
  technique for context: ZeTT, arXiv:2405.07883.
- Heavy deps (`tokenizers`, `transformers`, `numpy`) are extras, imported
  lazily behind guards — adapters must not force these onto the base install.
- Recipe is still the source of truth: adapter behavior for a language is
  driven by `languages/<iso>/recipe.yaml` targets/baselines, not hardcoded.
- Testing expectations for this epic: unit tests for the registry (using a
  local stand-in tokenizer, no network), and OMP math unit-tested on small
  synthetic matrices. Larger end-to-end adapter runs belong to the (separate)
  gpt2 PoC integration path, not presubmit.

## Cross-Story Dependencies

- 3.1 (registry) is a prerequisite for 3.2 and 3.3 — both adapters need to
  look up a target's format/vocab/special-tokens before grafting.
- 3.2 and 3.3 both depend on a trained Polish tokenizer candidate from Epic 2
  as their vocabulary source.
- 3.4 (OMP embedding init) is consumed by both 3.2 and 3.3's init plans —
  whichever adapter is used, the resulting new tokens need embeddings.
- Epic 3's adapters feed directly into Epic 4: the quality regression gate
  (Story 4.1) evaluates adapter output, and the gpt2 transplant PoC
  (Story 4.2) exercises extend/replace + OMP end to end on gpt2.
