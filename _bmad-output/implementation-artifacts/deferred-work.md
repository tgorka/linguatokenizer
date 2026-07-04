# Deferred Work

Findings surfaced during review that are real but out of scope for the story
that surfaced them. Append-only; each entry names its source spec.

- source_spec: `spec-1-1-recipe-schema-and-loader.md`
  summary: `Recipe` does not enforce that `train_corpora` and `eval_corpora` are disjoint, so a train/eval leak (same id+config+split in both) validates silently.
  evidence: The recipes and docstrings stress eval corpora are "never trained on," but nothing checks it; the Polish recipe reuses `wikimedia/wikipedia` across train/eval (legitimate via differing splits), yet an identical entry in both would pass unnoticed. Needs a cross-field validator that allows same-id-different-split but rejects true duplicates.

- source_spec: `spec-1-1-recipe-schema-and-loader.md`
  summary: Duplicate `candidate.name` and duplicate `baselines` entries are accepted with no uniqueness check.
  evidence: Candidates and baselines are keyed/persisted by name downstream; two `pl-bpe-32k` candidates would collide in output artifacts. A `model_validator` asserting name/id uniqueness would fail these fast.

- source_spec: `spec-1-1-recipe-schema-and-loader.md`
  summary: `yaml.safe_load` silently collapses duplicate YAML keys to the last value, so a mistyped/duplicated recipe key loads the wrong config undetected.
  evidence: PyYAML's default loader does not raise on duplicate mapping keys; a `recipe.yaml` with two `vocab_size:` (or a duplicated corpus block) would parse without complaint. A loader that raises on duplicate keys would preserve the "fail fast on malformed recipes" intent.

- source_spec: `spec-1-1-recipe-schema-and-loader.md`
  summary: String fields (`language.code`, `language.name`, `CorpusSpec.id`, `Candidate.name`, `scripts` items) accept empty strings, and `scripts` accepts an empty list; no ISO-639-1/ISO-15924 shape checks.
  evidence: Docstrings/comments promise "ISO 639-1" codes and "ISO 15924" scripts, but `code: str` accepts `""`/`"Polish"`/`"pl-PL"` and `scripts: list[str]` accepts `[]`. `min_length=1` on the strings/list and a light code/script format check would make the documented contract enforced.

- source_spec: `spec-1-1-recipe-schema-and-loader.md`
  summary: `CorpusSpec.config`/`split` are optional for all sources, so an `hf` corpus missing its required `config`/`split` is not caught at load time.
  evidence: Optionality was chosen so `url`/`local` sources validate, but an `hf` source almost always needs `config`/`split`; a `source`-conditional validator (require them for `hf`, allow absence for `url`/`local`) would surface a common latent error earlier.
