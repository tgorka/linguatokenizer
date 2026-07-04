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

- source_spec: `spec-1-2-corpus-acquisition-to-the-nextcloud-data-plane.md`
  summary: `sample_corpora` reuses the same `random.Random(seed)` instance/seed for every corpus in the mix rather than deriving a per-source seed.
  evidence: Two corpora with correlated or identical upstream content ordering (e.g. two dumps of the same dataset under different ids) would draw reservoir positions using the same pseudo-random sequence rather than independent draws. Low real-world likelihood given typical recipe corpora, but worth a `random.Random((seed, i))`-style fix if a future recipe pairs near-duplicate sources.

- source_spec: `spec-1-2-corpus-acquisition-to-the-nextcloud-data-plane.md`
  summary: `sample_corpora` is not atomic across sources — if `resolve_source` raises partway through the corpus list (e.g. a `local` path removed mid-run, a `url` 404), the function raises with no manifest written and no partial-progress record of already-resolved sources.
  evidence: For a foundations story establishing the reproducibility contract, a partial-failure manifest (or at least a wrapped error naming which source failed and what had already succeeded) would aid debugging long acquisition runs; current behavior is a bare exception with no recorded state.

- source_spec: `spec-1-2-corpus-acquisition-to-the-nextcloud-data-plane.md`
  summary: The `url` source cache (`{data_dir}/corpora/url/<hash>.txt`) has no invalidation strategy — no ETag/Last-Modified/content-hash check — so an upstream URL's content changing after the first fetch is silently never picked up.
  evidence: `test_url_repeated_fetch_reuses_cache` explicitly relies on this (deletes the source file and expects the stale cache to still serve). For a reproducibility-focused project this is a reasonable default (stable inputs = stable outputs) but is an undocumented limitation a future story or doc note should flag.

- source_spec: `spec-1-2-corpus-acquisition-to-the-nextcloud-data-plane.md`
  summary: `_iter_hf_examples`'s "first `str`-valued column" fallback (when `"text"` is absent) can silently pick a metadata/id/url string column instead of the real document text for datasets with multiple string columns.
  evidence: This is the spec-directed fallback (Boundaries explicitly call for it) and is fine for FLORES/Wikipedia/OSCAR-style single-text-column datasets, but a dataset shaped like `{"text": ..., "id": ...}` vs. one shaped `{"id": ..., "content": ...}` would silently ingest `"id"` as a document. Worth a `text_field` override or better heuristic (e.g. longest string value) when a real dataset triggers it.

- source_spec: `spec-1-2-corpus-acquisition-to-the-nextcloud-data-plane.md`
  summary: `_resolve_url`'s cache write (`cache_path.write_bytes(raw)`) is not atomic — a crash or disk-full event between opening and finishing the write can leave a truncated cache file that a later run would treat as valid.
  evidence: Standard fix is write-to-temp-then-`os.replace()`; low likelihood in practice (small corpora fixtures today) but worth hardening before real multi-GB corpus downloads land in later stories.

- source_spec: `spec-1-2-corpus-acquisition-to-the-nextcloud-data-plane.md`
  summary: `resolve_source`'s graceful-skip contract is only implemented for `hf` sources; a `url` 404/network failure or a `local` missing path always raises `SourceError` rather than being representable as a skip.
  evidence: Plausible a future story wants "skip a flaky/removed `url` source with a warning" the same way gated `hf` sources are skipped today; the two source types currently have asymmetric failure contracts with no comment marking the asymmetry as intentional vs. incidental.

- source_spec: `spec-1-2-corpus-acquisition-to-the-nextcloud-data-plane.md`
  summary: `sample_corpora` does not detect or reject duplicate `CorpusSpec` entries (same `source`+`id`) within a single `corpora` list, so a copy-paste mistake in a recipe silently double-counts that source's weight share.
  evidence: Weight normalization sums `spec.weight` across the raw input list with no uniqueness check; two identical entries (e.g. the same `hf` id pasted twice with different domains) would each get their own quota, over-representing that corpus in the mix without any error or warning. A cross-entry uniqueness validator (on `source`+`id`, mirroring the disjointness gap already deferred for `Recipe.train_corpora`/`eval_corpora`) would catch this at the recipe-authoring stage instead.

- source_spec: `spec-1-2-corpus-acquisition-to-the-nextcloud-data-plane.md`
  summary: `sample_corpora`'s manifest write (`manifest_path.write_text(...)`) is not atomic and not lock-protected, so a crash mid-write or two concurrent `sample_corpora` calls for the same `language_code` can leave `manifest.json` truncated or interleaved.
  evidence: Distinct from the already-deferred `url`-cache non-atomic write: this is the final provenance manifest itself, which downstream training/benchmarking treats as authoritative. A write-to-temp-then-`os.replace()` pattern (plus documenting that concurrent same-language_code runs are unsupported) would close this before real multi-source acquisition runs land in later stories.
