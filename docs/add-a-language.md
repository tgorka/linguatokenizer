# Adding a language

The pipeline is recipe-driven and script-agnostic (Latin, Cyrillic, Devanagari,
Arabic, Thai, CJK — anything FLORES-200 covers). Proven on Polish, German
(Latin) and Russian (Cyrillic); Russian cuts tokens −54% to ~1.14× English.

## Steps

1. **Recipe** — copy `languages/_template/` to `languages/<iso639-1>/` and set
   the FineWeb-2 config, Wikipedia config, and FLORES config for the language.
   Local staged paths use `${TOKENIZER_DATA_DIR}/<iso>/train/*.txt`.

2. **Eval data** — `task data:flores` (once) stages all FLORES-200 languages.

3. **Training corpus** — stage a bounded sample (language-agnostic):
   ```bash
   scripts/op-run.sh uv run --no-sync python scripts/stage_corpus.py \
     --lang <iso> --fineweb <iso3_Script> --wiki <YYYYMMDD.iso> --mb 60
   ```

4. **Train + benchmark** end to end (writes `results/multilang-ours.{json,md}`):
   ```bash
   scripts/op-run.sh uv run --no-sync python scripts/train_benchmark_lang.py \
     --lang <iso> --candidate <iso>-bpe-32k
   ```

5. **Adapt onto a model** (optional) — `adapters.extend`/`replace` +
   `transplant_embeddings`, exactly as for Polish (see `docs/usage.md`).

## Byte-dense scripts (Devanagari, Thai, ...)

For scripts whose UTF-8 codepoints are 2–3 bytes and that the GPT-2 word-split
regex over-fragments — abugidas/abjads and scripts without word spaces
(Devanagari, Thai) — set **`split_words: false`** on the BPE candidate in the
recipe. The default regex caps these at ~1.5 chars/token; disabling it lets BPE
merge the sequences to ~4 chars/token, bringing them from a ~3–4× penalty to
parity (Hindi 3.05×→1.04×, Thai 1.99×→0.91×). Trade-off: without whitespace
splitting, training memory grows fast — stage a smaller `--mb` and train with a
lower `--n-docs` for these (Thai has no spaces at all, so keep it small).

## Config cheat-sheet

FineWeb-2 uses `<iso639-3>_<Script>` (e.g. `deu_Latn`, `rus_Cyrl`, `hin_Deva`,
`arb_Arab`, `tha_Thai`, `zho_Hans`); Wikipedia uses `<YYYYMMDD>.<iso639-1>`
(e.g. `20231101.de`); FLORES uses the FineWeb-style config. The cross-language
penalty landscape (`results/multilang-penalty.md`) shows which languages to
prioritize — non-Latin scripts (Hindi 4.3×, Thai 3.9×, Hebrew 2.9×) benefit
most.

## Backlog languages

Recipes ready to add per the roadmap: fr, es, pt, cs, uk, lt, zh, hi, he, ar,
th — each a copy-and-fill using the configs above.
