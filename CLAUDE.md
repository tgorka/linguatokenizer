# CLAUDE.md

**Read [`AGENT.md`](AGENT.md) — it is the canonical developer guide.** This
file is a quick pointer for coding agents.

## Quick reference

- Sync: `task setup` (or `task setup:all` for optional extras).
- Local gate before commit: `task check` (ruff lint + format check + ty + pytest).
- Pre-commit: `task precommit` (prek). CI mirrors both.
- Secrets: `task secrets:run -- <cmd>` (1Password `op.env`, vault `neuraffica`).

## Project shape

- One library: `packages/tokenizer-core/` (`corpora`, `train`, `benchmark`,
  `adapters`). First-party import root: `tokenizer_core`.
- Per-language recipes in `languages/<iso>/recipe.yaml` (Polish = `pl`).
- North Star metric: fertility (tokens/word) — `tokenizer_core.benchmark.fertility`.
- Corpora + large artifacts live on Nextcloud (`$TOKENIZER_DATA_DIR`), not git.

## Rules

- Always use `uv run`, never bare `python`.
- Heavy deps are extras, imported lazily behind guards.
- English only in committed docs/artifacts (repo may go public).
- Don't commit trained tokenizer binaries — commit the recipe.

## BMAD

Managed with BMAD; planning + implementation artifacts under `_bmad-output/`.
`bmad-loop` drives dev/review — run `task bmad:skills` once after clone to link
the BMM skills into `.claude/skills/`.
