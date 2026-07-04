# AGENT.md — developer & agent guide

Canonical reference for working in this repo (humans and coding agents).

## What this project is

Token-efficient, morphology-aware tokenizers for under-served languages
(Polish first) + adapters that graft them onto existing LLM tokenizers. The
objective metric is **fertility** (tokens/word) on held-out, sentence-aligned
text: bring Polish to ≤ the English baseline with no quality regression. See
`README.md` for the "why" and `languages/pl/README.md` for the Polish plan.

## Layout

- `packages/tokenizer-core/` — the one library (uv workspace member). Modules:
  - `corpora/` — acquire, sample, mix training/benchmark text.
  - `train/` — train candidate tokenizers (BPE / Unigram / SuperBPE).
  - `benchmark/` — fertility, compression, quality parity metrics.
  - `adapters/` — graft a tokenizer onto a model (extend/merge or replace) with
    quality-preserving embedding init (OMP training-free; gpt2 retrain PoC).
- `languages/<iso>/recipe.yaml` — per-language source of truth. `_template/` to
  add one. Trained tokenizer binaries are build artifacts (Nextcloud), not git.
- `results/` — benchmark outputs, committed for provenance.
- `_bmad-output/` — BMAD planning + implementation artifacts.

## Tooling

| Tool | Role |
|------|------|
| `uv` | Package/venv manager. `uv sync --all-packages`. Python ≥ 3.11. |
| `task` (go-task) | Task runner. `task --list`. |
| `ruff` | Lint + format (Google style, 80 col). |
| `ty` | Type checker (Astral). |
| `pytest` | Tests (`packages/**/tests`). |
| `prek` | Pre-commit runner (`uvx prek run --all-files`). |
| `op` (1Password) | Secret injection via `op.env`. |

## Common workflows

```sh
task setup            # sync workspace
task setup:all        # + all optional extras (train/corpora/adapters)
task check            # lint + format check + types + tests (local gate)
task fmt              # auto-format + safe fixes
task precommit:install && task precommit
```

Add a language: copy `languages/_template/` → `languages/<iso>/`, fill
`recipe.yaml`, acquire corpora, train + benchmark, iterate.

## Secrets (1Password)

`op.env` holds only `op://neuraffica/<item>/credential` references (safe to
commit) plus non-secret literals. `scripts/op-run.sh` resolves them for a child
process only (secrets never touch disk); it **skips gracefully** when `op` /
`OP_SERVICE_ACCOUNT_TOKEN` / `op.env` are absent, so lint/test still run.

```sh
task secrets:check                       # verify refs resolve (values masked)
task secrets:run -- uv run python foo.py # run a command with secrets injected
```

Vault items used: `hf-token` (HF corpora/model downloads), `wandb-api-key`
(tracking), `hyperstack-*` (remote GPU for the gpt2 PoC only). Non-secret
literals in `op.env`: `TOKENIZER_DATA_DIR` (Nextcloud data path), model/backend
pins.

## Data store

Real corpora and large tokenizer artifacts live on the mounted Nextcloud data
store at `$TOKENIZER_DATA_DIR` (default `/home/dev/nextcloud/data/tokenizer`),
**not** in git. The repo keeps recipes, code, tests, and small benchmark
result files. `data/` and `datasets/` are gitignored (keep `.gitkeep`).

## Remote GPU (optional — gpt2 PoC only)

The core deliverables (tokenizer training, fertility benchmarking, training-free
OMP transplant) need **no GPU**. Only the gpt2 retrain proof-of-concept does.
`scripts/gpu-run.sh` rsyncs the repo to the Hyperstack VM, runs a command on the
GPU, rsyncs `results/` back; wrapped in `op-run.sh`. Skips until
`HYPERSTACK_SSH_HOST` resolves. `task gpu:run -- <cmd>`.

## BMAD & bmad-loop

This repo is managed with the BMAD method. Planning artifacts (brief, PRD,
architecture, epics, stories) live in `_bmad-output/planning-artifacts/`;
implementation artifacts (sprint status, story files) in
`_bmad-output/implementation-artifacts/`. Config in `_bmad/` (English output).

The `bmad-loop` orchestrator drives dev/review sessions. It needs the BMM
dev/review skills at `.claude/skills/` — recreate the machine-local symlinks
after cloning with `task bmad:skills` (they're gitignored). Then
`bmad-loop validate --project .` and `bmad-loop run`.

## Pre-submit gates

Local (`task check`) → pre-commit (`task precommit`) → CI
(`.github/workflows/ci.yml`: Python checks + prek, both must pass).

## Conventions & pitfalls

- Always go through `uv run` (never a bare `python`) so the workspace venv +
  pins apply.
- Kebab-case directories, snake_case modules. First-party import root is
  `tokenizer_core`.
- Heavy deps (`tokenizers`, `datasets`, `transformers`, `numpy`) are **extras**
  (`train`/`corpora`/`adapters`); base install stays light. Runtime code that
  needs them imports lazily and guards the import.
- Keep trained tokenizer binaries out of git — commit the recipe, store the
  binary on Nextcloud.
- English only in committed docs/artifacts (the repo may be made public).
