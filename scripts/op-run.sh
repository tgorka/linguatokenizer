#!/usr/bin/env bash
# Run a command with secrets injected from 1Password.
#
# Resolves the `op://…` references in op.env via `op run`, exposing them as
# environment variables for the wrapped command ONLY — secrets never touch
# disk and never persist past the child process.
#
# Usage:
#   scripts/op-run.sh <command> [args...]
#   OP_ENV_FILE=other.env scripts/op-run.sh <command> [args...]
#
# Graceful fallback: when the `op` CLI is missing, OP_SERVICE_ACCOUNT_TOKEN is
# unset, or the env file is absent (CI, offline dev, contributors without vault
# access), the command runs unwrapped so non-secret work (lint/test) still
# passes. Tasks that genuinely need a secret will fail loudly on their own.
set -euo pipefail

# Repo root is resolved from this script's location, so the wrapper works no
# matter the caller's working directory (root `task lb:*` or `cd` into a pkg).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="${OP_ENV_FILE:-$REPO_ROOT/op.env}"

if [ "$#" -eq 0 ]; then
  echo "[op-run] usage: scripts/op-run.sh <command> [args...]" >&2
  exit 2
fi

if command -v op >/dev/null 2>&1 \
  && [ -n "${OP_SERVICE_ACCOUNT_TOKEN:-}" ] \
  && [ -f "$ENV_FILE" ]; then
  exec op run --env-file="$ENV_FILE" -- "$@"
fi

echo "[op-run] 1Password unavailable (no op CLI / OP_SERVICE_ACCOUNT_TOKEN / ${ENV_FILE##*/}) — running without injected secrets" >&2
exec "$@"
