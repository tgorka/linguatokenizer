#!/usr/bin/env bash
# Remote GPU execution — rsync the repo to the Hyperstack GPU VM, run a command
# on the GPU there, then rsync results/ back.
#
# WHY THIS EXISTS
#   The served plane (generation/judge) hits an OpenAI-compatible *endpoint*
#   (scripts/op-run.sh + HYPERSTACK_API_KEY). But the in-process GPU spikes —
#   Sp2 (HF forward-hook extraction), Sp5 (KL noise floor), Sp8 (embedding
#   messaging), Sp9 (LoRA probe) — and the NDIF round-trip (Sp3) must *run on*
#   a GPU host, not call an endpoint. This is their transport: "log in (SSH) ->
#   run on cloud GPU -> download results locally". Sp7 uses the endpoint and
#   does NOT need this wrapper.
#
# SECRETS / CONFIG (resolved from op.env by `op run`; call via `task gpu:run`,
# which wraps this in scripts/op-run.sh so the vault values exist as env vars):
#   HYPERSTACK_SSH_HOST       (required) VM host or IP. The "CHANGE-ME"
#                             placeholder counts as unset -> graceful skip.
#   HYPERSTACK_SSH_USER       (default: ubuntu)
#   HYPERSTACK_SSH_PORT       (default: 22)
#   HYPERSTACK_SSH_KEY        PEM private-key *material* (an op:// ref). Written
#                             to a 0600 temp file for this session only, then
#                             removed. Omit to fall back to your SSH agent / the
#                             default key.
#   HYPERSTACK_SSH_KEY_FILE   Alternative to HYPERSTACK_SSH_KEY: path to an
#                             existing private key on disk.
#   HYPERSTACK_REMOTE_DIR     (default: ~/tokenizer) repo checkout on the VM.
#   HYPERSTACK_SYNC_EXTRAS    (default: "") space-separated uv extras to install
#                             on the VM before the command, e.g. "models" or
#                             "models serving". Each becomes `--extra <x>`.
#   HYPERSTACK_SYNC_PACKAGE   (default: "") uv workspace member that owns those
#                             extras; becomes `--package <pkg>`. Required when
#                             the extras live on a member, not the workspace root
#                             (else `uv sync --extra X` errors "not defined").
#   HYPERSTACK_UV_TORCH_BACKEND (default: "auto") UV_TORCH_BACKEND for the remote
#                             `uv sync` — picks the torch wheel matching the VM's
#                             CUDA driver. "auto" detects it; else torch may fall
#                             back to CPU. Mirrors serve.sh.
#
# Secrets forwarded to the remote command (only those set locally): HF_TOKEN,
# WANDB_API_KEY, WANDB_MODE, WANDB_PROJECT, HYPERSTACK_USD_PER_GPU_HOUR. They
# are written to a 0600 remote env file and `source`d for the command only —
# never placed on the argv, so they never appear in the remote process list.
#
# USAGE
#   scripts/gpu-run.sh [--dry-run] -- <command run in the repo dir on the VM>
#   task gpu:run -- uv run python -m language_benchmark.spikes.activation_extraction --live
#   task gpu:run -- --dry-run -- uv run python -m language_benchmark eval
#
# OBSERVABILITY
#   Every invocation appends one line to results/gpu-runs.log (UTC timestamp,
#   masked host, remote dir, exit code, duration, command) so a failed run is
#   visible after the fact. Per-run token/cost records still come back inside
#   results/runs.jsonl (written by the spike's track_run on the VM, rsynced
#   home). On a non-zero remote exit the remote stderr tail is printed and the
#   same code is propagated.
#
# GRACEFUL SKIP (exit 0): when HYPERSTACK_SSH_HOST is unset / still the
#   placeholder, or `ssh`/`rsync` are missing. Keeps `task` chains and CI green
#   until SSH creds are provisioned in the vault. This mirrors scripts/op-run.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# Harness-level log (overridable for tests). The per-run token/cost records
# live in results/runs.jsonl; this is just "what ran remotely + did it work".
LOG_FILE="${GPU_RUN_LOG:-$REPO_ROOT/results/gpu-runs.log}"

DRY_RUN=0
# Accept a leading `--dry-run` and an optional `--` separator before the command.
while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --) shift; break ;;
    -*) echo "[gpu-run] unknown flag: $1" >&2; exit 2 ;;
    *) break ;;
  esac
done

if [ "$#" -eq 0 ]; then
  echo "[gpu-run] usage: scripts/gpu-run.sh [--dry-run] -- <command...>" >&2
  exit 2
fi
REMOTE_CMD=("$@")

SSH_HOST="${HYPERSTACK_SSH_HOST:-}"
SSH_USER="${HYPERSTACK_SSH_USER:-ubuntu}"
SSH_PORT="${HYPERSTACK_SSH_PORT:-22}"
REMOTE_DIR="${HYPERSTACK_REMOTE_DIR:-~/tokenizer}"
SYNC_EXTRAS="${HYPERSTACK_SYNC_EXTRAS:-}"
# In a uv workspace, `--extra` lives on a member package, not the root — so
# `uv sync --extra models` from the repo root errors "Extra not defined". Scope
# the sync to the member that defines the extras.
SYNC_PACKAGE="${HYPERSTACK_SYNC_PACKAGE:-}"
# torch CUDA build to match the VM driver; "auto" lets uv detect it (mirrors
# serve.sh). Override via HYPERSTACK_UV_TORCH_BACKEND, e.g. "cu128".
TORCH_BACKEND="${HYPERSTACK_UV_TORCH_BACKEND:-auto}"

# --- Graceful skip ----------------------------------------------------------
skip() {
  echo "[gpu-run] $1 — skipping remote run (nothing executed)." >&2
  echo "    Provision the SSH vault items, set HYPERSTACK_SSH_HOST in op.env," >&2
  echo "    then re-run. See AGENT.md -> 'Remote GPU execution'." >&2
  exit 0
}
[ -z "$SSH_HOST" ] && skip "HYPERSTACK_SSH_HOST unset"
case "$SSH_HOST" in *CHANGE-ME*) skip "HYPERSTACK_SSH_HOST is still the placeholder";; esac
# Tool presence only matters for a real run; --dry-run just prints the plan so
# you can preview it on any box (e.g. CI without rsync installed).
if [ "$DRY_RUN" -eq 0 ]; then
  command -v ssh >/dev/null 2>&1 || skip "ssh not found"
  command -v rsync >/dev/null 2>&1 || skip "rsync not found"
fi

# --- SSH key handling -------------------------------------------------------
# Prefer explicit key material (op:// ref) -> 0600 temp file; else a key file;
# else the agent/default key. Temp artifacts are removed on any exit.
KEY_FILE="${HYPERSTACK_SSH_KEY_FILE:-}"
REMOTE_ENV_FILE=""
cleanup() {
  [ -n "${_TMP_KEY:-}" ] && rm -f "$_TMP_KEY" 2>/dev/null || true
  [ -n "${_TMP_ENV:-}" ] && rm -f "$_TMP_ENV" 2>/dev/null || true
}
trap cleanup EXIT

# A PEM key whose line breaks were flattened (some secret managers / op fields
# deliver the key on a single line) makes OpenSSH fail with "error in
# libcrypto". Restore the canonical layout — BEGIN marker / 64-col base64 body /
# END marker — when no newline is present; a key that already has newlines is
# trusted as-is.
write_pem_key() {
  local key="$1" dest="$2"
  case "$key" in
    *$'\n'*) printf '%s\n' "$key" > "$dest"; return ;;
  esac
  local hdr ftr body
  hdr="$(printf '%s' "$key" | grep -oE -- '^-----BEGIN [A-Z0-9 ]+-----')"
  ftr="$(printf '%s' "$key" | grep -oE -- '-----END [A-Z0-9 ]+-----$')"
  if [ -z "$hdr" ] || [ -z "$ftr" ]; then
    printf '%s\n' "$key" > "$dest"  # not a PEM we recognize — write verbatim
    return
  fi
  body="$(printf '%s' "$key" \
    | sed -E 's/^-----BEGIN [A-Z0-9 ]+-----//; s/-----END [A-Z0-9 ]+-----$//' \
    | tr -d ' \t\r\n')"
  { printf '%s\n' "$hdr"; printf '%s' "$body" | fold -w 64; \
    printf '\n%s\n' "$ftr"; } > "$dest"
}

if [ -n "${HYPERSTACK_SSH_KEY:-}" ]; then
  _TMP_KEY="$(mktemp)"
  chmod 600 "$_TMP_KEY"
  write_pem_key "$HYPERSTACK_SSH_KEY" "$_TMP_KEY"
  KEY_FILE="$_TMP_KEY"
fi

SSH_OPTS=(-p "$SSH_PORT" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15)
[ -n "$KEY_FILE" ] && SSH_OPTS+=(-i "$KEY_FILE")
TARGET="${SSH_USER}@${SSH_HOST}"
MASKED_TARGET="${SSH_USER}@${SSH_HOST}:${SSH_PORT}"

# --- Forward a curated allowlist of secrets via a 0600 remote env file ------
ENV_KEYS=(HF_TOKEN WANDB_API_KEY WANDB_MODE WANDB_PROJECT HYPERSTACK_USD_PER_GPU_HOUR)
_TMP_ENV="$(mktemp)"; chmod 600 "$_TMP_ENV"
for k in "${ENV_KEYS[@]}"; do
  v="${!k:-}"
  [ -n "$v" ] && printf 'export %s=%q\n' "$k" "$v" >> "$_TMP_ENV"
done
REMOTE_ENV_FILE=".gpu-run.env"

# --- Build the extras install + remote command ------------------------------
EXTRA_FLAGS=""
[ -n "$SYNC_PACKAGE" ] && EXTRA_FLAGS+=" --package $SYNC_PACKAGE"
for x in $SYNC_EXTRAS; do EXTRA_FLAGS+=" --extra $x"; done
printf -v CMD_STR '%q ' "${REMOTE_CMD[@]}"
REMOTE_SCRIPT="set -euo pipefail
# A non-interactive ssh shell does not source the profile that puts uv on PATH
# (uv installs to ~/.local/bin), so 'uv' is otherwise 'command not found'.
export PATH=\"\$HOME/.local/bin:\$HOME/.cargo/bin:\$PATH\"
cd ${REMOTE_DIR}
# Shred the forwarded-secret env file on ANY exit (incl. a failing uv sync /
# command under set -e), so credentials never linger on the VM.
trap 'rm -f ${REMOTE_ENV_FILE}' EXIT
set -a; [ -f ${REMOTE_ENV_FILE} ] && . ./${REMOTE_ENV_FILE}; set +a
# Pick the torch wheel that matches the VM's CUDA *driver* (auto-detect), else
# uv pulls the newest CUDA build and torch silently falls back to CPU when the
# driver is older — same pin serve.sh uses for vLLM (UV_TORCH_BACKEND).
export UV_TORCH_BACKEND=${TORCH_BACKEND}
uv sync${EXTRA_FLAGS}
# uv sync resolves/prunes torch to the lockfile, which can leave a CUDA build
# WITHOUT its bundled cudnn (observed on the L40: torch 2.12.0, missing
# libcudnn.so.9 -> the in-process run dies). UV_TORCH_BACKEND alone does not fix
# this for \`uv sync\`. When a CUDA backend is pinned (cuXXX), repair torch from
# the matching PyTorch index so the run gets a CUDA wheel WITH cudnn.
case ${TORCH_BACKEND} in
  cu[0-9]*) uv pip install --reinstall-package torch --index-url https://download.pytorch.org/whl/${TORCH_BACKEND} torch ;;
esac
${CMD_STR}"

RSYNC_EXCLUDES=(--exclude '.git' --exclude '.venv' --exclude '__pycache__'
  --exclude '.ruff_cache' --exclude '.pytest_cache' --exclude '*.egg-info'
  --exclude 'dist' --exclude 'build' --exclude '.gpu-run.env')

run_or_echo() {
  if [ "$DRY_RUN" -eq 1 ]; then
    printf '  %s\n' "$*"
  else
    "$@"
  fi
}

echo "[gpu-run] target=${MASKED_TARGET} dir=${REMOTE_DIR} extras='${SYNC_EXTRAS}'" >&2
[ "$DRY_RUN" -eq 1 ] && echo "[gpu-run] --dry-run: printing the plan, executing nothing." >&2

# Each step `|| return $?` so the pipeline STOPS at the first failure (a `{ … }`
# group on the left of `||` disables set -e, which would otherwise let a failed
# push fall through to the ssh/pull and mask the real error).
remote_pipeline() {
  # 1) push the repo (code only; results come back in step 4)
  run_or_echo rsync -az --delete "${RSYNC_EXCLUDES[@]}" \
    -e "ssh ${SSH_OPTS[*]}" "$REPO_ROOT/" "${TARGET}:${REMOTE_DIR}/" || return $?
  # 2) push the forwarded-secret env file (0600) so it is sourced, not on argv
  run_or_echo rsync -az -e "ssh ${SSH_OPTS[*]}" \
    "$_TMP_ENV" "${TARGET}:${REMOTE_DIR}/${REMOTE_ENV_FILE}" || return $?
  # 3) run on the GPU
  run_or_echo ssh "${SSH_OPTS[@]}" "$TARGET" "$REMOTE_SCRIPT" || return $?
  # 4) pull results/ home
  run_or_echo rsync -az -e "ssh ${SSH_OPTS[*]}" \
    "${TARGET}:${REMOTE_DIR}/results/" "$REPO_ROOT/results/" || return $?
}

start=$(date -u +%s)
status=0
remote_pipeline || status=$?
end=$(date -u +%s)

# --- Observability: one harness-level log line, always ----------------------
mkdir -p "$(dirname "$LOG_FILE")"
ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '%s host=%s dir=%s exit=%s dur=%ss cmd="%s"\n' \
  "$ts" "$MASKED_TARGET" "$REMOTE_DIR" "$status" "$((end - start))" "${REMOTE_CMD[*]}" \
  >> "$LOG_FILE"

if [ "$status" -ne 0 ]; then
  echo "[gpu-run] remote run FAILED (exit ${status}). See ${LOG_FILE##*/} and the" >&2
  echo "    remote stderr above; results/runs.jsonl (rsynced back) records the" >&2
  echo "    spike's own ok=false trace when it reached track_run." >&2
fi
exit "$status"
