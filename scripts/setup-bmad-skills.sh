#!/usr/bin/env bash
# Link the BMM dev/review skills the bmad-loop orchestrator drives into this
# project's .claude/skills/ from the machine-local BMad Method plugin cache.
#
# These skills ship with the `bmad-method` plugin (not installed into the repo),
# but `bmad-loop validate` and the spawned dev/review sessions resolve them by
# bare name from .claude/skills/. The symlinks are machine-specific (they point
# at a versioned plugin-cache path) and are gitignored — every developer runs
# this once after cloning + installing the plugin.
#
# Usage: scripts/setup-bmad-skills.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DST="$REPO_ROOT/.claude/skills"
SKILLS=(
    bmad-dev-auto
    bmad-review-adversarial-general
    bmad-review-edge-case-hunter
)

# Find the newest bmad-method plugin skills dir.
CACHE_GLOB="$HOME/.claude/plugins/cache/bmad-method/bmad"
SRC=""
if [ -d "$CACHE_GLOB" ]; then
    SRC="$(find "$CACHE_GLOB" -maxdepth 2 -type d -name skills 2>/dev/null | sort -V | tail -1)"
fi

if [ -z "$SRC" ]; then
    echo "error: bmad-method plugin skills not found under $CACHE_GLOB" >&2
    echo "       install the BMad Method plugin, then re-run this script." >&2
    exit 1
fi

mkdir -p "$DST"
for s in "${SKILLS[@]}"; do
    if [ ! -d "$SRC/$s" ]; then
        echo "warning: $s not present in $SRC — skipping" >&2
        continue
    fi
    ln -sfn "$SRC/$s" "$DST/$s"
    echo "linked $s -> $SRC/$s"
done
echo "done. run 'bmad-loop validate --project $REPO_ROOT' to verify."
