#!/usr/bin/env bash
# Fetch the FLORES-200 sentence-aligned evaluation set into the data plane.
#
# FLORES-200 (CC-BY-SA-4.0) is the aligned multilingual dev/devtest set used to
# measure fertility on *identical content* across languages — the fair way to
# state "Polish costs Nx English for the same sentence". It is a small (~25 MB)
# public download, so unlike training corpora it is safe to (re-)fetch on demand.
#
# Idempotent: skips the download/extract if the target files already exist.
# Writes to $TOKENIZER_DATA_DIR/flores200 (default /home/dev/nextcloud/data/tokenizer).
set -euo pipefail

DATA_DIR="${TOKENIZER_DATA_DIR:-/home/dev/nextcloud/data/tokenizer}"
DEST="$DATA_DIR/flores200"
URL="https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
MARKER="$DEST/flores200_dataset/devtest/pol_Latn.devtest"

if [ -f "$MARKER" ]; then
    echo "FLORES-200 already present at $DEST"
    exit 0
fi

mkdir -p "$DEST"
echo "downloading FLORES-200 (CC-BY-SA-4.0) -> $DEST ..."
# -f: fail fast (non-zero) on HTTP 4xx/5xx instead of saving an error page that
# then dies with a confusing `tar: not in gzip format`.
curl -fsSL -o "$DEST/flores200_dataset.tar.gz" "$URL"
tar xzf "$DEST/flores200_dataset.tar.gz" -C "$DEST"
echo "extracted. pol/eng devtest:"
wc -l "$DEST/flores200_dataset/devtest/pol_Latn.devtest" \
      "$DEST/flores200_dataset/devtest/eng_Latn.devtest"
