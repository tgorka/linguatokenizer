#!/usr/bin/env python3
"""Stage a bounded, reproducible Polish training corpus into the data plane.

Tokenizer training only needs a few hundred MB of text for a 32k vocab, so we
sample a fixed byte budget from ungated Hugging Face sources and write it as
plain one-document-per-line text under ``$TOKENIZER_DATA_DIR/pl/train/``. This
keeps training fast, offline, and reproducible (no gated/streaming flakiness in
the dev loop), and keeps the large data off git.

Sources (all ungated, ``text`` column):
  - HuggingFaceFW/fineweb-2 [pol_Latn]  — high-quality filtered web (primary)
  - wikimedia/wikipedia [20231101.pl]   — clean encyclopedic text

Run: ``scripts/op-run.sh uv run python scripts/stage_pl_corpus.py``
Idempotent: skips a target file that already meets its byte budget.
"""

from __future__ import annotations

import os
from pathlib import Path

from datasets import load_dataset  # ty: ignore[unresolved-import]

# (dataset id, config, target bytes) — tune budgets here.
_SOURCES = [
    ("HuggingFaceFW/fineweb-2", "pol_Latn", 100 * 1024 * 1024, "fineweb2.txt"),
    ("wikimedia/wikipedia", "20231101.pl", 50 * 1024 * 1024, "wiki.txt"),
]


def _data_dir() -> Path:
    root = os.environ.get("TOKENIZER_DATA_DIR")
    if not root:
        raise SystemExit("$TOKENIZER_DATA_DIR is unset")
    return Path(root)


def _stage_one(dataset_id: str, config: str, budget: int, out: Path) -> None:
    if out.is_file() and out.stat().st_size >= budget * 0.98:
        print(f"skip {out.name}: already {out.stat().st_size / 1e6:.0f} MB")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(dataset_id, name=config, split="train", streaming=True)
    written = 0
    docs = 0
    with out.open("w", encoding="utf-8") as fh:
        for example in ds:
            text = example.get("text")
            if not isinstance(text, str):
                continue
            line = " ".join(text.split())  # collapse whitespace -> one line
            if not line:
                continue
            fh.write(line + "\n")
            written += len(line) + 1
            docs += 1
            if written >= budget:
                break
    print(f"wrote {out} — {docs} docs, {written / 1e6:.1f} MB")


def main() -> None:
    """Stage every configured source into ``$TOKENIZER_DATA_DIR/pl/train``."""
    dest = _data_dir() / "pl" / "train"
    for dataset_id, config, budget, name in _SOURCES:
        _stage_one(dataset_id, config, budget, dest / name)


if __name__ == "__main__":
    main()
