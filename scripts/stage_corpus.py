#!/usr/bin/env python3
"""Stage a bounded, reproducible training corpus for ANY language.

The language-agnostic generalization of ``stage_pl_corpus.py``: samples a byte
budget from ungated Hugging Face sources into
``$TOKENIZER_DATA_DIR/<lang>/train/*.txt`` so a per-language tokenizer can be
trained fast, offline, and reproducibly. Large data stays off git.

Run, e.g. German and Russian::

    scripts/op-run.sh uv run --no-sync python scripts/stage_corpus.py \
        --lang de --fineweb deu_Latn --wiki 20231101.de --mb 100
    scripts/op-run.sh uv run --no-sync python scripts/stage_corpus.py \
        --lang ru --fineweb rus_Cyrl --wiki 20231101.ru --mb 100

Idempotent: skips a target file that already meets its byte budget.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from datasets import load_dataset


def _data_dir() -> Path:
    root = os.environ.get("TOKENIZER_DATA_DIR")
    if not root:
        raise SystemExit("$TOKENIZER_DATA_DIR is unset")
    return Path(root)


def _stage(dataset_id: str, config: str, budget: int, out: Path) -> None:
    if out.is_file() and out.stat().st_size >= budget * 0.98:
        print(f"skip {out.name}: already {out.stat().st_size / 1e6:.0f} MB")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(dataset_id, name=config, split="train", streaming=True)
    written = docs = 0
    with out.open("w", encoding="utf-8") as fh:
        for example in ds:
            text = example.get("text")
            if not isinstance(text, str):
                continue
            line = " ".join(text.split())
            if not line:
                continue
            fh.write(line + "\n")
            written += len((line + "\n").encode("utf-8"))  # UTF-8 bytes
            docs += 1
            if written >= budget:
                break
    print(f"wrote {out} — {docs} docs, {written / 1e6:.1f} MB")


def main() -> None:
    """Stage FineWeb-2 + Wikipedia text for one language."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True, help="ISO 639-1 code, e.g. de")
    parser.add_argument(
        "--fineweb", required=True, help="FineWeb-2 config, e.g. deu_Latn"
    )
    parser.add_argument(
        "--wiki",
        required=True,
        help="wikimedia/wikipedia config, e.g. 20231101.de",
    )
    parser.add_argument(
        "--mb", type=int, default=100, help="total MB budget (70/30 web/wiki)"
    )
    args = parser.parse_args()

    dest = _data_dir() / args.lang / "train"
    budget = args.mb * 1024 * 1024
    _stage(
        "HuggingFaceFW/fineweb-2",
        args.fineweb,
        int(budget * 0.7),
        dest / "fineweb2.txt",
    )
    _stage(
        "wikimedia/wikipedia",
        args.wiki,
        int(budget * 0.3),
        dest / "wiki.txt",
    )


if __name__ == "__main__":
    main()
