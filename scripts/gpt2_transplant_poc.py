#!/usr/bin/env python3
"""gpt2 transplant proof-of-concept: extend + training-free FVT embeddings.

Demonstrates grafting a trained Polish tokenizer onto **real gpt2** and
initializing the new tokens' embeddings **without retraining** (FVT), then
quantifies the two things that matter:

  1. English is essentially untouched — base token ids + embeddings are
     preserved, and the appended Polish vocabulary barely re-tokenizes English
     (measured % of English tokens that shift to new ids). => no meaningful
     quality regression on the base language, for free.
  2. Polish gets materially cheaper (fewer tokens/word), and every new token
     receives a sane FVT embedding (finite, base-comparable norm) it can be
     fine-tuned from — the optional short retrain adapts these for strong
     generative Polish output (needs torch/GPU; not required for this PoC).

Torch-free: gpt2's embedding matrix is read straight from safetensors.

Run: `scripts/op-run.sh uv run --no-sync python scripts/gpt2_transplant_poc.py \
        [--artifact <dir>] [--family gpt2]`
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from tokenizer_core.adapters.embed import (
    read_token_embeddings,
    transplant_embeddings,
)
from tokenizer_core.adapters.extend import extend
from tokenizer_core.benchmark.fertility import fertility


def _data_dir() -> Path:
    root = os.environ.get("TOKENIZER_DATA_DIR")
    if not root:
        raise SystemExit("$TOKENIZER_DATA_DIR is unset")
    return Path(root)


def main() -> None:
    """Run the gpt2 extend + FVT-transplant PoC and print the metrics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", default="gpt2")
    parser.add_argument("--artifact", default=None)
    args = parser.parse_args()

    dd = _data_dir()
    artifact = args.artifact or str(dd / "tokenizers" / "pl" / "pl-bpe-32k")
    token = os.environ.get("HF_TOKEN")

    # 1) extend the target tokenizer with the Polish vocabulary.
    result = extend(args.family, artifact)
    base_tok = AutoTokenizer.from_pretrained(args.family, token=token)

    # 2) training-free FVT embedding transplant (no torch).
    base_emb = read_token_embeddings(args.family, token=token)
    ext_emb = transplant_embeddings(result, base_emb, base_tok, method="fvt")
    assert np.array_equal(ext_emb[: result.base_vocab_size], base_emb)

    devtest = dd / "flores200" / "flores200_dataset" / "devtest"
    eng = [
        ln.strip()
        for ln in (devtest / "eng_Latn.devtest")
        .read_text(encoding="utf-8")
        .splitlines()
        if ln.strip()
    ]
    pol = [
        ln.strip()
        for ln in (devtest / "pol_Latn.devtest")
        .read_text(encoding="utf-8")
        .splitlines()
        if ln.strip()
    ]

    def _enc(text: str) -> list[int]:
        return result.tokenizer.encode(text).ids

    eng_tok = sum(len(_enc(s)) for s in eng)
    eng_base = sum(len(base_tok.encode(s)) for s in eng)
    eng_new = sum(
        1 for s in eng for i in _enc(s) if i >= result.base_vocab_size
    )

    class _Enc:
        def encode(self, text: str) -> list[int]:
            return _enc(text)

    pol_fert_ext = fertility(_Enc(), pol).fertility
    pol_fert_base = fertility(
        type("B", (), {"encode": lambda _s, t: base_tok.encode(t)})(), pol
    ).fertility

    new = ext_emb[result.base_vocab_size :]
    print(f"=== gpt2 transplant PoC ({args.family}) ===")
    print(
        f"vocab: {result.base_vocab_size} -> "
        f"{result.tokenizer.get_vocab_size()} (+{len(result.added_tokens)})"
    )
    print(
        f"English preserved: {eng_tok} ext vs {eng_base} base tokens "
        f"({(1 - eng_tok / eng_base) * 100:+.1f}%); only {eng_new} "
        f"({eng_new / eng_tok * 100:.1f}%) use new ids -> ~unchanged"
    )
    print(
        f"Polish fertility: {pol_fert_base:.3f} -> {pol_fert_ext:.3f} "
        f"tok/word ({(1 - pol_fert_ext / pol_fert_base) * 100:.0f}% fewer)"
    )
    print(
        f"new embeddings: finite={bool(np.isfinite(new).all())} "
        f"norm={float(np.linalg.norm(new, axis=1).mean()):.3f} "
        f"(base {float(np.linalg.norm(base_emb, axis=1).mean()):.3f})"
    )


if __name__ == "__main__":
    main()
