#!/usr/bin/env python3
"""Cross-model old-vs-new tokenizer benchmark (token counts + encode speed).

For each target model, compares its **stock** tokenizer against our Polish
tokenizer applied two ways — **extend** (graft Polish vocab, keep base ids) and
**replace** (our standalone tokenizer) — on the aligned FLORES Polish sentences.
Reports, per model:

- tokens/sentence (stock / extended / replace) and the % token reduction. Cost
  scales linearly with tokens, so the reduction % is also the cost-savings % on
  both the input and output legs (see `docs/datacenter-optimizer.md`).
- encode speed (tokens/sec) for stock vs our tokenizer — confirming the
  efficiency win doesn't cost tokenization throughput.

Full inference wall-clock (prefill/decode) needs a served model on GPU and is
projected analytically by `tokenizer_core.economics`; this is CPU-only.

Writes results/cross-model-benchmark.{json,md}. Run:
`scripts/op-run.sh uv run --no-sync python scripts/cross_model_benchmark.py`
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from tokenizers import Tokenizer
from transformers import AutoTokenizer

from tokenizer_core.adapters.extend import extend
from tokenizer_core.adapters.registry import AdapterError
from tokenizer_core.benchmark.timing import encode_timing

# (registry family, HF id)
_MODELS = [
    ("gpt2", "gpt2"),
    ("llama", "meta-llama/Llama-3.1-8B"),
    ("mistral", "mistralai/Mistral-7B-v0.1"),
    ("qwen", "Qwen/Qwen2.5-7B"),
    ("deepseek", "deepseek-ai/DeepSeek-V3"),
]


def _lines(path: Path) -> list[str]:
    return [
        ln.strip()
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]


def main() -> None:
    """Run the cross-model benchmark and write the report."""
    dd = Path(os.environ["TOKENIZER_DATA_DIR"])
    token = os.environ.get("HF_TOKEN")
    pol = _lines(
        dd / "flores200" / "flores200_dataset" / "devtest" / "pol_Latn.devtest"
    )
    n = len(pol)
    artifact = str(dd / "tokenizers" / "pl" / "pl-bpe-32k")

    # Our standalone (replace) tokenizer — model-independent.
    ours = Tokenizer.from_file(f"{artifact}/tokenizer.json")

    class _Ours:
        def encode(self, text: str) -> list[int]:
            return ours.encode(text).ids

    ours_tok = sum(len(ours.encode(s).ids) for s in pol)
    ours_time = encode_timing(_Ours(), pol)

    rows = {}
    for family, hf_id in _MODELS:
        entry: dict[str, object] = {"hf_id": hf_id}
        try:
            stock = AutoTokenizer.from_pretrained(hf_id, token=token)
        except Exception as err:
            rows[family] = {"error": str(err)[:100]}
            continue

        class _Stock:
            def __init__(self, tk: object) -> None:
                self._tk = tk

            def encode(self, text: str) -> list[int]:
                return self._tk.encode(text, add_special_tokens=False)

        stock_enc = _Stock(stock)
        stock_tok = sum(
            len(stock.encode(s, add_special_tokens=False)) for s in pol
        )
        entry["stock_tok_per_sent"] = round(stock_tok / n, 2)
        entry["replace_tok_per_sent"] = round(ours_tok / n, 2)
        entry["replace_reduction_pct"] = round(
            (1 - ours_tok / stock_tok) * 100, 1
        )
        entry["stock_tokens_per_sec"] = round(
            encode_timing(stock_enc, pol).tokens_per_sec
        )
        entry["ours_tokens_per_sec"] = round(ours_time.tokens_per_sec)

        # extend path (graft Polish vocab onto the stock tokenizer).
        try:
            ext = extend(family, artifact)
            ext_tok = sum(len(ext.tokenizer.encode(s).ids) for s in pol)
            entry["extend_tok_per_sent"] = round(ext_tok / n, 2)
            entry["extend_reduction_pct"] = round(
                (1 - ext_tok / stock_tok) * 100, 1
            )
        except AdapterError as err:
            entry["extend_tok_per_sent"] = None
            entry["extend_note"] = f"not supported: {str(err)[:60]}"
        rows[family] = entry

    out = {
        "eval": "FLORES-200 devtest pol_Latn (1012 sentences)",
        "our_tokenizer": "pl-bpe-32k",
        "note": (
            "reduction % = cost-savings % (cost is linear in tokens, input + "
            "output legs). encode speed is CPU tokenizer throughput, not "
            "inference latency."
        ),
        "models": rows,
    }
    results = Path("results")
    (results / "cross-model-benchmark.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_md(results / "cross-model-benchmark.md", out)
    print("wrote results/cross-model-benchmark.{json,md}")
    for fam, r in rows.items():
        if "error" not in r:
            print(
                f"{fam}: stock {r['stock_tok_per_sent']} -> extend "
                f"{r.get('extend_tok_per_sent')} / replace "
                f"{r['replace_tok_per_sent']} tok/sent "
                f"(replace -{r['replace_reduction_pct']}%)"
            )


def _write_md(path: Path, out: dict) -> None:
    rows = []
    for fam, r in out["models"].items():
        if "error" in r:
            rows.append(f"| {fam} | skipped: {r['error']} | | | | |")
            continue
        ext = r.get("extend_tok_per_sent")
        ext_cell = f"{ext} (−{r['extend_reduction_pct']}%)" if ext else "n/a"
        rows.append(
            f"| {fam} | {r['stock_tok_per_sent']} | {ext_cell} | "
            f"{r['replace_tok_per_sent']} (−{r['replace_reduction_pct']}%) | "
            f"{r['stock_tokens_per_sec']:,} | {r['ours_tokens_per_sec']:,} |"
        )
    lines = [
        "# Cross-model tokenizer benchmark (Polish)",
        "",
        f"Generated by `scripts/cross_model_benchmark.py` on {out['eval']}. "
        "Tokens/sentence for each model's **stock** tokenizer vs our Polish "
        "tokenizer applied by **extend** (graft, base ids kept) and **replace** "
        "(standalone). Reduction % = cost-savings % (cost is linear in tokens).",
        "",
        "| Model | Stock tok/sent | Extend | Replace | Stock enc tok/s | Ours enc tok/s |",
        "|-------|---------------:|--------|---------|----------------:|---------------:|",
        *rows,
        "",
        "Encode speed is CPU tokenizer throughput (not inference latency — that "
        "needs a served model on GPU, projected by `tokenizer_core.economics`). "
        "`replace` gives the largest token cut (full standalone efficiency) but "
        "needs an embedding transplant + short retrain; `extend` keeps English "
        "intact for a smaller Polish cut. Full data in "
        "`cross-model-benchmark.json`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
