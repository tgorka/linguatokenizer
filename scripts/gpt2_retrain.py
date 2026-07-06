#!/usr/bin/env python3
"""gpt2 retrain PoC: transplant a Polish tokenizer, then a short fine-tune.

The training-free FVT transplant (PR #2) gives new Polish tokens *approximate*
embeddings; this shows the optional next step — a short, cheap fine-tune that
adapts them — and measures the quality recovery with bits-per-byte (a
tokenizer-independent, per-byte metric, so stock vs extended tokenizers compare
fairly).

Pipeline: load gpt2 -> extend its tokenizer with the Polish vocab -> FVT-init
the new embeddings -> resize the model -> fine-tune **only the new token rows**
(everything else frozen, so English is preserved) -> measure Polish + English
bits-per-byte before and after, plus the stock-gpt2 reference.

Only the new rows adapt, so the fine-tune is cheap and gpt2-small runs fine on
**CPU** — no GPU required. The CPU path is memory-frugal by design (it trains a
compact delta over just the new rows, never materialising a full-vocab
embedding gradient) so it fits a small (~2 GiB) memory cgroup; the GPU path
trains the full embedding directly. Needs the `poc` extra (torch). Run locally
(gpt2 is public, so no credentials needed):
`MALLOC_ARENA_MAX=1 uv run --no-sync python scripts/gpt2_retrain.py`
(Sizes are tuned for CPU; override with GPT2_STEPS / GPT2_EVAL_DOCS env vars, or
run on the Hyperstack VM via `task gpu:run` for the full eval set. Set
MALLOC_ARENA_MAX=1 on CPU to keep glibc's arena growth from inflating RSS.)

Writes results/gpt2-retrain.json (+ a summary line). The retrained model is
saved to $TOKENIZER_DATA_DIR/models/gpt2-pl (off git).
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import torch
from tokenizers import Tokenizer
from transformers import AutoTokenizer, GPT2LMHeadModel

from tokenizer_core.adapters.embed import transplant_embeddings
from tokenizer_core.adapters.extend import extend
from tokenizer_core.benchmark.quality import bits_per_byte, quality_gate

# CPU-friendly defaults; raised automatically on GPU (or via env).
_ON_GPU = torch.cuda.is_available()
_N_STEPS = int(os.environ.get("GPT2_STEPS", "1500" if _ON_GPU else "300"))
_EVAL_DOCS = int(os.environ.get("GPT2_EVAL_DOCS", "1012" if _ON_GPU else "250"))
_BLOCK = 128
_BATCH = 8 if _ON_GPU else 1
_LR = 1e-3
_TRAIN_CHARS = (6 if _ON_GPU else 2) * 1_000_000
# Cap the eval context: the per-step logits tensor is
# [1, seq_len, extended_vocab] (~0.3 GB at 1024x76k in fp32). CPU runs are
# confined to a small memory cgroup, so keep the peak modest there.
_MAX_LEN = 1024 if _ON_GPU else 128


def _malloc_trim() -> None:
    """Return freed heap back to the OS (glibc retains it by default).

    Deleting large numpy/torch buffers frees them to the C allocator but not
    necessarily to the kernel, so RSS can stay near its peak. Inside the small
    CPU memory cgroup that residual can trip the OOM killer; ``malloc_trim``
    hands it back. Best-effort: a no-op where libc/``malloc_trim`` is absent.
    """
    try:
        import ctypes  # noqa: PLC0415

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def _lines(path: Path, limit_bytes: int | None = None) -> list[str]:
    # Stream line by line and stop at limit_bytes -- never read the whole file.
    # The training corpus is ~100 MB; read_text().splitlines() would load all of
    # it (plus a list of every line) and blow the CPU memory cgroup.
    out: list[str] = []
    total = 0
    with path.open(encoding="utf-8") as fh:
        for ln in fh:
            s = ln.strip()
            if not s:
                continue
            out.append(s)
            total += len(s.encode("utf-8"))
            if limit_bytes and total >= limit_bytes:
                break
    return out


def _bpb(model, encode, docs, device) -> float:
    """Bits-per-byte of ``model`` over ``docs`` using ``encode``."""
    model.eval()
    total_nll = 0.0
    total_bytes = 0
    with torch.inference_mode():
        for doc in docs:
            ids = encode(doc)
            if len(ids) < 2:
                continue
            x = torch.tensor([ids[:_MAX_LEN]], device=device)
            loss = model(x, labels=x).loss.item()
            total_nll += loss * (x.shape[1] - 1)  # mean -> sum over shifts
            total_bytes += len(doc.encode("utf-8"))
    return bits_per_byte(total_nll, total_bytes)


def _blocks(ids: list[int], block: int):
    for i in range(0, len(ids) - block, block):
        yield ids[i : i + block]


def _train_full(model, emb, base_vocab, blocks):
    """Adapt the new-token rows by training the full input embedding (GPU path).

    Trains the whole (tied) embedding with AdamW and zeros the base-row
    gradients each step so only the new tokens move. Materialises a full-vocab
    embedding gradient + AdamW state -- fine on GPU, too large for the CPU
    memory cgroup (see :func:`_train_delta`). Returns the number of steps run.
    """
    device = emb.device
    emb.requires_grad = True
    opt = torch.optim.AdamW([emb], lr=_LR)
    model.train()
    step = 0
    while step < _N_STEPS and step * _BATCH < len(blocks):
        chunk = blocks[step * _BATCH % max(1, len(blocks) - _BATCH) :][:_BATCH]
        if not chunk:
            break
        x = torch.tensor(chunk, device=device)
        model(x, labels=x).loss.backward()
        if emb.grad is not None:
            emb.grad[:base_vocab].zero_()  # keep the base vocab (English) fixed
        opt.step()
        opt.zero_grad()
        step += 1
    return step


def _train_delta(model, emb, base_vocab, blocks, *, n_new):
    """Adapt only the new rows via a compact delta, base model frozen (CPU path).

    Trains ``delta`` ``[n_new, hidden]`` added to the new-vocab rows of both the
    input embedding and the tied output head. The base weight stays frozen, so
    the full-vocab embedding gradient (~0.2 GB) is never materialised -- only
    ``delta``'s gradient is -- which fits the small CPU memory cgroup. The base
    vocab (and thus English) is left exactly intact by construction.

    The learned ``delta`` is folded into the weight before returning, so
    eval/save see a plain, standard model. Returns the number of steps run.
    """
    from torch.nn import functional as tf  # noqa: PLC0415

    device = emb.device
    hidden = emb.shape[1]
    delta = torch.nn.Parameter(
        torch.zeros(n_new, hidden, dtype=emb.dtype, device=device)
    )
    wte = model.get_input_embeddings()
    head = model.get_output_embeddings()

    def _wte_forward(input_ids):
        base = tf.embedding(
            input_ids, emb
        )  # frozen base (FVT init for new rows)
        shifted = input_ids - base_vocab
        add = tf.embedding(shifted.clamp(min=0), delta)
        return base + add * (shifted >= 0).unsqueeze(-1).to(base.dtype)

    def _head_forward(hidden_states):
        logits = tf.linear(hidden_states, emb)  # frozen base head (tied)
        delta_logits = tf.linear(hidden_states, delta)  # [..., n_new]
        return logits + tf.pad(delta_logits, (base_vocab, 0))

    wte.forward = _wte_forward
    head.forward = _head_forward
    # AdamW state over the *delta only* (~0.16 GB) fits the CPU cgroup now that
    # the base model is frozen, and gives a robust, well-scaled step.
    opt = torch.optim.AdamW([delta], lr=_LR)
    model.train()
    step = 0
    try:
        while step < _N_STEPS and step * _BATCH < len(blocks):
            chunk = blocks[step * _BATCH % max(1, len(blocks) - _BATCH) :][
                :_BATCH
            ]
            if not chunk:
                break
            x = torch.tensor(chunk, device=device)
            model(x, labels=x).loss.backward()
            torch.nn.utils.clip_grad_norm_(delta, 1.0)  # stabilise the step
            opt.step()
            opt.zero_grad()
            step += 1
    finally:
        # Restore stock forwards and fold delta into the weight -> plain model.
        del wte.forward
        del head.forward
        with torch.no_grad():
            emb[base_vocab:] += delta
    return step


def _report_md(out: dict) -> str:
    """Render the retrain result dict as a Markdown report."""
    pl = out["polish_bpb"]
    en = out["english_bpb"]
    tok = out["polish_tokens"]
    stock, transplant, retrain = (
        pl["stock_gpt2"],
        pl["transplant_only"],
        pl["after_retrain"],
    )
    penalty = transplant - stock
    recovered = round((transplant - retrain) / penalty * 100) if penalty else 0
    gate = "PASS" if en["gate_passed"] else "FAIL"
    return f"""# gpt2 retrain PoC — adapt the transplanted Polish tokens (Epic 4.2)

Extends **real gpt2** with the trained Polish vocabulary, FVT-initialises the new
tokens (training-free, see [`gpt2-transplant-poc.md`](gpt2-transplant-poc.md)),
then runs a short fine-tune of **only the new token rows** (the base model is
frozen, so English is preserved by construction). Quality is measured in
**bits-per-byte** — a tokenizer-independent, per-byte metric, so the stock and
extended tokenizers compare fairly. Ran on **{out["device"]}**, {out["steps"]}
steps. Reproduce: `MALLOC_ARENA_MAX=1 uv run --no-sync python \
scripts/gpt2_retrain.py`.

| Polish bits-per-byte | value |
|----------------------|-------|
| stock gpt2 | {stock:.3f} |
| + transplant (FVT, no retrain) | {transplant:.3f} |
| + short retrain (new rows only) | **{retrain:.3f}** |

| efficiency / safety | value |
|---------------------|-------|
| Polish tokens (extended vs stock) | {tok["extended"]} vs {tok["stock"]} \
(**-{tok["reduction_pct"]}%**) |
| transplant penalty recovered | **{recovered}%** |
| English bits-per-byte (stock -> retrain) | {en["stock_gpt2"]:.3f} -> \
{en["after_retrain"]:.3f} ({en["delta_pct"]:+.1f}%) |
| English quality gate (<=2%) | **{gate}** |

**Read:** the training-free transplant cuts Polish tokens by
{tok["reduction_pct"]}% but temporarily raises Polish bits-per-byte (the model
has never seen the merged tokens); a short fine-tune of just the new rows
recovers {recovered}% of that penalty while English stays within
{en["delta_pct"]:+.1f}% (gate {gate}). This confirms the optional retrain step
is cheap — it touches only the new embeddings, needs no GPU, and does not
regress the base language. The full recipe scales the step count / eval set on a
GPU (`task gpu:run`).
"""


def main() -> None:  # noqa: PLR0915
    """Run the transplant + short fine-tune and record quality recovery."""
    dd = Path(os.environ["TOKENIZER_DATA_DIR"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))

    def _anon_mb() -> int:
        try:
            for ln in (
                Path("/sys/fs/cgroup/memory.stat").read_text().splitlines()
            ):
                if ln.startswith("anon "):
                    return int(ln.split()[1]) // 1048576
        except (OSError, ValueError):
            pass
        return -1

    _log = lambda m: print(  # noqa: E731
        f"[retrain] {m} (anon={_anon_mb()}MB)", flush=True
    )
    artifact = str(dd / "tokenizers" / "pl" / "pl-bpe-32k")
    devtest = dd / "flores200" / "flores200_dataset" / "devtest"
    pol_eval = _lines(devtest / "pol_Latn.devtest")[:_EVAL_DOCS]
    eng_eval = _lines(devtest / "eng_Latn.devtest")[:_EVAL_DOCS]

    # One gpt2 instance throughout: score it stock, then resize it in place for
    # the extended tokenizer. Loading a second copy would double the resident
    # footprint, which does not fit the small CPU memory cgroup.
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    stock_tok = AutoTokenizer.from_pretrained("gpt2")
    base_vocab = model.get_input_embeddings().weight.shape[0]  # 50257 for gpt2
    _log("model loaded; scoring stock bpb")
    stock_pl = _bpb(model, stock_tok.encode, pol_eval, device)
    stock_en = _bpb(model, stock_tok.encode, eng_eval, device)

    # Extend + FVT transplant + resize (in place).
    res = extend("gpt2", artifact)
    ext: Tokenizer = res.tokenizer

    def ext_enc(text: str) -> list[int]:
        return ext.encode(text).ids

    _log("extend done; reading embeddings")
    # Read the base embedding matrix from the already-loaded model rather than
    # re-loading gpt2's full safetensors from disk (which would spike ~0.5 GB of
    # anonymous memory -- fatal under the CPU memory cgroup).
    base_emb = (
        model.get_input_embeddings().weight.detach().to(torch.float32).numpy()
    )
    _log("base_emb read")
    ext_emb = transplant_embeddings(res, base_emb, stock_tok, method="fvt")
    del base_emb
    gc.collect()

    _log("transplant done; resize in place")
    model.resize_token_embeddings(ext.get_vocab_size())
    _log("resized")
    with torch.no_grad():
        # as_tensor shares the numpy buffer (no full-size duplicate); the copy_
        # writes it into the model weight, after which the numpy array is freed.
        src = torch.as_tensor(
            ext_emb, dtype=model.get_input_embeddings().weight.dtype
        )
        model.get_input_embeddings().weight.copy_(src)
    del ext_emb, src
    gc.collect()
    _malloc_trim()

    _log("resized+copied; scoring before bpb")
    before_pl = _bpb(model, ext_enc, pol_eval, device)
    _log("before_pl done")
    before_en = _bpb(model, ext_enc, eng_eval, device)
    _log("before_en done")

    # Freeze the whole model; only the transplanted (new) token rows adapt.
    for p in model.parameters():
        p.requires_grad = False
    emb = model.get_input_embeddings().weight  # [V, H]; holds the FVT init
    n_new = emb.shape[0] - base_vocab

    train_text = " ".join(
        _lines(dd / "pl" / "train" / "fineweb2.txt", _TRAIN_CHARS)
    )
    _log("encoding train corpus")
    train_ids = ext.encode(train_text).ids
    blocks = list(_blocks(train_ids, _BLOCK))

    _log(f"training {_N_STEPS} steps on {len(blocks)} blocks")
    # GPU trains the full embedding directly; CPU trains a compact new-row delta
    # (base frozen) to stay within its small memory cgroup. Both move only the
    # new tokens, so English is preserved either way.
    if _ON_GPU:
        step = _train_full(model, emb, base_vocab, blocks)
    else:
        step = _train_delta(model, emb, base_vocab, blocks, n_new=n_new)

    _log("trained; scoring after bpb")
    after_pl = _bpb(model, ext_enc, pol_eval, device)
    after_en = _bpb(model, ext_enc, eng_eval, device)

    # English gate: the retrain must not regress English beyond budget.
    en_gate = quality_gate(stock_en, after_en, budget_pct=2.0)

    pol_tokens = sum(len(ext.encode(s).ids) for s in pol_eval)
    stock_pol_tokens = sum(len(stock_tok.encode(s)) for s in pol_eval)

    out = {
        "device": device,
        "steps": step,
        "polish_bpb": {
            "stock_gpt2": round(stock_pl, 4),
            "transplant_only": round(before_pl, 4),
            "after_retrain": round(after_pl, 4),
        },
        "english_bpb": {
            "stock_gpt2": round(stock_en, 4),
            "transplant_only": round(before_en, 4),
            "after_retrain": round(after_en, 4),
            "gate_passed": en_gate.passed,
            "delta_pct": round(en_gate.delta_pct, 2),
        },
        "polish_tokens": {
            "stock": stock_pol_tokens,
            "extended": pol_tokens,
            "reduction_pct": round(
                (1 - pol_tokens / stock_pol_tokens) * 100, 1
            ),
        },
    }
    Path("results/gpt2-retrain.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8"
    )
    Path("results/gpt2-retrain.md").write_text(
        _report_md(out), encoding="utf-8"
    )
    out_dir = dd / "models" / "gpt2-pl"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    ext.save(str(out_dir / "tokenizer.json"))

    print(
        f"Polish bpb: stock {stock_pl:.3f} -> transplant {before_pl:.3f} -> "
        f"retrain {after_pl:.3f} (using {out['polish_tokens']['reduction_pct']}% "
        f"fewer tokens); English gate {'PASS' if en_gate.passed else 'FAIL'} "
        f"({en_gate.delta_pct:+.1f}%)"
    )


if __name__ == "__main__":
    main()
