"""Baseline comparison table: score real Hugging Face tokenizers on FLORES.

`benchmark.fertility` can score any single :class:`~tokenizer_core.benchmark.
fertility.Encoder` over aligned corpora (Story 1.3), but nothing yet turns a
recipe's ``baselines`` list (``gpt2``, Llama, Mistral, Qwen, Gemma, Bielik-v3,
...) into the committed ``results/baselines-<lang>-en.json``/``.md`` table
that states the real-data Polish fertility penalty. This module is that
missing piece: :func:`compare_baselines` loads each baseline tokenizer,
scores it with :func:`~tokenizer_core.benchmark.fertility.aligned_fertility`
against one FLORES target/English pair, and gracefully records any baseline
that fails to load as a skip rather than aborting the whole run.

``transformers`` (the ``adapters`` extra) is imported lazily inside
:func:`_load_baseline_encoder` only -- this module's own import, like
``fertility.py``'s, stays light.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from tokenizer_core.benchmark.fertility import (
    AlignedFertilityReport,
    Encoder,
    aligned_fertility,
)
from tokenizer_core.corpora.sources import resolve_source
from tokenizer_core.train.config import CorpusSpec

if TYPE_CHECKING:
    from tokenizer_core.train.config import Recipe

_FLORES_SOURCE = "hf"
_FLORES_ID = "facebook/flores"
_ENGLISH_CONFIG = "eng_Latn"
# A FLORES-200 language config: ISO 639-3 + ISO 15924 script, e.g. deu_Latn,
# rus_Cyrl, hin_Deva, zho_Hans. Matches any script, not just Latin.
_FLORES_CONFIG_RE = re.compile(r"^[a-z]{3}_[A-Z][a-z]{3}$")


class CompareError(Exception):
    """Raised when the baseline comparison cannot be set up at all.

    This is distinct from a single baseline failing to load (recorded as a
    skipped :class:`BaselineOutcome`, never raised) -- it signals a problem
    with the comparison's own setup, e.g. the recipe has no FLORES eval
    corpus to derive the aligned pair from.
    """


@dataclass(frozen=True)
class BaselineOutcome:
    """The result of scoring (or failing to score) one baseline tokenizer.

    Attributes:
        name: The baseline's Hugging Face model id, as given in
            ``recipe.baselines``.
        report: The :class:`~tokenizer_core.benchmark.fertility.
            AlignedFertilityReport` if the baseline loaded and scored
            successfully, else ``None``.
        skip_reason: A human-readable reason naming the baseline if it was
            skipped, else ``None``. Exactly one of ``report``/``skip_reason``
            is set.
    """

    name: str
    report: AlignedFertilityReport | None
    skip_reason: str | None

    def as_dict(self) -> dict[str, object]:
        """Return the outcome as a plain JSON-serializable dict."""
        return {
            "name": self.name,
            "report": self.report.as_dict()
            if self.report is not None
            else None,
            "skip_reason": self.skip_reason,
        }


@dataclass(frozen=True)
class ComparisonReport:
    """The full baseline comparison table for one target language.

    Attributes:
        language_code: ISO 639-1 code of the target language (e.g. ``pl``).
        outcomes: One :class:`BaselineOutcome` per ``recipe.baselines`` entry,
            in the same order as the recipe.
    """

    language_code: str
    outcomes: tuple[BaselineOutcome, ...]

    def as_dict(self) -> dict[str, object]:
        """Return the report as a plain JSON-serializable dict."""
        return {
            "language_code": self.language_code,
            "outcomes": [outcome.as_dict() for outcome in self.outcomes],
        }


def _load_baseline_encoder(name: str) -> tuple[Encoder | None, str | None]:
    """Load a Hugging Face tokenizer by model id, never raising.

    Args:
        name: Hugging Face model id (e.g. ``gpt2``,
            ``meta-llama/Llama-3.1-8B``).

    Returns:
        An ``(encoder, skip_reason)`` tuple. Exactly one is not ``None``:
        the loaded encoder on success, or a human-readable reason naming
        ``name`` and the underlying error on any failure (network error,
        gated repo without accepted terms, missing token, bad id, missing
        ``transformers`` extra, ...).
    """
    try:
        import transformers  # noqa: PLC0415

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            name, token=os.environ.get("HF_TOKEN")
        )
    except Exception as err:
        # Import inside the guard too: in an extras-free environment the
        # missing `transformers` should skip the baseline, not crash the run.
        return None, f"failed to load baseline {name!r}: {err}"
    return tokenizer, None


def _english_twin(spec: CorpusSpec) -> CorpusSpec:
    """Copy a FLORES ``CorpusSpec`` with its config swapped to English."""
    return spec.model_copy(update={"config": _ENGLISH_CONFIG})


def _find_aligned_pair(recipe: Recipe) -> tuple[CorpusSpec, CorpusSpec]:
    """Return the (target, English) sentence-aligned eval corpus specs.

    Two layouts are supported, in priority order:

    1. **Explicit local/hf pair** — the recipe has one eval entry configured
       ``eng_Latn`` (English) and one other FLORES-config entry (``<iso639-3>_
       <Script>``, e.g. ``pol_Latn``, ``rus_Cyrl``, ``hin_Deva``) — any script,
       not just Latin. This is the ungated, reproducible path: ``source="local"``
       entries pointing at the FLORES-200 files from ``scripts/fetch-flores.sh``.
    2. **Back-compat single ``facebook/flores`` hf entry** — the target spec is
       that entry and English is derived by swapping its ``config`` to
       ``eng_Latn`` (both configs live under the one gated hf dataset).

    Raises:
        CompareError: If neither layout is present.
    """
    english = next(
        (s for s in recipe.eval_corpora if s.config == _ENGLISH_CONFIG),
        None,
    )
    target = next(
        (
            s
            for s in recipe.eval_corpora
            if s.config is not None
            and s.config != _ENGLISH_CONFIG
            and _FLORES_CONFIG_RE.match(s.config)
        ),
        None,
    )
    if english is not None and target is not None:
        return target, english

    for spec in recipe.eval_corpora:
        if spec.source == _FLORES_SOURCE and spec.id == _FLORES_ID:
            return spec, _english_twin(spec)

    raise CompareError(
        "recipe has no aligned eval pair: expected either a local/hf "
        f"'{_ENGLISH_CONFIG}' entry plus a target '*_Latn' entry, or a "
        f"single source={_FLORES_SOURCE!r} id={_FLORES_ID!r} entry"
    )


def compare_baselines(
    recipe: Recipe,
    *,
    data_dir: str | Path | None = None,
) -> ComparisonReport:
    """Score every baseline in ``recipe.baselines`` on aligned FLORES text.

    Args:
        recipe: The language recipe naming the baselines and FLORES eval
            corpus to use.
        data_dir: Data-plane root passed through to
            :func:`~tokenizer_core.corpora.sources.resolve_source`.

    Returns:
        A :class:`ComparisonReport` with one :class:`BaselineOutcome` per
        ``recipe.baselines`` entry, in order.

    Raises:
        CompareError: If ``recipe.eval_corpora`` has no aligned pair to derive
            the target/English corpora from (see :func:`_find_aligned_pair`).
        ValueError: Propagated unchanged from
            :func:`~tokenizer_core.benchmark.fertility.aligned_fertility` if
            the resolved target/English FLORES pair have different lengths.
    """
    target_spec, english_spec = _find_aligned_pair(recipe)

    target_docs, _ = resolve_source(target_spec, data_dir=data_dir)
    english_docs, _ = resolve_source(english_spec, data_dir=data_dir)
    target_docs = list(target_docs)
    english_docs = list(english_docs)

    outcomes = []
    for name in recipe.baselines:
        encoder, skip_reason = _load_baseline_encoder(name)
        if encoder is None:
            outcomes.append(
                BaselineOutcome(name=name, report=None, skip_reason=skip_reason)
            )
            continue
        report = aligned_fertility(encoder, target_docs, english_docs)
        outcomes.append(
            BaselineOutcome(name=name, report=report, skip_reason=None)
        )

    return ComparisonReport(
        language_code=recipe.language.code,
        outcomes=tuple(outcomes),
    )


def _format_outcome_row(outcome: BaselineOutcome) -> str:
    """Return one Markdown table row for a single baseline outcome."""
    if outcome.report is None:
        # Collapse any newlines/pipes in the skip reason (e.g. a multi-line
        # gated-repo 403) to a single cell so the table stays well-formed.
        reason = " ".join((outcome.skip_reason or "").split()).replace("|", "/")
        return f"| {outcome.name} | -- | -- | -- | skipped: {reason} |"
    report = outcome.report
    return (
        f"| {outcome.name} | {report.target.fertility:.3f} | "
        f"{report.english.fertility:.3f} | {report.fertility_vs_english:.3f} | "
        "scored |"
    )


def write_comparison_markdown(
    report: ComparisonReport, path: str | Path
) -> Path:
    """Write a Markdown baseline comparison table to ``path``.

    Args:
        report: The :class:`ComparisonReport` to render.
        path: Destination file path. Parent directories are created if they
            do not already exist.

    Returns:
        ``path`` as a :class:`Path` (not resolved to an absolute/canonical
        form -- a relative input stays relative).
    """
    dest_path = Path(path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Baseline Comparison",
        "",
        f"Target language: `{report.language_code}` vs. English (FLORES-aligned)",
        "",
        "| Baseline | PL fertility | EN fertility | fertility_vs_english | Status |",
        "|----------|-------------:|-------------:|----------------------:|--------|",
    ]
    lines.extend(_format_outcome_row(outcome) for outcome in report.outcomes)
    lines.append("")

    dest_path.write_text("\n".join(lines), encoding="utf-8")
    return dest_path
