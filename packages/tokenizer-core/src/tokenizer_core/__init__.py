"""Token-efficient tokenizers for under-served languages, plus model adapters.

The package is organized into four areas:

- ``corpora``: acquire, sample, and mix training/benchmark text.
- ``train``: train candidate tokenizers (BPE / Unigram / superword variants).
- ``benchmark``: measure fertility, compression, and quality parity.
- ``adapters``: graft a trained tokenizer onto an existing model's tokenizer
  (extend/merge or full replace) with quality-preserving embedding init.

The North Star metric is *fertility* (tokens per word) on a held-out corpus:
the goal is Polish fertility at or below the English baseline without quality
regression. See ``benchmark.fertility``.
"""

__version__ = "0.0.0"
