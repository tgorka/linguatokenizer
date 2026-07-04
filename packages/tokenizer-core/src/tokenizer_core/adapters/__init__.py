"""Model-tokenizer adapters: target-model registry and grafting utilities.

``tokenizer_core.adapters`` is the stable import surface for Stories 3.2-3.4
(extend/replace/OMP embedding-init adapters). Today it re-exports the
target-model registry (Story 3.1); later stories add the adapters themselves.
"""

from tokenizer_core.adapters.registry import (
    REGISTRY,
    AdapterError,
    TargetModelEntry,
    TokenizerFormat,
    load_target_tokenizer,
)

__all__ = [
    "REGISTRY",
    "AdapterError",
    "TargetModelEntry",
    "TokenizerFormat",
    "load_target_tokenizer",
]
