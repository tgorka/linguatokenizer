"""Model-tokenizer adapters: target-model registry and grafting utilities.

``tokenizer_core.adapters`` is the stable import surface for Stories 3.2-3.4
(extend/replace/OMP embedding-init adapters). It re-exports the target-model
registry (Story 3.1) and the ``extend`` adapter + shared result contract
(Story 3.2); later stories add the remaining adapters.
"""

from tokenizer_core.adapters.base import AdapterResult
from tokenizer_core.adapters.bundle import export_bundle, load_bundle
from tokenizer_core.adapters.embed import (
    EmbedError,
    read_token_embeddings,
    transplant_embeddings,
)
from tokenizer_core.adapters.extend import extend
from tokenizer_core.adapters.registry import (
    REGISTRY,
    AdapterError,
    TargetModelEntry,
    TokenizerFormat,
    load_target_tokenizer,
)
from tokenizer_core.adapters.replace import replace

__all__ = [
    "REGISTRY",
    "AdapterError",
    "AdapterResult",
    "EmbedError",
    "TargetModelEntry",
    "TokenizerFormat",
    "export_bundle",
    "extend",
    "load_bundle",
    "load_target_tokenizer",
    "read_token_embeddings",
    "replace",
    "transplant_embeddings",
]
