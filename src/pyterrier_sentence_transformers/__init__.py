import pyterrier as pt

from .indexer import SentenceTransformersIndexer
from .base import SentenceTransformerConfig
from .retriever import SentenceTransformersRetriever

if not pt.started():
    raise RuntimeError(
        "pyterrier must be started via `pt.init() before importing "
        "from pyterrier_sentence_transformers"
    )

__all__ = [
    "SentenceTransformersIndexer",
    "SentenceTransformerConfig",
    "SentenceTransformersRetriever"
]
