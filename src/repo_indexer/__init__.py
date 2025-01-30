"""Repository indexing tool for RAG applications."""

from .chunking import Chunk, CodeChunker, DocumentationChunker
from .embeddings import (
    ANTHROPIC_CLAUDE,
    AZURE_ADA,
    COHERE_EMBED,
    LOCAL_E5_SMALL,
    OPENAI_ADA_002,
    EmbeddingGenerator,
    LiteLLMEmbeddings,
    LocalE5Embeddings,
    RandomEmbeddings,
)
from .indexer import RepoIndexer

__version__ = "0.1.0"
__all__ = [
    # Main classes
    "RepoIndexer",
    "Chunk",

    # Chunking
    "CodeChunker",
    "DocumentationChunker",

    # Embeddings
    "EmbeddingGenerator",
    "LiteLLMEmbeddings",
    "LocalE5Embeddings",
    "RandomEmbeddings",

    # Model identifiers
    "OPENAI_ADA_002",
    "AZURE_ADA",
    "COHERE_EMBED",
    "ANTHROPIC_CLAUDE",
    "LOCAL_E5_SMALL",
]
