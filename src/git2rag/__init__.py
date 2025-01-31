"""Git repository to RAG pipeline."""

from .indexer import RepoIndexer
from .chunking import ChunkingStrategy

__version__ = "0.1.0"

__all__ = ["RepoIndexer", "ChunkingStrategy"]
