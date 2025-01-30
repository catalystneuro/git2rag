"""Client modules for external services."""

from .qdrant import QdrantManager, SearchResult

__all__ = ["QdrantManager", "SearchResult"]
