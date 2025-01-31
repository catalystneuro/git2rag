"""Search functionality for querying the vector database."""

from typing import List, Optional

from .embeddings import generate_embeddings
from .clients import QdrantManager, SearchResult


def search_collection(
    query: str,
    qdrant_manager: QdrantManager,
    collection_name: str,
    limit: int = 5,
    vector_names: List[str] = ["raw", "processed"],
    model: str = "text-embedding-ada-002",
    api_key: Optional[str] = None,
) -> List[SearchResult]:
    """Search for relevant content using multiple vector types.

    Args:
        query: Search query text
        qdrant: QdrantManager instance
        collection_name: Name of the collection to search
        limit: Maximum number of results to return
        vector_names: Which vector to search ("raw" and/or "processed")
        model: Model to use for generating query embedding
        api_key: Optional API key for embedding model

    Returns:
        List of search results with scores
    """
    # Generate query embedding
    query_embedding = generate_embeddings(
        texts=[query],
        model=model,
        api_key=api_key,
        batch_size=1
    )[0]

    # Search using each vector type and collect all results
    all_results = []
    seen_ids = set()

    for vector_name in vector_names:
        results = qdrant_manager.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            vector_name=vector_name,
            limit=limit,
        )

        # Add unique results
        for result in results:
            result_id = result.id  # Use Point's ID for deduplication
            if result_id not in seen_ids:
                seen_ids.add(result_id)
                all_results.append(result)  # Use SearchResult directly from QdrantManager

    # Sort by score and return top results
    return sorted(all_results, key=lambda x: x.score, reverse=True)[:limit]
