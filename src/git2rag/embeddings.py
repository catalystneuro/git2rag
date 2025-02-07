"""Embedding generation for text chunks."""

import os
from typing import List, Optional
from litellm import embedding


def generate_embeddings(
    texts: List[str],
    model: str = "text-embedding-ada-002",
    batch_size: int = 100,
) -> List[List[float]]:
    """Generate embeddings for a list of texts using litellm.

    Args:
        texts: List of texts to generate embeddings for
        model: Model identifier (e.g., 'text-embedding-ada-002')
        batch_size: Number of texts to process in each API call (default: 100)

    Returns:
        List of embedding vectors

    Raises:
        Exception: If the API call fails
    """
    embeddings = []
    # Process in batches
    total_texts = len(texts)
    for i in range(0, total_texts, batch_size):
        batch = texts[i : i + batch_size]
        response = embedding(
            model=model,
            input=batch,
        )
        batch_embeddings = [data["embedding"] for data in response.data]
        embeddings.extend(batch_embeddings)
        print(f"Generated embeddings for texts {i + len(batch)}/{total_texts}")

    return embeddings
