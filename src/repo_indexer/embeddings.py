"""Embedding generation for text chunks."""

import os
from typing import List, Optional, Protocol, Union

import numpy as np
import requests


class EmbeddingGenerator(Protocol):
    """Protocol for embedding generators."""

    def generate(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        ...


class LocalE5Embeddings:
    """Local multilingual-e5-small embeddings."""

    def __init__(
        self,
        url: str = "http://0.0.0.0:8001/v1/embeddings",
        batch_size: int = 32,  # Reduced batch size
        vector_size: int = 384,  # E5-small embedding size
        fallback_to_random: bool = True,
    ):
        """Initialize the embedding generator.

        Args:
            url: URL of the local embedding server
            batch_size: Number of texts to process in each API call
            vector_size: Size of the embedding vectors
            fallback_to_random: Whether to fall back to random embeddings on error
        """
        self.url = url
        self.batch_size = batch_size
        self.vector_size = vector_size
        self.fallback_to_random = fallback_to_random

    def _get_single_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        response = requests.post(
            self.url,
            headers={"Content-Type": "application/json"},
            json={
                "input": text,  # OpenAI-style API format
                "model": "intfloat/multilingual-e5-small",
            },
            timeout=30.0,  # 30 second timeout
        )
        response.raise_for_status()
        result = response.json()
        return result["data"][0]["embedding"]

    def generate(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using local server.

        Args:
            texts: List of texts to generate embeddings for

        Returns:
            List of embedding vectors
        """
        try:
            embeddings = []
            # Process one text at a time to avoid token limit issues
            for i, text in enumerate(texts, 1):
                if i % 10 == 0:  # Progress update every 10 texts
                    print(f"Processing text {i}/{len(texts)}...")
                try:
                    embedding = self._get_single_embedding(text)
                    embeddings.append(embedding)
                except Exception as e:
                    print(f"Error embedding text {i}: {e}")
                    if not self.fallback_to_random:
                        raise
                    print("Using random embedding for this text")
                    embeddings.append(self._generate_random(1)[0])

            return embeddings

        except Exception as e:
            print(f"Error getting embeddings: {e}")
            if not self.fallback_to_random:
                raise
            print("Falling back to random embeddings")
            return self._generate_random(len(texts))

    def _generate_random(self, count: int) -> List[List[float]]:
        """Generate random embeddings for testing or fallback.

        Args:
            count: Number of random embeddings to generate

        Returns:
            List of random embedding vectors
        """
        return [
            list(np.random.uniform(-1, 1, self.vector_size))
            for _ in range(count)
        ]


class LiteLLMEmbeddings:
    """LiteLLM-based embedding generator supporting multiple providers."""

    def __init__(
        self,
        model: str = "text-embedding-ada-002",
        api_key: Optional[str] = None,
        batch_size: int = 100,
        vector_size: int = 1536,
        fallback_to_random: bool = True,
    ):
        """Initialize the embedding generator.

        Args:
            model: Model identifier (e.g., 'text-embedding-ada-002', 'azure/azure-embedding-model')
            api_key: API key for the selected provider. If None, uses environment variables.
            batch_size: Number of texts to process in each API call.
            vector_size: Size of the embedding vectors.
            fallback_to_random: Whether to fall back to random embeddings on error.
        """
        try:
            from litellm import embedding
            self.embedding = embedding
        except ImportError:
            print("Warning: litellm not installed. Using random embeddings.")
            self.embedding = None

        self.model = model
        self.batch_size = batch_size
        self.vector_size = vector_size
        self.fallback_to_random = fallback_to_random

        # Set API key if provided
        if api_key:
            provider = model.split("/")[0] if "/" in model else "openai"
            os.environ[f"{provider.upper()}_API_KEY"] = api_key

    def generate(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using the configured provider.

        Args:
            texts: List of texts to generate embeddings for

        Returns:
            List of embedding vectors
        """
        if not self.embedding:
            return self._generate_random(len(texts))

        try:
            embeddings = []
            # Process in batches to avoid rate limits
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                response = self.embedding(
                    model=self.model,
                    input=batch,
                )
                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)

            return embeddings

        except Exception as e:
            print(f"Error getting embeddings: {e}")
            if not self.fallback_to_random:
                raise
            print("Falling back to random embeddings")
            return self._generate_random(len(texts))

    def _generate_random(self, count: int) -> List[List[float]]:
        """Generate random embeddings for testing or fallback.

        Args:
            count: Number of random embeddings to generate

        Returns:
            List of random embedding vectors
        """
        return [
            list(np.random.uniform(-1, 1, self.vector_size))
            for _ in range(count)
        ]


class RandomEmbeddings:
    """Random embedding generator for testing."""

    def __init__(self, vector_size: int = 1536):
        """Initialize the random embedding generator.

        Args:
            vector_size: Size of the embedding vectors.
        """
        self.vector_size = vector_size

    def generate(self, texts: List[str]) -> List[List[float]]:
        """Generate random embeddings for testing.

        Args:
            texts: List of texts to generate embeddings for

        Returns:
            List of random embedding vectors
        """
        return [
            list(np.random.uniform(-1, 1, self.vector_size))
            for _ in texts
        ]


# Common model identifiers
OPENAI_ADA_002 = "text-embedding-ada-002"
AZURE_ADA = "azure/text-embedding-ada-002"
COHERE_EMBED = "cohere/embed-english-v3.0"
ANTHROPIC_CLAUDE = "anthropic/claude-2"
LOCAL_E5_SMALL = "intfloat/multilingual-e5-small"
