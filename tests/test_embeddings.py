"""Tests for embedding generation functionality."""

import os
from unittest.mock import patch, MagicMock
import pytest
import numpy as np
from repo_indexer.embeddings import (
    RandomEmbeddings,
    LocalE5Embeddings,
    LiteLLMEmbeddings,
    OPENAI_ADA_002,
)


@pytest.fixture
def sample_texts():
    """Sample texts for testing embeddings."""
    return [
        "This is the first test text",
        "Here is another sample",
        "And a third one for good measure",
    ]


def test_random_embeddings():
    """Test that RandomEmbeddings generates vectors of correct size and shape."""
    vector_size = 384
    generator = RandomEmbeddings(vector_size=vector_size)
    texts = ["Test text 1", "Test text 2"]

    embeddings = generator.generate(texts)

    assert len(embeddings) == len(texts)
    assert all(len(emb) == vector_size for emb in embeddings)
    assert all(isinstance(val, float) for emb in embeddings for val in emb)
    assert all(-1 <= val <= 1 for emb in embeddings for val in emb)


def test_random_embeddings_consistency():
    """Test that different calls produce different random embeddings."""
    generator = RandomEmbeddings(vector_size=384)
    text = ["Same text"]

    embedding1 = generator.generate(text)
    embedding2 = generator.generate(text)

    # Embeddings should be different on each call
    assert not np.allclose(embedding1, embedding2)


class MockResponse:
    """Mock response for requests."""

    def __init__(self, json_data):
        self._json_data = json_data

    def json(self):
        return self._json_data

    def raise_for_status(self):
        pass


@pytest.fixture
def mock_e5_response():
    """Mock response for E5 embedding server."""

    def create_response(text):
        return MockResponse({"data": [{"embedding": list(np.random.uniform(-1, 1, 384))}]})

    return create_response


def test_local_e5_embeddings(sample_texts, mock_e5_response):
    """Test LocalE5Embeddings with mocked server responses."""
    generator = LocalE5Embeddings(fallback_to_random=False)

    with patch("requests.post", side_effect=mock_e5_response):
        embeddings = generator.generate(sample_texts)

        assert len(embeddings) == len(sample_texts)
        assert all(len(emb) == 384 for emb in embeddings)
        assert all(isinstance(val, float) for emb in embeddings for val in emb)


def test_local_e5_embeddings_fallback():
    """Test that LocalE5Embeddings falls back to random when server fails."""
    generator = LocalE5Embeddings(fallback_to_random=True)
    texts = ["Test text"]

    with patch("requests.post", side_effect=Exception("Server error")):
        embeddings = generator.generate(texts)

        assert len(embeddings) == len(texts)
        assert len(embeddings[0]) == 384
        assert all(isinstance(val, float) for emb in embeddings for val in emb)


def test_local_e5_embeddings_no_fallback():
    """Test that LocalE5Embeddings raises error when fallback is disabled."""
    generator = LocalE5Embeddings(fallback_to_random=False)
    texts = ["Test text"]

    with patch("requests.post", side_effect=Exception("Server error")):
        with pytest.raises(Exception):
            generator.generate(texts)


@pytest.fixture
def mock_litellm():
    """Mock LiteLLM embedding function."""

    class MockEmbeddingResponse:
        def __init__(self, texts):
            self.data = [MagicMock(embedding=list(np.random.uniform(-1, 1, 1536))) for _ in texts]

    mock = MagicMock()
    mock.return_value = MockEmbeddingResponse(["dummy"])
    return mock


def test_litellm_embeddings(sample_texts, mock_litellm):
    """Test LiteLLMEmbeddings with mocked litellm."""
    generator = LiteLLMEmbeddings(
        model=OPENAI_ADA_002, api_key="test-key", fallback_to_random=False
    )

    with patch.object(generator, "embedding", mock_litellm):
        embeddings = generator.generate(sample_texts)

        assert len(embeddings) == len(sample_texts)
        assert all(len(emb) == 1536 for emb in embeddings)
        assert all(isinstance(val, float) for emb in embeddings for val in emb)

        # Verify API key was set
        assert os.environ.get("OPENAI_API_KEY") == "test-key"


def test_litellm_embeddings_fallback():
    """Test LiteLLMEmbeddings fallback behavior."""
    generator = LiteLLMEmbeddings(fallback_to_random=True)
    texts = ["Test text"]

    # Simulate litellm import failure
    generator.embedding = None
    embeddings = generator.generate(texts)

    assert len(embeddings) == len(texts)
    assert len(embeddings[0]) == 1536
    assert all(isinstance(val, float) for emb in embeddings for val in emb)


def test_litellm_embeddings_batching(mock_litellm):
    """Test that LiteLLMEmbeddings correctly handles batching."""
    batch_size = 2
    texts = ["Text 1", "Text 2", "Text 3"]
    generator = LiteLLMEmbeddings(batch_size=batch_size, fallback_to_random=False)

    with patch.object(generator, "embedding", mock_litellm):
        generator.generate(texts)

        # Should have been called twice: once for texts[0:2], once for texts[2:]
        assert mock_litellm.call_count == 2

        # Verify batch sizes
        first_call = mock_litellm.call_args_list[0][1]["input"]
        second_call = mock_litellm.call_args_list[1][1]["input"]
        assert len(first_call) == 2
        assert len(second_call) == 1
