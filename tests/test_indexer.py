"""Tests for the RepoIndexer class."""

import pytest
from repo_indexer.indexer import RepoIndexer
from repo_indexer.embeddings import LOCAL_E5_SMALL


@pytest.fixture
def indexer():
    """Create a RepoIndexer instance for testing."""
    return RepoIndexer(
        qdrant_url="http://localhost:6333",
        collection_name="test_collection",
        embedding_model=LOCAL_E5_SMALL,
    )


@pytest.fixture
def test_data():
    """Sample test data for indexing."""
    return [
        {
            "content": "def test_function():\n    return 'test'",
            "source_file": "test.py",
            "chunk_type": "code",
            "start_line": 1,
            "end_line": 2,
        },
        {
            "content": "# Configuration\nDEBUG=true",
            "source_file": "config.env",
            "chunk_type": "configuration",
            "start_line": 1,
            "end_line": 2,
        },
    ]


def test_indexer_initialization(indexer):
    """Test that indexer is initialized with correct attributes."""
    assert indexer.collection_name == "test_collection"
    assert indexer.embedding_model == LOCAL_E5_SMALL
    assert indexer.client is not None


def test_search_with_type_filter(indexer, test_data):
    """Test searching with chunk type filtering."""
    # Generate embeddings and store test data
    embeddings = indexer.embedding_generator.generate([item["content"] for item in test_data])
    points = []
    for i, (item, embedding) in enumerate(zip(test_data, embeddings)):
        points.append(
            {
                "id": i,
                "vector": embedding,
                "payload": {
                    "content": item["content"],
                    "source_file": item["source_file"],
                    "chunk_type": item["chunk_type"],
                    "start_line": item["start_line"],
                    "end_line": item["end_line"],
                },
            }
        )
    indexer.client.upsert(collection_name=indexer.collection_name, points=points)

    # Test search with type filter
    results = indexer.search("test function", limit=1, chunk_type="code")
    assert len(results) == 1
    assert results[0]["type"] == "code"
    assert "test_function" in results[0]["content"]


def test_search_relevance(indexer, test_data):
    """Test that search results are ordered by relevance."""
    # Generate embeddings and store test data
    embeddings = indexer.embedding_generator.generate([item["content"] for item in test_data])
    points = []
    for i, (item, embedding) in enumerate(zip(test_data, embeddings)):
        points.append(
            {
                "id": i,
                "vector": embedding,
                "payload": {
                    "content": item["content"],
                    "source_file": item["source_file"],
                    "chunk_type": item["chunk_type"],
                    "start_line": item["start_line"],
                    "end_line": item["end_line"],
                },
            }
        )
    indexer.client.upsert(collection_name=indexer.collection_name, points=points)

    # Search for configuration-related content
    results = indexer.search("configuration settings", limit=2)
    assert len(results) > 0
    # First result should be the configuration file
    assert results[0]["type"] == "configuration"
    # Scores should be in descending order
    for i in range(len(results) - 1):
        assert results[i]["score"] >= results[i + 1]["score"]
