"""Tests for Qdrant client functionality."""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from repo_indexer.clients.qdrant import QdrantManager, SearchResult
from qdrant_client.http.models import Distance
from qdrant_client.models import PointStruct


@pytest.fixture
def qdrant_manager():
    """Create a QdrantManager instance for testing."""
    return QdrantManager(url="http://localhost:6333", batch_size=50)


@pytest.fixture
def mock_qdrant_client():
    """Create a mock Qdrant client."""
    return MagicMock()


@pytest.fixture
def sample_points():
    """Create sample points for testing."""
    return [
        PointStruct(
            id=1,
            vector=[0.1, 0.2, 0.3],
            payload={
                "content": "Test content 1",
                "source_file": "test1.py",
                "chunk_type": "code",
                "start_line": 1,
                "end_line": 5,
                "context": "class Test",
            },
        ),
        PointStruct(
            id=2,
            vector=[0.4, 0.5, 0.6],
            payload={
                "content": "Test content 2",
                "source_file": "test2.md",
                "chunk_type": "documentation",
                "start_line": 1,
                "end_line": 3,
                "context": "# Section",
            },
        ),
    ]


def test_create_collection(qdrant_manager, mock_qdrant_client):
    """Test collection creation."""
    with patch.object(qdrant_manager, "client", mock_qdrant_client):
        # Mock existing collections
        mock_qdrant_client.get_collections.return_value.collections = []

        # Test creating new collection
        result = qdrant_manager.create_collection(name="test_collection", vector_size=384)
        assert result is True
        mock_qdrant_client.create_collection.assert_called_once()

        # Test attempting to create existing collection
        mock_qdrant_client.get_collections.return_value.collections = [
            MagicMock(name="test_collection")
        ]
        result = qdrant_manager.create_collection(name="test_collection", vector_size=384)
        assert result is False


def test_insert_points(qdrant_manager, mock_qdrant_client, sample_points):
    """Test point insertion with batching."""
    with patch.object(qdrant_manager, "client", mock_qdrant_client):
        qdrant_manager.insert_points(collection_name="test_collection", points=sample_points)

        mock_qdrant_client.upsert.assert_called_once_with(
            collection_name="test_collection", points=sample_points
        )


def test_search(qdrant_manager, mock_qdrant_client, sample_points):
    """Test vector search functionality."""
    with patch.object(qdrant_manager, "client", mock_qdrant_client):
        # Mock search results
        mock_result = MagicMock()
        mock_result.score = 0.95
        mock_result.payload = sample_points[0].payload
        mock_qdrant_client.search.return_value = [mock_result]

        # Perform search
        results = qdrant_manager.search(
            collection_name="test_collection", query_vector=[0.1, 0.2, 0.3], limit=1
        )

        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].score == 0.95
        assert results[0].content == "Test content 1"
        assert results[0].type == "code"


def test_search_with_filters(qdrant_manager, mock_qdrant_client):
    """Test search with filtering conditions."""
    with patch.object(qdrant_manager, "client", mock_qdrant_client):
        qdrant_manager.search(
            collection_name="test_collection",
            query_vector=[0.1, 0.2, 0.3],
            filter_conditions={"chunk_type": "code", "source_file": {"path": ".*\\.py$"}},
        )

        # Verify that search was called with correct filters
        mock_qdrant_client.search.assert_called_once()
        call_kwargs = mock_qdrant_client.search.call_args[1]
        assert "query_filter" in call_kwargs
        assert call_kwargs["collection_name"] == "test_collection"


def test_count_points(qdrant_manager, mock_qdrant_client):
    """Test point counting functionality."""
    with patch.object(qdrant_manager, "client", mock_qdrant_client):
        mock_qdrant_client.count.return_value.count = 42

        count = qdrant_manager.count_points(
            collection_name="test_collection", filter_conditions={"chunk_type": "code"}
        )

        assert count == 42
        mock_qdrant_client.count.assert_called_once()


def test_scroll_points(qdrant_manager, mock_qdrant_client, sample_points):
    """Test point scrolling functionality."""
    with patch.object(qdrant_manager, "client", mock_qdrant_client):
        mock_qdrant_client.scroll.return_value = (sample_points, "next_offset")

        points, offset = qdrant_manager.scroll_points(collection_name="test_collection", limit=2)

        assert len(points) == 2
        assert offset == "next_offset"
        mock_qdrant_client.scroll.assert_called_once()


def test_update_vectors(qdrant_manager, mock_qdrant_client):
    """Test vector updating functionality."""
    with patch.object(qdrant_manager, "client", mock_qdrant_client):
        points = [(1, [0.1, 0.2, 0.3]), (2, [0.4, 0.5, 0.6])]

        qdrant_manager.update_vectors(collection_name="test_collection", points=points)

        mock_qdrant_client.update_vectors.assert_called_once_with(
            collection_name="test_collection", points=points
        )


def test_list_collections(qdrant_manager, mock_qdrant_client):
    """Test listing collections."""
    with patch.object(qdrant_manager, "client", mock_qdrant_client):
        # Mock collections list
        collection1 = MagicMock(name="collection1")
        collection2 = MagicMock(name="collection2")
        mock_qdrant_client.get_collections.return_value.collections = [collection1, collection2]

        # Mock collection info
        def get_collection_mock(name):
            info = MagicMock()
            info.name = name
            info.config.params.vectors.size = 384 if name == "collection1" else 1536
            info.config.params.vectors.distance = Distance.COSINE
            info.status = "green"
            info.vectors_count = 100 if name == "collection1" else 200
            return info

        mock_qdrant_client.get_collection.side_effect = get_collection_mock

        # Test listing collections
        collections = qdrant_manager.list_collections()
        assert len(collections) == 2

        # Verify first collection
        assert collections[0]["name"] == "collection1"
        assert collections[0]["vector_size"] == 384
        assert collections[0]["vectors_count"] == 100
        assert collections[0]["status"] == "green"

        # Verify second collection
        assert collections[1]["name"] == "collection2"
        assert collections[1]["vector_size"] == 1536
        assert collections[1]["vectors_count"] == 200
        assert collections[1]["status"] == "green"

        # Test handling inaccessible collections
        mock_qdrant_client.get_collection.side_effect = Exception("Access error")
        collections = qdrant_manager.list_collections()
        assert len(collections) == 0


def test_get_collection_info(qdrant_manager, mock_qdrant_client):
    """Test collection info retrieval."""
    with patch.object(qdrant_manager, "client", mock_qdrant_client):
        # Mock successful retrieval
        mock_info = MagicMock()
        mock_info.name = "test_collection"
        mock_info.config.params.vectors.size = 384
        mock_info.config.params.vectors.distance = Distance.COSINE
        mock_info.status = "green"
        mock_info.vectors_count = 150

        mock_qdrant_client.get_collection.return_value = mock_info

        info = qdrant_manager.get_collection_info("test_collection")
        assert info["name"] == "test_collection"
        assert info["vector_size"] == 384
        assert info["distance"] == Distance.COSINE
        assert info["status"] == "green"
        assert info["vectors_count"] == 150

        # Mock failed retrieval
        mock_qdrant_client.get_collection.side_effect = Exception("Not found")
        info = qdrant_manager.get_collection_info("nonexistent")
        assert info is None
