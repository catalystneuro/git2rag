"""Tests for repository content summarization."""

import pytest
from unittest.mock import patch, MagicMock
from repo_indexer.indexer import RepoIndexer
from repo_indexer.chunking import Chunk


@pytest.fixture
def mock_repo_indexer():
    """Create a RepoIndexer instance with mocked dependencies."""
    with (
        patch("repo_indexer.indexer.QdrantManager"),
        patch("repo_indexer.indexer.LocalE5Embeddings"),
    ):
        indexer = RepoIndexer(qdrant_url="http://localhost:6333")
        return indexer


def test_summarize_chunks_basic(mock_repo_indexer):
    """Test basic chunk summarization."""
    # Create test chunks
    chunks = [
        Chunk(content_raw="def test(): pass", source_file="/test.py", chunk_type="code"),
        Chunk(
            content_raw="# Test documentation", source_file="/docs.md", chunk_type="documentation"
        ),
    ]

    # Add chunks to repository
    mock_repo_indexer.repositories["test_repo"] = {"chunks": chunks}

    # Mock summarize_content
    expected_summaries = ["Summarized code: test function", "Summarized doc: test documentation"]

    with patch("repo_indexer.indexer.summarize_content", side_effect=expected_summaries):
        mock_repo_indexer.summarize_chunks("test_repo")

        # Verify summaries were stored
        assert chunks[0].content_processed == expected_summaries[0]
        assert chunks[1].content_processed == expected_summaries[1]


def test_summarize_chunks_with_custom_params(mock_repo_indexer):
    """Test summarization with custom parameters."""
    chunks = [Chunk(content_raw="Test content", source_file="/test.txt")]

    mock_repo_indexer.repositories["test_repo"] = {"chunks": chunks}

    custom_prompt = "Custom summarization prompt:"
    expected_summary = "Custom summary"

    with patch(
        "repo_indexer.indexer.summarize_content", return_value=expected_summary
    ) as mock_summarize:
        mock_repo_indexer.summarize_chunks(
            repo_url="test_repo",
            model="anthropic/claude-2",
            custom_prompt=custom_prompt,
            max_tokens=300,
            temperature=0.5,
        )

        # Verify custom parameters were passed
        mock_summarize.assert_called_once_with(
            text_content="Test content",
            model="anthropic/claude-2",
            custom_prompt=custom_prompt,
            max_tokens=300,
            temperature=0.5,
        )

        assert chunks[0].content_processed == expected_summary


def test_summarize_chunks_error_handling(mock_repo_indexer):
    """Test error handling during summarization."""
    chunks = [
        Chunk(content_raw="Test content 1", source_file="/test1.txt"),
        Chunk(content_raw="Test content 2", source_file="/test2.txt"),
    ]

    mock_repo_indexer.repositories["test_repo"] = {"chunks": chunks}

    # First call succeeds, second fails
    def mock_summarize(text_content, **kwargs):
        if "content 1" in text_content:
            return "Summary 1"
        raise Exception("API error")

    with patch("repo_indexer.indexer.summarize_content", side_effect=mock_summarize):
        with pytest.raises(Exception) as exc_info:
            mock_repo_indexer.summarize_chunks("test_repo")

        assert "API error" in str(exc_info.value)
        # First chunk should be summarized despite second failing
        assert chunks[0].content_processed == "Summary 1"


def test_summarize_chunks_empty_repo(mock_repo_indexer):
    """Test summarization with empty repository."""
    mock_repo_indexer.repositories["empty_repo"] = {"chunks": []}

    with patch("repo_indexer.indexer.summarize_content") as mock_summarize:
        mock_repo_indexer.summarize_chunks("empty_repo")
        mock_summarize.assert_not_called()


def test_summarize_chunks_invalid_repo(mock_repo_indexer):
    """Test summarization with invalid repository."""
    with pytest.raises(KeyError):
        mock_repo_indexer.summarize_chunks("nonexistent_repo")
