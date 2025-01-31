"""Tests for text summarization functionality."""

import os
import pytest
from unittest.mock import patch, MagicMock
from repo_indexer.summarizer import summarize_content, DEFAULT_SUMMARIZER_PROMPT


def test_summarize_content_basic():
    """Test basic summarization with mocked API response."""
    test_text = "This is a test text that should be summarized."
    expected_summary = "Test text summary."

    # Mock the litellm completion response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=expected_summary))]

    with patch("repo_indexer.summarizer.completion", return_value=mock_response):
        summary = summarize_content(test_text)
        assert summary == expected_summary


def test_summarize_content_with_custom_prompt():
    """Test summarization with custom prompt."""
    test_text = "Test text for custom prompt."
    custom_prompt = "Custom summarization prompt:"
    expected_summary = "Custom prompt summary."

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=expected_summary))]

    with patch("repo_indexer.summarizer.completion", return_value=mock_response) as mock_completion:
        summary = summarize_content(test_text, custom_prompt=custom_prompt)

        # Verify custom prompt was used
        call_args = mock_completion.call_args[1]
        messages = call_args["messages"]
        assert messages[0]["content"].startswith(custom_prompt)
        assert summary == expected_summary


def test_summarize_content_with_api_key():
    """Test summarization with custom API key."""
    test_text = "Test text for API key."
    test_api_key = "test-api-key"
    expected_summary = "API key test summary."

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=expected_summary))]

    with (
        patch("repo_indexer.summarizer.completion", return_value=mock_response),
        patch.dict("os.environ", clear=True),
    ):
        summary = summarize_content(test_text, model="openai/gpt-4", api_key=test_api_key)

        # Verify API key was set
        assert os.environ.get("OPENAI_API_KEY") == test_api_key
        assert summary == expected_summary


def test_summarize_content_error_handling():
    """Test error handling in summarization."""
    test_text = "Test text for error."
    error_message = "API error"

    with patch("repo_indexer.summarizer.completion", side_effect=Exception(error_message)):
        with pytest.raises(Exception) as exc_info:
            summarize_content(test_text)
        assert error_message in str(exc_info.value)


def test_summarize_content_parameters():
    """Test summarization with custom parameters."""
    test_text = "Test text for parameters."
    expected_summary = "Parameter test summary."

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=expected_summary))]

    with patch("repo_indexer.summarizer.completion", return_value=mock_response) as mock_completion:
        summary = summarize_content(
            test_text, model="anthropic/claude-2", max_tokens=300, temperature=0.5
        )

        # Verify parameters were passed correctly
        call_args = mock_completion.call_args[1]
        assert call_args["model"] == "anthropic/claude-2"
        assert call_args["max_tokens"] == 300
        assert call_args["temperature"] == 0.5
        assert summary == expected_summary


def test_default_prompt_content():
    """Test that default prompt contains key elements."""
    assert "semantic meaning" in DEFAULT_SUMMARIZER_PROMPT
    assert "core concepts" in DEFAULT_SUMMARIZER_PROMPT
    assert "relationships" in DEFAULT_SUMMARIZER_PROMPT
    assert "key entities" in DEFAULT_SUMMARIZER_PROMPT
