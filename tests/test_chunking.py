"""Tests for content chunking functionality."""

import pytest
from repo_indexer.chunking import (
    BaseChunker,
    CodeChunker,
    DocumentationChunker,
    get_chunker,
    Chunk
)

@pytest.fixture
def base_chunker():
    """Create a BaseChunker instance with small chunk size for testing."""
    return BaseChunker(chunk_size=50, overlap=10, max_tokens=100)

@pytest.fixture
def code_chunker():
    """Create a CodeChunker instance with small chunk size for testing."""
    return CodeChunker(chunk_size=100, overlap=20, max_tokens=100)

@pytest.fixture
def doc_chunker():
    """Create a DocumentationChunker instance with small chunk size for testing."""
    return DocumentationChunker(chunk_size=100, overlap=20, max_tokens=100)

def test_base_chunker_simple_text(base_chunker):
    """Test basic text chunking with the base chunker."""
    content = "This is a test.\nIt has multiple lines.\nEach line is different.\nTesting chunking behavior."
    chunks = base_chunker.chunk_content(content, "test.txt")

    assert len(chunks) > 0
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert all(len(chunk.content) <= base_chunker.chunk_size for chunk in chunks)
    assert all(chunk.chunk_type == "text" for chunk in chunks)

def test_code_chunker_class_detection(code_chunker):
    """Test that CodeChunker properly detects and chunks classes."""
    content = '''
class TestClass:
    """Test class docstring."""

    def __init__(self):
        self.value = 42

    def test_method(self):
        return self.value

def standalone_function():
    return True
'''
    chunks = code_chunker.chunk_content(content, "test.py")

    assert len(chunks) > 0
    assert any("class TestClass" in chunk.content for chunk in chunks)
    assert any("def test_method" in chunk.content for chunk in chunks)
    assert any("def standalone_function" in chunk.content for chunk in chunks)
    assert all(chunk.chunk_type == "code" for chunk in chunks)

def test_doc_chunker_sections(doc_chunker):
    """Test that DocumentationChunker properly handles markdown sections."""
    content = '''# Section 1
This is the first section content.

## Subsection 1.1
This is a subsection.

# Section 2
Another top-level section.
'''
    chunks = doc_chunker.chunk_content(content, "test.md")

    assert len(chunks) > 0
    assert all(chunk.chunk_type == "documentation" for chunk in chunks)
    # Check that sections are preserved in context
    assert any(chunk.context and "# Section 1" in chunk.context for chunk in chunks)
    assert any(chunk.context and "# Section 2" in chunk.context for chunk in chunks)

def test_chunker_selection():
    """Test that appropriate chunkers are selected based on file extension."""
    # Python file should get CodeChunker
    chunker = get_chunker("test.py")
    assert isinstance(chunker, CodeChunker)

    # Markdown file should get DocumentationChunker
    chunker = get_chunker("test.md")
    assert isinstance(chunker, DocumentationChunker)

    # Unknown extension should get BaseChunker
    chunker = get_chunker("test.xyz")
    assert isinstance(chunker, BaseChunker)
    assert not isinstance(chunker, (CodeChunker, DocumentationChunker))

def test_chunk_metadata():
    """Test that chunks contain correct metadata."""
    chunker = BaseChunker(chunk_size=50)
    content = "Line 1\nLine 2\nLine 3"
    chunks = chunker.chunk_content(content, "test.txt")

    assert len(chunks) > 0
    chunk = chunks[0]
    assert chunk.source_file == "test.txt"
    assert chunk.start_line is not None
    assert chunk.end_line is not None
    assert chunk.start_line <= chunk.end_line

def test_code_chunker_nested_structures(code_chunker):
    """Test that CodeChunker handles nested class/function structures."""
    content = '''
class OuterClass:
    def outer_method(self):
        class InnerClass:
            def inner_method(self):
                return True
        return InnerClass()

    def another_method(self):
        return False
'''
    chunks = code_chunker.chunk_content(content, "test.py")

    assert len(chunks) > 0
    assert any("class OuterClass" in chunk.content for chunk in chunks)
    assert any("def outer_method" in chunk.content for chunk in chunks)
    assert any("class InnerClass" in chunk.content for chunk in chunks)
    assert any("def another_method" in chunk.content for chunk in chunks)

def test_doc_chunker_mixed_content(doc_chunker):
    """Test that DocumentationChunker handles mixed content types."""
    content = '''# API Reference

## Functions

```python
def example():
    return True
```

## Configuration

Settings are stored in `config.json`:
```json
{
    "debug": true
}
```
'''
    chunks = doc_chunker.chunk_content(content, "api.md")

    assert len(chunks) > 0
    assert all(chunk.chunk_type == "documentation" for chunk in chunks)
    # Check that code blocks are kept together where possible
    assert any("```python\ndef example()" in chunk.content for chunk in chunks)
    assert any("```json\n{" in chunk.content for chunk in chunks)
