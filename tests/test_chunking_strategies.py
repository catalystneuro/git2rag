"""Tests for different chunking strategies."""

from repo_indexer.chunking import (
    ChunkingStrategy,
    chunk_file_content,
    filter_chunks,
)


def test_file_chunking():
    """Test one-chunk-per-file strategy."""
    # Test Python file
    py_content = """def main():
    print("Hello world!")

if __name__ == "__main__":
    main()"""

    py_chunks = chunk_file_content(
        file_content=py_content, file_path="/src/main.py", strategy=ChunkingStrategy.FILE
    )
    assert len(py_chunks) == 1  # One chunk for the file
    assert py_chunks[0].chunk_type == "code"
    assert "def main():" in py_chunks[0].content
    assert py_chunks[0].start_line == 0

    # Test Markdown file
    md_content = """# Project Title

This is a test README file.

## About

Some information about the project."""

    md_chunks = chunk_file_content(
        file_content=md_content, file_path="/README.md", strategy=ChunkingStrategy.FILE
    )
    assert len(md_chunks) == 1  # One chunk for the file
    assert md_chunks[0].chunk_type == "documentation"
    assert "# Project Title" in md_chunks[0].content
    assert md_chunks[0].start_line == 0


def test_marker_chunking():
    """Test marker-based chunking strategy."""
    content = """class Calculator:
    def __init__(self):
        self.value = 0

    def add(self, x):
        self.value += x
        return self.value

def main():
    calc = Calculator()
    print(calc.add(5))

if __name__ == "__main__":
    main()"""

    chunks = chunk_file_content(
        file_content=content, file_path="/src/main.py", strategy=ChunkingStrategy.MARKER
    )
    assert len(chunks) > 1  # Should split into multiple chunks

    # Verify class chunk
    class_chunk = next(c for c in chunks if "class Calculator:" in c.content)
    assert class_chunk.chunk_type == "code"
    assert "def __init__" in class_chunk.content
    assert "def add" in class_chunk.content

    # Verify main function chunk
    main_chunk = next(c for c in chunks if "def main():" in c.content)
    assert main_chunk.chunk_type == "code"
    assert "calc = Calculator()" in main_chunk.content


def test_semantic_chunking():
    """Test semantic chunking strategy (currently falls back to marker-based)."""
    content = """# Project Overview

This project provides a calculator implementation.

## Features

- Basic arithmetic operations
- Memory storage
- Scientific functions

## Installation

Follow these steps to install:
1. Clone the repository
2. Run pip install
3. Import and use"""

    chunks = chunk_file_content(
        file_content=content, file_path="/README.md", strategy=ChunkingStrategy.SEMANTIC
    )
    assert len(chunks) > 1  # Should split into multiple chunks

    # Verify sections are split
    overview_chunk = next(c for c in chunks if "# Project Overview" in c.content)
    assert overview_chunk.chunk_type == "documentation"
    assert "calculator implementation" in overview_chunk.content

    features_chunk = next(c for c in chunks if "## Features" in c.content)
    assert features_chunk.chunk_type == "documentation"
    assert "Basic arithmetic" in features_chunk.content


def test_chunking_with_size_limits():
    """Test chunking with size and token limits."""
    # Create a long file that should be split based on size
    content = """# This is a very long file that should be split into multiple chunks
# based on the size and token limits

def function_1():
    \"\"\"This is function 1.\"\"\"
    print("Function 1")
""" + "\n".join(
        f"    print('Line {i}')" for i in range(100)
    )

    chunks = chunk_file_content(
        file_content=content,
        file_path="/src/long.py",
        strategy=ChunkingStrategy.MARKER,
        chunk_size=200,  # Small chunk size to force splitting
        max_tokens=50,  # Small token limit to force splitting
    )
    assert len(chunks) > 1  # Should split into multiple chunks
    assert all(len(c.content) <= 400 for c in chunks)  # Check size limits


def test_filter_chunks():
    """Test chunk filtering functionality."""
    # Create test chunks
    py_content = "def test(): pass"
    md_content = "# Test"
    js_content = "function test() {}"

    chunks = [
        *chunk_file_content(py_content, "/src/test.py"),
        *chunk_file_content(md_content, "/docs/test.md"),
        *chunk_file_content(js_content, "/src/test.js"),
    ]

    # Test file type filtering
    py_chunks = filter_chunks(chunks, file_types=[".py"])
    assert len(py_chunks) == 1
    assert py_chunks[0].source_file == "/src/test.py"

    # Test token count filtering
    small_chunks = filter_chunks(chunks, min_tokens=5)
    assert len(small_chunks) < len(chunks)

    # Test combined filtering
    filtered = filter_chunks(chunks, min_tokens=2, max_tokens=10, file_types=[".py", ".js"])
    assert all(c.source_file.endswith((".py", ".js")) for c in filtered)
    assert all(2 <= len(c.content.split()) <= 10 for c in filtered)
