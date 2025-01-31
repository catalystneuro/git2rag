"""Specialized chunking strategies for different types of repository content."""

import re
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import List, Optional, Union, Tuple


class ChunkingStrategy(Enum):
    """Available chunking strategies."""

    FILE = auto()  # One chunk per file
    MARKER = auto()  # Based on file type markers (classes, functions, sections)
    SEMANTIC = auto()  # LLM-assisted semantic chunking


@dataclass
class Chunk:
    """A chunk of content with metadata."""

    source_file: str
    content_raw: str
    content_processed: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    chunk_type: str = "text"
    context: Optional[str] = None
    embedding_raw: Optional[List[float]] = None
    embedding_processed: Optional[List[float]] = None


def _estimate_tokens(text: str) -> int:
    """Estimate number of tokens in text."""
    return len(text.split())


def filter_chunks(
    chunks: List[Chunk],
    min_tokens: Optional[int] = None,
    max_tokens: Optional[int] = None,
    file_types: Optional[List[str]] = None,
) -> List[Chunk]:
    """Filter chunks based on token count and file types.

    Args:
        chunks: List of chunks to filter
        min_tokens: Minimum number of tokens per chunk
        max_tokens: Maximum number of tokens per chunk
        file_types: List of file extensions to include (e.g. ['.py', '.md'])

    Returns:
        Filtered list of chunks
    """
    # Filter out empty chunks
    filtered = [chunk for chunk in chunks if chunk.content_raw.strip()]

    # Apply token count filters
    if min_tokens is not None or max_tokens is not None:
        result = []
        for chunk in filtered:
            token_count = _estimate_tokens(chunk.content_raw)
            if min_tokens is not None and token_count < min_tokens:
                continue
            if max_tokens is not None and token_count > max_tokens:
                continue
            result.append(chunk)
        filtered = result

    # Apply file type filter
    if file_types:
        # Normalize file types to lowercase with leading dot
        normalized_types = [t if t.startswith(".") else f".{t.lower()}" for t in file_types]
        filtered = [
            chunk
            for chunk in filtered
            if any(chunk.source_file.lower().endswith(t) for t in normalized_types)
        ]

    return filtered


@dataclass(frozen=True)
class ChunkingConfig:
    """Configuration for chunking strategies.

    Args:
        strategy: Chunking strategy to use
        chunk_size: Target size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        max_tokens: Maximum number of tokens per chunk
    """

    strategy: ChunkingStrategy = ChunkingStrategy.FILE
    chunk_size: int = 400
    overlap: int = 50
    max_tokens: int = 512


class BaseChunker:
    """Base class for chunking strategies."""

    def __init__(self, config: ChunkingConfig):
        """Initialize chunker with configuration.

        Args:
            config: Chunking configuration
        """
        self.config = config

    def chunk_content(self, content: str, filepath: str) -> List[Chunk]:
        """Chunk content based on strategy."""
        if self.config.strategy == ChunkingStrategy.FILE:
            return self._chunk_by_file(content, filepath)
        elif self.config.strategy == ChunkingStrategy.MARKER:
            return self._chunk_by_markers(content, filepath)
        elif self.config.strategy == ChunkingStrategy.SEMANTIC:
            return self._chunk_semantically(content, filepath)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.config.strategy}")

    def _chunk_by_file(self, content: str, filepath: str) -> List[Chunk]:
        """Create a single chunk for the entire file."""
        return [
            Chunk(
                content_raw=content,
                source_file=filepath,
                start_line=0,
                end_line=len(content.split("\n")),
                chunk_type=self._get_chunk_type(filepath),
            )
        ]

    def _chunk_by_markers(self, content: str, filepath: str) -> List[Chunk]:
        """Default marker-based chunking."""
        chunks = []
        lines = content.split("\n")
        current_chunk_lines = []
        current_chunk_start = 0

        for i, line in enumerate(lines):
            current_chunk_lines.append(line)
            chunk_text = "\n".join(current_chunk_lines)

            if (
                len(chunk_text) >= self.config.chunk_size
                or _estimate_tokens(chunk_text) >= self.config.max_tokens * 0.8
            ):

                # Try to find a good break point
                break_points = ["\n\n", ". ", "\n", " "]
                for break_point in break_points:
                    pos = chunk_text.rfind(break_point, 0, self.config.chunk_size)
                    if pos > self.config.chunk_size // 2:
                        break_text = chunk_text[: pos + len(break_point)]
                        remainder = chunk_text[pos + len(break_point) :]

                        chunks.append(
                            Chunk(
                                content=break_text,
                                source_file=filepath,
                                start_line=current_chunk_start,
                                end_line=i,
                                chunk_type=self._get_chunk_type(filepath),
                            )
                        )

                        current_chunk_lines = remainder.split("\n")
                        current_chunk_start = i - len(current_chunk_lines) + 1
                        break
                else:
                    chunks.append(
                        Chunk(
                            content=chunk_text[: self.config.chunk_size],
                            source_file=filepath,
                            start_line=current_chunk_start,
                            end_line=i,
                            chunk_type=self._get_chunk_type(filepath),
                        )
                    )
                    current_chunk_lines = [chunk_text[self.config.chunk_size :]]
                    current_chunk_start = i

        if current_chunk_lines:
            chunks.append(
                Chunk(
                    content="\n".join(current_chunk_lines),
                    source_file=filepath,
                    start_line=current_chunk_start,
                    end_line=len(lines),
                    chunk_type=self._get_chunk_type(filepath),
                )
            )

        return chunks

    def _chunk_semantically(self, content: str, filepath: str) -> List[Chunk]:
        """Placeholder for LLM-assisted semantic chunking."""
        # TODO: Implement semantic chunking with LLM assistance
        return self._chunk_by_markers(content, filepath)

    def _get_chunk_type(self, filepath: str) -> str:
        """Get chunk type based on file extension."""
        ext = Path(filepath).suffix.lower()
        if ext in {".py", ".js", ".java", ".cpp", ".h", ".hpp"}:
            return "code"
        elif ext in {".md", ".rst", ".txt"}:
            return "documentation"
        else:
            return "text"


class CodeChunker(BaseChunker):
    """Chunking strategy for source code files."""

    def _chunk_by_markers(self, content: str, filepath: str) -> List[Chunk]:
        """Chunk code by classes and functions."""
        chunks = []
        lines = content.split("\n")
        current_chunk_lines = []
        current_chunk_start = 0
        in_function = False
        in_class = False
        current_context = None

        for i, line in enumerate(lines):
            # Handle class definitions
            if re.match(r"^\s*class\s+", line):
                if current_chunk_lines and not in_class:  # Only end chunk if not already in a class
                    chunks.append(
                        Chunk(
                            content="\n".join(current_chunk_lines),
                            source_file=filepath,
                            start_line=current_chunk_start,
                            end_line=i,
                            chunk_type="code",
                            context=current_context,
                        )
                    )
                    current_chunk_lines = []
                in_class = True
                current_context = line.strip()
                current_chunk_start = i
            # Handle function definitions outside classes
            elif re.match(r"^\s*def\s+", line) and not in_class:
                if current_chunk_lines:
                    chunks.append(
                        Chunk(
                            content="\n".join(current_chunk_lines),
                            source_file=filepath,
                            start_line=current_chunk_start,
                            end_line=i,
                            chunk_type="code",
                            context=current_context,
                        )
                    )
                    current_chunk_lines = []
                in_function = True
                current_context = line.strip()
                current_chunk_start = i

            current_chunk_lines.append(line)
            chunk_text = "\n".join(current_chunk_lines)

            # Check if we should complete the current chunk
            chunk_complete = False
            if not in_class:  # Only auto-split chunks outside classes
                if in_function and re.match(r"^\s*$", line):
                    in_function = False
                    chunk_complete = True
                elif (
                    len(chunk_text) >= self.config.chunk_size
                    or _estimate_tokens(chunk_text) >= self.config.max_tokens * 0.8
                ):
                    chunk_complete = True
            elif (
                in_class
                and not line.strip()
                and i + 1 < len(lines)
                and not lines[i + 1].startswith(" ")
            ):
                # End class when we hit an empty line followed by non-indented line
                in_class = False
                in_function = False
                chunk_complete = True

            if chunk_complete and current_chunk_lines:
                chunks.append(
                    Chunk(
                        content="\n".join(current_chunk_lines),
                        source_file=filepath,
                        start_line=current_chunk_start,
                        end_line=i + 1,
                        chunk_type="code",
                        context=current_context,
                    )
                )
                current_chunk_lines = []
                current_chunk_start = i + 1

        if current_chunk_lines:
            chunks.append(
                Chunk(
                    content="\n".join(current_chunk_lines),
                    source_file=filepath,
                    start_line=current_chunk_start,
                    end_line=len(lines),
                    chunk_type="code",
                    context=current_context,
                )
            )

        return chunks


class DocumentationChunker(BaseChunker):
    """Chunking strategy for documentation files."""

    def _chunk_by_markers(self, content: str, filepath: str) -> List[Chunk]:
        """Chunk documentation by sections."""
        chunks = []
        lines = content.split("\n")
        current_chunk_lines = []
        current_chunk_start = 0
        current_section = None

        for i, line in enumerate(lines):
            if re.match(r"^#+\s+", line) or re.match(r"^[=-]+$", line):
                if current_chunk_lines:
                    chunks.append(
                        Chunk(
                            content="\n".join(current_chunk_lines),
                            source_file=filepath,
                            start_line=current_chunk_start,
                            end_line=i,
                            chunk_type="documentation",
                            context=current_section,
                        )
                    )
                    current_chunk_lines = []
                current_section = line.strip()
                current_chunk_start = i

            current_chunk_lines.append(line)
            chunk_text = "\n".join(current_chunk_lines)

            if (
                len(chunk_text) >= self.config.chunk_size
                or _estimate_tokens(chunk_text) >= self.config.max_tokens * 0.8
            ):
                break_points = ["\n\n", ". ", "\n", " "]
                for break_point in break_points:
                    pos = chunk_text.rfind(break_point, 0, self.config.chunk_size)
                    if pos > self.config.chunk_size // 2:
                        break_text = chunk_text[: pos + len(break_point)]
                        remainder = chunk_text[pos + len(break_point) :]

                        chunks.append(
                            Chunk(
                                content=break_text,
                                source_file=filepath,
                                start_line=current_chunk_start,
                                end_line=i,
                                chunk_type="documentation",
                                context=current_section,
                            )
                        )

                        current_chunk_lines = remainder.split("\n")
                        current_chunk_start = i - len(current_chunk_lines) + 1
                        break

        if current_chunk_lines:
            chunks.append(
                Chunk(
                    content="\n".join(current_chunk_lines),
                    source_file=filepath,
                    start_line=current_chunk_start,
                    end_line=len(lines),
                    chunk_type="documentation",
                    context=current_section,
                )
            )

        return chunks


def get_chunker(filepath: str, config: ChunkingConfig) -> BaseChunker:
    """Get appropriate chunker for a file type."""
    ext = Path(filepath).suffix.lower()

    if ext in {".py", ".js", ".java", ".cpp", ".h", ".hpp"}:
        return CodeChunker(config)
    elif ext in {".md", ".rst", ".txt"}:
        return DocumentationChunker(config)
    else:
        return BaseChunker(config)


def chunk_file_content(
    file_content: str,
    file_path: str,
    strategy: ChunkingStrategy = ChunkingStrategy.FILE,
    chunk_size: int = 400,
    overlap: int = 50,
    max_tokens: int = 512,
) -> List[Chunk]:
    """Process repository content and return chunks.

    Args:
        file_content: Content of the file to chunk
        file_path: Path to the file
        strategy: Chunking strategy to use
        chunk_size: Target size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        max_tokens: Maximum number of tokens per chunk

    Returns:
        List of content chunks
    """
    config = ChunkingConfig(
        strategy=strategy, chunk_size=chunk_size, overlap=overlap, max_tokens=max_tokens
    )

    chunker = get_chunker(
        filepath=file_path,
        config=config,
    )

    return chunker.chunk_content(
        content=file_content,
        filepath=file_path,
    )
