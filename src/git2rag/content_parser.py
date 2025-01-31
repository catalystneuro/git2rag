"""Functions for parsing repository content."""

from pathlib import Path
from typing import List, Optional, Tuple


def break_into_files(
    content: str,
    include_extensions: Optional[List[str]] = None,
    exclude_extensions: Optional[List[str]] = None,
) -> List[Tuple[str, str]]:
    """Break repository content into individual files.

    Args:
        content: Raw repository content with file markers
        include_extensions: List of file extensions to include (e.g. ['.py', '.md'])
        exclude_extensions: List of file extensions to exclude

    Returns:
        List of (filepath, content) tuples
    """
    files = []
    current_filepath = None
    current_content = []
    marker = "=" * 48

    lines = content.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]

        if line == marker:
            # Skip the "File:" line and the next marker
            if i + 2 < len(lines) and lines[i + 1].startswith("File: "):
                current_filepath = lines[i + 1].replace("File: ", "").strip()
                i += 3  # Skip both markers and the File: line
                current_content = []

                # Read content until next marker or end
                while i < len(lines) and lines[i] != marker:
                    current_content.append(lines[i])
                    i += 1

                if current_content:
                    # Check file extension filters
                    ext = Path(current_filepath).suffix.lower()
                    should_include = True

                    if include_extensions:
                        normalized_includes = [
                            e if e.startswith(".") else f".{e.lower()}" for e in include_extensions
                        ]
                        should_include = ext in normalized_includes

                    if exclude_extensions and should_include:
                        normalized_excludes = [
                            e if e.startswith(".") else f".{e.lower()}" for e in exclude_extensions
                        ]
                        should_include = ext not in normalized_excludes

                    if should_include:
                        files.append((current_filepath, "\n".join(current_content)))
                continue
        i += 1

    return files
