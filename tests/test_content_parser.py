"""Tests for content parser functions."""

from repo_indexer.content_parser import break_into_files


def test_break_into_files():
    """Test breaking content into files."""
    content = """================================================
File: /README.md
================================================
[![PyPI version](https://badge.fury.io/py/neuroconv.svg)](https://badge.fury.io/py/neuroconv.svg)
![Daily Tests](https://github.com/catalystneuro/neuroconv/actions/workflows/dailies.yml/badge.svg)

## About

NeuroConv is a Python package for converting neurophysiology data.

================================================
File: /docs/api/interfaces.rst
================================================
Interfaces
==========

.. toctree::
    :maxdepth: 4

    interfaces.ecephys
    interfaces.ophys
"""

    # Test with no filters
    files = break_into_files(content)
    assert len(files) == 2

    # Check first file
    assert files[0].startswith("/README.md\n")
    assert "About" in files[0]

    # Check second file
    assert files[1].startswith("/docs/api/interfaces.rst\n")
    assert "toctree" in files[1]

    # Test with include filter
    files = break_into_files(content, include_extensions=[".rst"])
    assert len(files) == 1
    assert files[0].startswith("/docs/api/interfaces.rst\n")

    # Test with exclude filter
    files = break_into_files(content, exclude_extensions=[".rst"])
    assert len(files) == 1
    assert files[0].startswith("/README.md\n")

    # Test with both filters
    files = break_into_files(
        content, include_extensions=[".md", ".rst"], exclude_extensions=[".rst"]
    )
    assert len(files) == 1
    assert files[0].startswith("/README.md\n")
