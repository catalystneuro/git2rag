# Quick Start Guide

This guide shows you how to quickly get started with repo-indexer to index and search through your code repositories.

## Prerequisites

- Python 3.8+
- Running Qdrant instance (e.g., at http://localhost:6333)
- OpenAI API access

## Environment Setup

Export the required environment variables:

```bash
# OpenAI API key for embeddings
export OPENAI_API_KEY="your-api-key"

# Optional: Qdrant API key if your instance requires authentication
export QDRANT_API_KEY="your-qdrant-key"
```

## Installation

```bash
pip install repo-indexer
```

## Basic Usage

### Initialize Indexer

```python
from repo_indexer.indexer import RepoIndexer
from repo_indexer.embeddings import OPENAI_ADA_002

# Initialize with OpenAI embeddings
indexer = RepoIndexer(
    qdrant_url="http://localhost:6333",
    api_key=os.getenv("OPENAI_API_KEY"),
    qdrant_api_key=os.getenv("QDRANT_API_KEY"),  # Optional
    embedding_model=OPENAI_ADA_002
)
```

### Index a Repository

```python
# Index a local repository
indexer.index_repository("/path/to/local/repo")

# Or index a GitHub repository
indexer.index_repository("https://github.com/username/repo.git")
```

### Search Code

```python
# Basic search
results = indexer.search(
    query="function to handle HTTP requests",
    limit=5  # Number of results to return
)

# Print results
for result in results:
    print(f"\nScore: {result['score']:.3f}")
    print(f"File: {result['file']}")
    print(f"Lines: {result['lines'][0]}-{result['lines'][1]}")
    print(f"Content:\n{result['content']}")

# Search with filters
results = indexer.search(
    query="database configuration",
    limit=5,
    chunk_type="configuration",  # Filter by content type
    file_extension="py"  # Filter by file extension
)
```

## Advanced Usage

### List Collections

```python
from repo_indexer.clients import QdrantManager

# Initialize Qdrant manager
manager = QdrantManager(
    url="http://localhost:6333",
    api_key=os.getenv("QDRANT_API_KEY")  # Optional
)

# List all collections
collections = manager.list_collections()
for collection in collections:
    print(f"Collection: {collection['name']}")
    print(f"Vectors: {collection['vectors_count']}")
    print(f"Status: {collection['status']}")
```

### Clean Up

```python
# Delete a collection
manager.delete_collection("repo_content")
```

## Content Types

repo-indexer automatically detects and categorizes content:

- `code`: Source code files
- `documentation`: Markdown, RST, and other documentation
- `configuration`: Config files, env files, etc.

You can use these types in search filters to narrow down results.

## Error Handling

```python
try:
    indexer.index_repository("/path/to/repo")
except Exception as e:
    print(f"Error indexing repository: {e}")
```

For more detailed information, check the [full documentation](https://github.com/yourusername/repo-indexer#readme).
