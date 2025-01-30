# Repository Indexer for RAG

A tool for indexing GitHub repositories to make their content accessible through RAG (Retrieval-Augmented Generation) systems. This tool processes repository content, including documentation, source code, and configuration files, into semantically meaningful chunks that can be efficiently searched using vector similarity.

## Features

- 📚 Smart content chunking
  - Context-aware splitting for different file types
  - Code chunks preserve class/function boundaries
  - Documentation chunks respect section structure
  - Configurable chunk sizes and overlap

- 🔍 Efficient vector search
  - Multiple embedding providers through LiteLLM
  - Qdrant vector database for fast retrieval
  - Filter by content type or file extension
  - Preserves file context and line numbers

- 🎯 Specialized handling for:
  - Python source code (preserves class/function context)
  - Markdown documentation (respects headers)
  - RST documentation
  - Configuration files (YAML, JSON, etc.)

## Installation

1. Install dependencies:
```bash
pip install repo-indexer
```

2. Start Qdrant:
```bash
docker-compose up -d
```

3. Set up environment variables:
```bash
# Required for your chosen embedding provider:
export OPENAI_API_KEY="your-api-key"      # For OpenAI
export AZURE_API_KEY="your-api-key"       # For Azure
export COHERE_API_KEY="your-api-key"      # For Cohere
export ANTHROPIC_API_KEY="your-api-key"   # For Anthropic

# Optional:
export QDRANT_URL="http://localhost:6333"  # Default Qdrant URL
export QDRANT_API_KEY="your-qdrant-api-key"  # If using authentication
```

## Usage

### Command Line

1. Index a repository:
```bash
# Using OpenAI (default)
repo-indexer index https://github.com/username/repo

# Using Azure
repo-indexer index https://github.com/username/repo \
  --embedding-model azure/text-embedding-ada-002 \
  --api-key your-azure-key

# Using Cohere
repo-indexer index https://github.com/username/repo \
  --embedding-model cohere/embed-english-v3.0 \
  --api-key your-cohere-key
```

2. Search content:
```bash
# Basic search
repo-indexer search "How do I implement feature X?"

# Filter by content type
repo-indexer search "error handling" --type code

# Filter by file extension
repo-indexer search "configuration options" --ext yaml

# Specify embedding model
repo-indexer search "query" --embedding-model azure/text-embedding-ada-002
```

### Python API

```python
from repo_indexer import RepoIndexer, AZURE_ADA, COHERE_EMBED

# Using OpenAI (default)
indexer = RepoIndexer(
    qdrant_url="http://localhost:6333",
    api_key="your-openai-key",
)

# Using Azure
indexer = RepoIndexer(
    qdrant_url="http://localhost:6333",
    api_key="your-azure-key",
    embedding_model=AZURE_ADA,
)

# Using Cohere
indexer = RepoIndexer(
    qdrant_url="http://localhost:6333",
    api_key="your-cohere-key",
    embedding_model=COHERE_EMBED,
)

# Index and search
indexer.index_repository("https://github.com/username/repo")
results = indexer.search(
    query="How do I implement feature X?",
    limit=5,
    chunk_type="code",  # Optional: Filter by content type
    file_extension="py",  # Optional: Filter by extension
)
```

### Custom Embedding Providers

You can implement custom embedding providers by implementing the `EmbeddingGenerator` protocol:

```python
from repo_indexer import EmbeddingGenerator
from typing import List

class CustomEmbeddings(EmbeddingGenerator):
    def generate(self, texts: List[str]) -> List[List[float]]:
        # Your embedding logic here
        return embeddings

indexer = RepoIndexer(
    qdrant_url="http://localhost:6333",
    embedding_generator=CustomEmbeddings(),
)
```

## Development

The project consists of several components:

1. `chunking.py`: Smart content chunking strategies
2. `embeddings.py`: Embedding generation with multiple providers
3. `indexer.py`: Main indexing and search functionality
4. `cli.py`: Command-line interface

### Running Tests

```bash
# Install development dependencies
pip install repo-indexer[dev]

# Run tests
pytest
```

## Contributing

Contributions are welcome! Some areas that could use improvement:

- Additional chunking strategies for other file types
- Support for more embedding providers
- Improved context preservation
- Batch processing for large repositories
- Caching and incremental updates

## License

MIT License - feel free to use and modify as needed.
