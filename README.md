# Git2RAG: Repository Indexer for RAG

A tool for indexing GitHub repositories to make their content accessible through RAG (Retrieval-Augmented Generation) systems. This tool processes repository content, including documentation, source code, and configuration files, into semantically meaningful chunks that can be efficiently searched using vector similarity.

## Features

- 📚 Smart content chunking
  - Multiple chunking strategies (FILE, MARKER, SEMANTIC)
  - Context-aware splitting for different file types
  - Code chunks preserve class/function boundaries
  - Documentation chunks respect section structure
  - LLM-assisted semantic chunking for optimal content breaks
  - Configurable chunk sizes and overlap

- 🔍 Efficient vector search
  - Multiple embedding providers through LiteLLM
  - Qdrant vector database for fast retrieval
  - Filter by content type or file extension
  - Preserves file context and line numbers

- 🎯 Specialized handling for:
  - Python source code (preserves class/function context)
  - JavaScript, Java, C++, and other code files
  - Markdown documentation (respects headers)
  - RST documentation
  - Configuration files (YAML, JSON, etc.)

- 🤖 Content processing
  - Optional chunk summarization using LLMs
  - Raw and processed content embeddings
  - Batch processing for efficiency
  - Customizable filtering options

## Installation

1. Install the package:
```bash
pip install git2rag
```

2. Start Qdrant:
```bash
docker-compose up -d
```

3. Set up environment variables:
```bash
# Required for your chosen embedding provider
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
# Basic indexing
git2rag index https://github.com/username/repo

# With semantic chunking
git2rag index https://github.com/username/repo --strategy semantic

# With chunk summarization
git2rag index https://github.com/username/repo --summarize

# Customize chunking
git2rag index https://github.com/username/repo \
  --chunk-size 500 \
  --overlap 50 \
  --max-tokens 1000
```

2. Search content:
```bash
# Basic search
git2rag search "How do I implement feature X?"

# Filter by content type
git2rag search "error handling" --type code

# Filter by file extension
git2rag search "configuration options" --ext yaml

# Search in processed content
git2rag search "query" --vector-name processed
```

### Python API

```python
from git2rag import RepoIndexer, ChunkingStrategy

# Initialize indexer
indexer = RepoIndexer(
    qdrant_url="http://localhost:6333",
    api_key="your-api-key",  # For your chosen embedding provider
    embedding_model="text-embedding-ada-002",  # Or any model supported by LiteLLM
)

# Index with default settings
indexer.index_repository("https://github.com/username/repo")

# Index with semantic chunking and summarization
indexer.index_repository(
    repo_url="https://github.com/username/repo",
    strategy=ChunkingStrategy.SEMANTIC,
    summarize=True,
    summary_model="openai/gpt-4o-mini",
    embedding_from="both",  # Generate embeddings for both raw and processed content
)

# Search
results = indexer.search(
    query="How do I implement feature X?",
    limit=5,
    chunk_type="code",  # Optional: Filter by content type
    file_extension="py",  # Optional: Filter by extension
    vector_name="processed",  # Optional: Search in processed content
)

# Access results
for result in results:
    print(f"Score: {result['score']}")
    print(f"File: {result['file']}")
    print(f"Lines: {result['lines']}")
    print(f"Content: {result['content']}")
    print(f"Context: {result['context']}")
```

## Development

The project consists of several components:

1. `chunking/`: Smart content chunking strategies
   - `base.py`: Base chunking implementation and utilities
   - `llm_chunking.py`: LLM-assisted semantic chunking

2. `clients/`: Database client implementations
   - `qdrant.py`: Qdrant vector database integration

3. Core functionality:
   - `indexer.py`: Main indexing and search functionality
   - `embeddings.py`: Embedding generation through LiteLLM
   - `content_parser.py`: Repository content parsing
   - `summarizer.py`: Content summarization using LLMs

### Running Tests

```bash
# Install development dependencies
pip install git2rag[dev]

# Run tests
pytest

# Run specific test files
pytest tests/test_chunking.py
pytest tests/test_indexer.py
```

## Contributing

Contributions are welcome! Some areas that could use improvement:

- Additional chunking strategies for other file types
- Support for more embedding providers through LiteLLM
- Improved context preservation
- Batch processing for large repositories
- Caching and incremental updates
- Additional vector database integrations

## License

MIT License - feel free to use and modify as needed.
