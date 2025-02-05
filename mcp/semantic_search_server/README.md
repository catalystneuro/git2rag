# NeurConv Semantic Search MCP Server

This MCP server provides semantic search functionality specifically designed to help users learn about NeurConv. It uses advanced natural language processing to understand queries, expand search terms, and provide relevant information from the NeurConv documentation and related resources.

## Features

- Semantic search optimized for NeurConv-related queries
- Query expansion using LLM to improve search accuracy
- Context-aware searching to provide more relevant results
- Automatic filtering of results for relevance
- Result summarization for quick understanding
- Support for multiple vector embeddings in search
- Hybrid search combining semantic and keyword-based approaches

## Installation

1. Make sure you have Python 3.11+ installed
2. Install the required packages:
```bash
pip install mcp uv
```

## Running the Server

0. Export the required environment variables:
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

1. Navigate to the project directory:
```bash
cd semantic_search_server
```

2. Run the server:
```bash
mcp dev src/main.py
```

or in editable mode:
```bash
mcp dev src/main.py --with-editable .
```

## Using the Server

The server provides one tool:

### semantic_search

Performs semantic search to help learn about NeurConv.

Input schema:
```json
{
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Search query text for semantic search"
        },
        "context": {
            "type": "string",
            "description": "Context in which the query is relevant"
        }
    },
    "required": ["query", "context"]
}
```

Example usage:
```json
{
    "query": "How do I convert my neurophysiology data to NWB format?",
    "context": "I'm trying to standardize my lab's data collection process"
}
```

The response will include:
- Alternative search queries generated to expand the search
- Filtered and ranked search results
- A summary of the relevant information (if enabled)
- References to source documentation (if enabled)

## Configuration

The server can be configured through environment variables:

- `QDRANT_URL`: URL of your Qdrant instance (default: "http://localhost:6333")
- `QDRANT_API_KEY`: API key for Qdrant authentication (optional)
- `COLLECTION_NAME`: Name of the Qdrant collection to search (default: "default")
- `MODEL`: LLM model to use (default: "openai/o3-mini")
- `TIMEOUT`: Search timeout in seconds (default: 60.0)
- `LIMIT`: Maximum number of results to return (default: 10)

## MCP Configuration

To use this server with an MCP client, add the following configuration:

```json
{
    "mcpServers": {
        "neuroconv-search": {
            "command": "python",
            "args": ["/path/to/semantic_search_server/src/main.py"],
            "env": {
                "QDRANT_URL": "your-qdrant-url",
                "QDRANT_API_KEY": "your-api-key",
                "COLLECTION_NAME": "neuroconv-docs"
            }
        }
    }
}
```

## Example Queries

Here are some example queries you can try:

1. Basic usage:
```json
{
    "query": "What file formats does NeurConv support?",
    "context": "I need to convert my existing data files"
}
```

2. Specific conversion:
```json
{
    "query": "How do I convert Spike2 files to NWB?",
    "context": "I have experimental data recorded in Spike2"
}
```

3. Best practices:
```json
{
    "query": "What are the recommended practices for organizing my NWB files?",
    "context": "I'm setting up a new data management system"
}
```
