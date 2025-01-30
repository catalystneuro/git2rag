# repo-indexer Examples

This directory contains example notebooks and scripts demonstrating how to use repo-indexer in various scenarios.

## Available Examples

### index_neuroconv.ipynb
A Jupyter notebook showing how to:
- Index the NeuroCONV repository (a neurophysiology data conversion tool)
- Search through code, documentation, and configuration
- Find specific functionality like data conversion functions
- Explore test cases and error handling
- Clean up indexed data

## Prerequisites

To run these examples, you need:
1. Python 3.8+
2. Jupyter Notebook/Lab
3. Running Qdrant instance (e.g., at http://localhost:6333)
4. Either:
   - OpenAI API key (export as OPENAI_API_KEY) for OpenAI embeddings
   - Local embedding server (using text-embeddings-inference) for free, local embeddings

## Running Examples

1. Install dependencies:
```bash
pip install repo-indexer jupyter
```

2. Start Jupyter:
```bash
jupyter notebook
```

3. Navigate to the examples directory and open the desired notebook

## Notes

- When running in Jupyter notebooks, nest_asyncio is required and must be initialized:
  ```python
  import nest_asyncio
  nest_asyncio.apply()
  ```
- Examples support both OpenAI embeddings (text-embedding-ada-002) and local embeddings (multilingual-e5-small)
- For local embeddings:
  1. Start the server:
     ```bash
     docker run -p 8001:80 ghcr.io/huggingface/text-embeddings-inference:cpu-0.3 --model-id intfloat/multilingual-e5-small
     ```
  2. Configure server URL (choose one):
     - Use default: http://localhost:8001/v1/embeddings
     - Set environment variable: `export EMBEDDING_SERVER_URL="http://localhost:8001/v1/embeddings"`
     - Create custom generator: `embedding_generator=LocalE5Embeddings(url="http://localhost:8001/v1/embeddings")`
- Make sure your Qdrant instance is running before executing examples
- Some examples may take time to run (e.g., indexing large repositories)
- Remember to clean up collections after running examples to free up space
