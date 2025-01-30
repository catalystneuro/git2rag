# Setup Guide

This guide walks you through setting up the Repository Indexer with a local Qdrant vector database.

## Prerequisites

- Docker and Docker Compose
- Python 3.8 or higher
- pip (Python package installer)

## Setup Steps

### 1. Start Qdrant Database

The project includes a `docker-compose.yml` that configures Qdrant with persistent storage and the necessary API ports.

```bash
# Start Qdrant in the background
docker-compose up -d

# Verify it's running
docker-compose ps
```

You should see Qdrant running on:
- REST API: http://localhost:6333
- GRPC API: localhost:6334

The database files will be stored in `./qdrant_storage/` directory.

### 2. Install Python Dependencies

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install "qdrant-client" "numpy" "openai>=1.0.0" "gitingest"
```

### 3. Configure Environment

Create a `.env` file in your project directory:

```bash
# .env
OPENAI_API_KEY="your-openai-api-key"
QDRANT_URL="http://localhost:6333"
```

Or set environment variables directly:

```bash
# Linux/macOS
export OPENAI_API_KEY="your-openai-api-key"
export QDRANT_URL="http://localhost:6333"

# Windows (PowerShell)
$env:OPENAI_API_KEY="your-openai-api-key"
$env:QDRANT_URL="http://localhost:6333"
```

### 4. Verify Setup

Run the test script to verify everything is working:

```bash
python repo_indexer.py
```

You should see output showing:
1. Successful connection to Qdrant
2. Repository content being processed
3. Embeddings being generated
4. Search results being returned

## Qdrant Management

### Monitoring

Access the Qdrant OpenAPI dashboard at http://localhost:6333/dashboard to:
- View collections
- Monitor metrics
- Execute queries directly

### Data Persistence

Qdrant data is persisted in `./qdrant_storage/`. To start fresh:

1. Stop Qdrant:
```bash
docker-compose down
```

2. Remove storage:
```bash
rm -rf ./qdrant_storage
```

3. Restart Qdrant:
```bash
docker-compose up -d
```

### Configuration

The default configuration should work for most use cases, but you can customize Qdrant by adding configuration to `docker-compose.yml`:

```yaml
services:
  qdrant:
    environment:
      - QDRANT_ALLOW_RECOVERY_MODE=true
      - QDRANT_CPU_BUDGET=4  # Number of CPU cores to use
      - QDRANT_MEMORY_BUDGET=4000000000  # Memory budget in bytes (4GB)
```

See [Qdrant Configuration](https://qdrant.tech/documentation/configuration/) for more options.

## Troubleshooting

### Common Issues

1. **Qdrant Connection Failed**
   - Verify Docker is running
   - Check `docker-compose ps` output
   - Ensure ports 6333-6334 are not in use
   - Try restarting with `docker-compose restart`

2. **OpenAI API Issues**
   - Verify API key is set correctly
   - Check API key has required permissions
   - Monitor rate limits in OpenAI dashboard

3. **Memory Issues**
   - Adjust Qdrant memory budget in docker-compose.yml
   - Process large repositories in batches
   - Monitor Docker resource usage

### Logs

View Qdrant logs:
```bash
docker-compose logs -f qdrant
```

### Support

If you encounter issues:
1. Check Qdrant logs
2. Verify environment variables
3. Ensure all dependencies are installed
4. Open an issue on GitHub with:
   - Error messages
   - Environment details
   - Steps to reproduce
