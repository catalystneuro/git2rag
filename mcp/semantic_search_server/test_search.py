import asyncio
from src.search import search  # your local search function
from src.config import config

async def test():
    result = search(
        query="What is NeuroConv?",
        context="Basic introduction",
        qdrant_url=config.qdrant_url,
        collection_name=config.collection_name,
        keywords=config.keywords,
        qdrant_api_key=config.qdrant_api_key,
        timeout=config.timeout,
        return_digest_summary=config.return_digest_summary,
        return_references=config.return_references,
        limit=config.limit,
        model=config.model,
    )
    print("Result:", result)

asyncio.run(test())
