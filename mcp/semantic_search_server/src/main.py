#!/usr/bin/env python3
import asyncio
from mcp.server.fastmcp import FastMCP

from search import search
from config import config


mcp = FastMCP("semantic-search-server", version="0.1.0")


@mcp.tool()
def search_about_neuroconv(query: str, context: str) -> str:
    """
    Search to learn about NeuroConv.

    Args:
      query: The search query text.
      context: The relevant context for the query.

    Returns:
      The string representation of the search result.
    """
    result = search(
        query=query,
        context=context,
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
    return str(result)


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(mcp.run())
    finally:
        loop.close()
