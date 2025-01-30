"""Command-line interface for repository indexing."""

import argparse
import os
from typing import Optional

from .embeddings import (
    ANTHROPIC_CLAUDE,
    AZURE_ADA,
    COHERE_EMBED,
    LOCAL_E5_SMALL,
    OPENAI_ADA_002,
)
from .indexer import RepoIndexer, get_vector_size


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Index and search GitHub repositories for RAG applications."
    )

    # Main commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Index command
    index_parser = subparsers.add_parser("index", help="Index a repository")
    index_parser.add_argument("repo_url", help="URL of the repository to index")
    index_parser.add_argument(
        "--collection",
        default="repo_content",
        help="Name of the Qdrant collection (default: repo_content)",
    )
    index_parser.add_argument(
        "--upsert-batch-size",
        type=int,
        default=100,
        help="Number of vectors to upsert at once (default: 100)",
    )

    # Search command
    search_parser = subparsers.add_parser("search", help="Search indexed content")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument(
        "--collection",
        default="repo_content",
        help="Name of the Qdrant collection to search (default: repo_content)",
    )
    search_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of results (default: 5)",
    )
    search_parser.add_argument(
        "--type",
        choices=["code", "documentation", "text"],
        help="Filter by content type",
    )
    search_parser.add_argument(
        "--ext",
        help="Filter by file extension (e.g., py, md, rst)",
    )

    # Common options
    for p in [index_parser, search_parser]:
        p.add_argument(
            "--qdrant-url",
            default=os.getenv("QDRANT_URL", "http://localhost:6333"),
            help="Qdrant server URL (default: $QDRANT_URL or http://localhost:6333)",
        )
        p.add_argument(
            "--qdrant-api-key",
            default=os.getenv("QDRANT_API_KEY"),
            help="Qdrant API key (default: $QDRANT_API_KEY)",
        )
        p.add_argument(
            "--api-key",
            default=os.getenv("OPENAI_API_KEY"),
            help="API key for the embedding provider (default: $OPENAI_API_KEY)",
        )
        p.add_argument(
            "--embedding-model",
            default=LOCAL_E5_SMALL,
            choices=[
                LOCAL_E5_SMALL,  # Default to local E5 model
                OPENAI_ADA_002,
                AZURE_ADA,
                COHERE_EMBED,
                ANTHROPIC_CLAUDE,
            ],
            help=f"Embedding model to use (default: {LOCAL_E5_SMALL})",
        )
        p.add_argument(
            "--embedding-url",
            default="http://0.0.0.0:8001/v1/embeddings",
            help="URL for local embedding server (default: http://0.0.0.0:8001/v1/embeddings)",
        )
        p.add_argument(
            "--vector-size",
            type=int,
            help="Size of embedding vectors (default: auto-detected based on model)",
        )

    return parser


def format_result(result: dict, show_context: bool = True) -> str:
    """Format a search result for display."""
    output = []

    # Score and file
    output.append(f"Score: {result['score']:.3f}")
    output.append(f"File: {result['file']}")

    # Context (if available and requested)
    if show_context and result["context"]:
        output.append(f"Context: {result['context']}")

    # Line numbers (if available)
    if result["lines"][0] is not None:
        output.append(f"Lines: {result['lines'][0]}-{result['lines'][1]}")

    # Content preview
    content = result["content"]
    if len(content) > 200:
        content = content[:200] + "..."
    output.append("\nContent:")
    output.append(content)

    return "\n".join(output)


def main(args: Optional[argparse.Namespace] = None) -> None:
    """Main entry point."""
    if args is None:
        parser = create_parser()
        args = parser.parse_args()

    # Initialize indexer
    indexer = RepoIndexer(
        qdrant_url=args.qdrant_url,
        api_key=args.api_key,
        qdrant_api_key=args.qdrant_api_key,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        vector_size=args.vector_size,
        upsert_batch_size=args.upsert_batch_size if hasattr(args, "upsert_batch_size") else 100,
    )

    if args.command == "index":
        # Index repository
        indexer.index_repository(args.repo_url)

    elif args.command == "search":
        # Search repository
        results = indexer.search(
            query=args.query,
            limit=args.limit,
            chunk_type=args.type,
            file_extension=args.ext,
        )

        # Print results
        for i, result in enumerate(results, 1):
            if i > 1:
                print("\n" + "-" * 40 + "\n")  # Separator between results
            print(format_result(result))

    else:
        parser = create_parser()
        parser.print_help()


if __name__ == "__main__":
    main()
