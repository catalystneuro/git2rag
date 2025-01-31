"""Main indexing and search functionality."""

import os
from dataclasses import asdict
from typing import Dict, List, Optional, Any

from gitingest import ingest
from qdrant_client.models import PointStruct
from pathlib import Path

from .chunking import (
    ChunkingStrategy,
    Chunk,
    chunk_file_content,
    filter_chunks,
)
from .content_parser import break_into_files
from .clients import QdrantManager
from .embeddings import generate_embeddings
from .summarizer import summarize_content


class RepoIndexer:
    """Index and search repository content."""

    def __init__(
        self,
        qdrant_url: str,
        api_key: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        collection_name: str = "repo_content",
        embedding_model: str = "text-embedding-ada-002",
    ):
        """Initialize the indexer.

        Args:
            qdrant_url: URL of the Qdrant server
            api_key: API key for the embedding provider
            qdrant_api_key: API key for Qdrant server
            collection_name: Name of the Qdrant collection
            embedding_model: Model identifier for embeddings
        """
        # Store repository metadata
        self.repositories: Dict[str, Dict[str, Any]] = {}

        # Initialize Qdrant manager
        self.qdrant = QdrantManager(
            url=qdrant_url,
            api_key=qdrant_api_key,
            batch_size=100  # Default batch size for Qdrant operations
        )
        self.collection_name = collection_name

        # Store configuration
        self.embedding_model = embedding_model
        self.api_key = api_key

        # Set vector size based on model (OpenAI models)
        self.vector_size = 1536  # OpenAI models use 1536 dimensions

        # Ensure collection exists
        self.qdrant.create_collection(
            name=self.collection_name,
            vector_size=self.vector_size
        )

    def parse_repo(
        self,
        repo_url: str,
        include_extensions: Optional[List[str]] = None,
        exclude_extensions: Optional[List[str]] = None,
    ) -> None:
        """Parse a repository and store its raw content.

        Args:
            repo_url: URL or path of the repository
            include_extensions: List of file extensions to include (e.g. ['.py', '.md'])
            exclude_extensions: List of file extensions to exclude
        """
        print(f"Parsing repository: {repo_url}")
        summary, tree, full_content = ingest(repo_url)
        files_content = break_into_files(
            full_content,
            include_extensions=include_extensions,
            exclude_extensions=exclude_extensions,
        )

        # Store metadata
        self.repositories[repo_url] = {
            "summary": summary,
            "tree": tree,
            "content": files_content,
            "chunks": []
        }

    def _estimate_tokens(self, text: str) -> int:
        """Estimate number of tokens in text.

        This is a rough estimate based on whitespace-splitting times 3.
        Real token count may be different depending on the tokenizer.

        Args:
            text: Input text

        Returns:
            Estimated number of tokens
        """
        return len(text.split()) * 3

    def generate_chunks(
        self,
        repo_url: str | None = None,
        strategy: ChunkingStrategy = ChunkingStrategy.FILE,
        min_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        file_types: Optional[List[str]] = None,
        chunk_size: int = 400,
        overlap: int = 50,
    ) -> None:
        """Generate chunks from repository content.

        Args:
            repo_url: URL or path of the repository
            strategy: Chunking strategy to use (FILE, MARKER, or SEMANTIC)
            min_tokens: Minimum number of tokens per chunk
            max_tokens: Maximum number of tokens per chunk
            file_types: List of file extensions to include (e.g. ['.py', '.md'])
            chunk_size: Target size of each chunk in characters
            overlap: Number of characters to overlap between chunks
        """
        if not repo_url:
            # If repo URL is not provided, use the first repository
            repo_url = next(iter(self.repositories.keys()))

        print(f"Generating chunks using {strategy.name} strategy...")
        all_chunks = []
        for content in self.repositories[repo_url]["content"]:
            file_chunks = chunk_file_content(
                file_content=content[1],
                file_path=content[0],
                strategy=strategy,
                chunk_size=chunk_size,
                overlap=overlap,
                max_tokens=max_tokens or 512,
            )
            all_chunks.extend(file_chunks)

        # Apply filters
        all_chunks = filter_chunks(
            all_chunks,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            file_types=file_types,
        )

        print(f"Generated {len(all_chunks)} chunks")

        # Store chunks in repository metadata
        self.repositories[repo_url]["chunks"] = all_chunks

    def summarize_chunks(
        self,
        repo_url: str,
        model: str = "openai/gpt-4o-mini",
        custom_prompt: Optional[str] = None,
        max_tokens: int = 500,
        batch_size: int = 20,
    ) -> None:
        """Generate summaries for repository chunks.

        Args:
            repo_url: URL or path of the repository
            model: The model to use for summarization
            custom_prompt: Optional custom prompt for summarization
            max_tokens: Maximum tokens in each summary
            temperature: Temperature for generation
            batch_size: Size of batches for processing chunks (default: 20)
        """
        repo_chunks = self.repositories[repo_url]["chunks"]
        print(f"Generating summaries for {len(repo_chunks)} chunks...")

        # Get all raw content for batch processing
        raw_contents = [chunk.content_raw for chunk in repo_chunks]

        # Process all chunks in batches
        summaries = summarize_content(
            text_content=raw_contents,
            model=model,
            custom_prompt=custom_prompt,
            max_tokens=max_tokens,
            batch_size=batch_size
        )

        # Update chunks with their summaries
        for chunk, summary in zip(repo_chunks, summaries):
            chunk.content_processed = summary

        print("Summaries generated successfully")

    def generate_embeddings(
        self,
        repo_url: str,
        embedding_from: str = "both",  # "raw", "processed", or "both"
        batch_size: int = 100
    ) -> None:
        """Generate embeddings for repository chunks.

        Args:
            repo_url: URL or path of the repository
            embedding_from: Source for embeddings ("raw", "processed", or "both")
            batch_size: Size of batches for processing embeddings (default: 100)
        """
        repo_chunks = self.repositories[repo_url]["chunks"]

        if embedding_from == "both":
            # Create duplicates of chunks for both raw and processed embeddings
            processed_chunks = []
            for chunk in repo_chunks:
                # Create a duplicate chunk for processed content
                processed_chunk = Chunk(
                    source_file=chunk.source_file,
                    content_raw=chunk.content_raw,
                    content_processed=chunk.content_processed,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    chunk_type=chunk.chunk_type,
                    context=chunk.context
                )
                processed_chunks.append(processed_chunk)

            # Set embedding source and generate embeddings for raw content
            print(f"Generating raw content embeddings for {len(repo_chunks)} chunks...")
            texts = [chunk.content_raw for chunk in repo_chunks]
            embeddings = generate_embeddings(
                texts=texts,
                model=self.embedding_model,
                api_key=self.api_key,
                batch_size=batch_size
            )
            for chunk, embedding in zip(repo_chunks, embeddings):
                chunk.embedding = embedding
                chunk.embedding_from = "content_raw"

            # Set embedding source and generate embeddings for processed content
            print(f"Generating processed content embeddings for {len(processed_chunks)} chunks...")
            texts = [chunk.content_processed for chunk in processed_chunks]
            embeddings = generate_embeddings(
                texts=texts,
                model=self.embedding_model,
                api_key=self.api_key,
                batch_size=batch_size
            )
            for chunk, embedding in zip(processed_chunks, embeddings):
                chunk.embedding = embedding
                chunk.embedding_from = "content_processed"

            # Combine both sets of chunks
            self.repositories[repo_url]["chunks"] = repo_chunks + processed_chunks
            print(f"Total chunks after combining: {len(self.repositories[repo_url]['chunks'])}")

        else:
            # Generate embeddings for either raw or processed content
            print(f"Generating embeddings for {len(repo_chunks)} chunks...")
            texts = [
                chunk.content_raw if embedding_from == "raw" else chunk.content_processed
                for chunk in repo_chunks
            ]
            embeddings = generate_embeddings(
                texts=texts,
                model=self.embedding_model,
                api_key=self.api_key,
                batch_size=batch_size
            )
            for chunk, embedding in zip(repo_chunks, embeddings):
                chunk.embedding = embedding
                chunk.embedding_from = f"content_{embedding_from}"

    def insert_chunks(self, repo_url: str) -> None:
        """Insert chunks and their embeddings into the vector database.

        Args:
            chunks: List of content chunks
            embeddings: List of embedding vectors
            repo_url: Repository URL for metadata
        """
        print("Storing vectors...")
        repo_chunks = self.repositories[repo_url]["chunks"]
        points = [
            PointStruct(
                id=i,
                vector=chunk.embedding,
                payload={
                    "content": chunk.content_raw,
                    "content_processed": chunk.content_processed,
                    "source_file": chunk.source_file,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "chunk_type": chunk.chunk_type,
                    "context": chunk.context,
                    "repository": repo_url,
                    "embedding_model": self.embedding_model,
                    "embedding_from": chunk.embedding_from,
                }
            )
            for i, chunk in enumerate(repo_chunks)
        ]

        self.qdrant.insert_points(
            collection_name=self.collection_name,
            points=points
        )
        print(f"Stored {len(points)} chunks successfully")

    def index_repository(
        self,
        repo_url: str,
        strategy: ChunkingStrategy = ChunkingStrategy.FILE,
        min_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        file_types: Optional[List[str]] = None,
        chunk_size: int = 400,
        overlap: int = 50,
    ) -> None:
        """Index a repository's content.

        This is a convenience method that runs all steps in sequence:
        1. Parse the repository
        2. Generate chunks with optional filtering
        3. Generate embeddings
        4. Store in vector database

        Args:
            repo_url: URL or path of the repository
            strategy: Chunking strategy to use (FILE, MARKER, or SEMANTIC)
            min_tokens: Minimum number of tokens per chunk
            max_tokens: Maximum number of tokens per chunk
            file_types: List of file extensions to include (e.g. ['.py', '.md'])
            chunk_size: Target size of each chunk in characters
            overlap: Number of characters to overlap between chunks
        """
        # Step 1: Parse repo
        if repo_url not in self.repositories:
            self.parse_repo(repo_url)

        # Step 2: Generate chunks with filters
        self.generate_chunks(
            repo_url=repo_url,
            strategy=strategy,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            file_types=file_types,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        # Step 3: Generate embeddings
        self.generate_embeddings(repo_url)

        # Step 4: Store in database
        self.insert_chunks(repo_url)

    def search(
        self,
        query: str,
        limit: int = 5,
        chunk_type: Optional[str] = None,
        file_extension: Optional[str] = None,
        embedding_from: Optional[str] = None,  # "content_raw" or "content_processed"
        batch_size: int = 100,
    ) -> List[dict]:
        """Search for relevant content.

        Args:
            query: Search query
            limit: Maximum number of results to return
            chunk_type: Filter by chunk type (e.g., "code", "documentation")
            file_extension: Filter by file extension
            embedding_from: Filter by embedding source ("content_raw" or "content_processed")
            batch_size: Batch size for embedding generation
        """
        # Get query embedding
        query_embedding = generate_embeddings(
            texts=[query],
            model=self.embedding_model,
            api_key=self.api_key,
            batch_size=batch_size
        )[0]

        # Build filter conditions
        filter_conditions = {}
        if chunk_type:
            filter_conditions["chunk_type"] = chunk_type
        if file_extension:
            filter_conditions["source_file"] = {
                "path": f".*\\.{file_extension}$"
            }
        if embedding_from:
            filter_conditions["embedding_from"] = embedding_from

        # Search using manager
        results = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit,
            filter_conditions=filter_conditions
        )

        # Convert SearchResult objects to dicts for backward compatibility
        return [
            {
                "score": result.score,
                "content": result.content,
                "file": result.file,
                "type": result.type,
                "context": result.context,
                "lines": result.lines,
            }
            for result in results
        ]
