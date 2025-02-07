"""Main indexing and search functionality."""

import os
import logging
from datetime import datetime
from dataclasses import asdict
from typing import Dict, List, Optional, Any, Literal

from gitingest import ingest
from qdrant_client.models import PointStruct
from pathlib import Path

from git2rag.utils.file import get_file_type
from git2rag.chunking import (
    ChunkingStrategy,
    Chunk,
    chunk_file_content,
    filter_chunks,
)
from git2rag.content_parser import break_into_files
from git2rag.clients import QdrantManager
from git2rag.embeddings import generate_embeddings
from git2rag.summarizer import summarize_content


class RepoIndexer:
    """Index and search repository content."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        log_level: int = logging.INFO,
    ):
        """Initialize the indexer.

        Args:
            api_key: API key for the embedding provider
            log_level: Logging level (default: logging.INFO)
        """
        # Setup logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        # Add handler if none exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.repositories: Dict[str, Dict[str, Any]] = {}
        self.qdrant_manager: Optional[QdrantManager] = None
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY", None)
            self.api_key = api_key
            if not api_key:
                self.logger.warning("API key not provided and OPENAI_API_KEY not set.")

    def parse_repo(
        self,
        repo_url: str,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ) -> None:
        """Parse a repository and store its raw content.

        Args:
            repo_url: URL or path of the repository
            include_patterns: Optional list of file patterns to include
            exclude_patterns: Optional list of file patterns to exclude
        """
        self.logger.info(f"Parsing repository: {repo_url}")
        try:
            summary, tree, full_content = ingest(
                source=repo_url,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
            )
            files_content = break_into_files(full_content)

            # Store metadata
            self.repositories[repo_url] = {
                "summary": summary,
                "tree": tree,
                "content": files_content,
                "chunks": [],
            }
            self.logger.info(f"Successfully parsed repository with {len(files_content)} files")
        except Exception as e:
            self.logger.error(f"Failed to parse repository: {str(e)}", exc_info=True)
            raise RuntimeError(f"Repository parsing failed: {str(e)}")

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

    def _process_file_chunks(
        self,
        content: tuple,
        strategy: Dict[str, List[ChunkingStrategy]],
        chunk_size: int,
        overlap: int,
    ) -> List[Chunk]:
        """Process a single file and generate chunks using specified strategies.

        Args:
            content: Tuple of (file_path, file_content)
            strategy: Dictionary mapping file types to chunking strategies
            chunk_size: Target size of each chunk in characters
            overlap: Number of characters to overlap between chunks

        Returns:
            List of generated chunks
        """
        file_path, file_content = content
        file_type = get_file_type(file_path=file_path)
        file_chunks = []

        for st in strategy[file_type]:
            self.logger.debug(f"Chunking file: {file_path} with strategy: {st.name}")
            chunks = chunk_file_content(
                file_content=file_content,
                file_path=file_path,
                strategy=st,
                chunk_size=chunk_size,
                overlap=overlap,
            )
            file_chunks.extend(chunks)

        return file_chunks

    def generate_chunks(
        self,
        repo_url: Optional[str] = None,
        strategy: Dict[str, List[ChunkingStrategy]] = None,
        min_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        file_types: Optional[List[str]] = None,
        chunk_size: int = 400,
        overlap: int = 50,
        max_workers: int = 10,
    ) -> None:
        """Generate chunks from repository content using parallel processing.

        Args:
            repo_url: URL or path of the repository
            strategy: dict
            min_tokens: Minimum number of tokens per chunk
            max_tokens: Maximum number of tokens per chunk
            file_types: List of file extensions to include (e.g. ['.py', '.md'])
            chunk_size: Target size of each chunk in characters
            overlap: Number of characters to overlap between chunks
            max_workers: Maximum number of worker threads (default: 4)
        """
        import concurrent.futures

        if not repo_url:
            # If repo URL is not provided, use the first repository
            repo_url = next(iter(self.repositories.keys()))

        if not strategy:
            strategy = {
                "code": [ChunkingStrategy.FILE, ChunkingStrategy.SEMANTIC],
                "docs": [ChunkingStrategy.FILE, ChunkingStrategy.SEMANTIC],
            }

        n_files = len(self.repositories[repo_url]["content"])
        self.logger.info(f"Generating chunks for {n_files} files using {max_workers} workers.")

        all_chunks = []
        try:
            # Process files in parallel using ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Create tasks for each file
                future_to_file = {
                    executor.submit(
                        self._process_file_chunks,
                        content,
                        strategy,
                        chunk_size,
                        overlap,
                    ): content[0]
                    for content in self.repositories[repo_url]["content"]
                }

                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        file_chunks = future.result()
                        all_chunks.extend(file_chunks)
                        self.logger.debug(f"Successfully processed file: {file_path}")
                    except Exception as e:
                        self.logger.error(f"Failed to process file {file_path}: {str(e)}")
                        raise

            # Apply filters
            self.logger.debug("Applying chunk filters")
            all_chunks = filter_chunks(
                all_chunks,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
                file_types=file_types,
            )

            self.logger.info(f"Generated {len(all_chunks)} chunks")

            # Store chunks in repository metadata
            self.repositories[repo_url]["chunks"] = all_chunks
        except Exception as e:
            self.logger.error(f"Failed to generate chunks: {str(e)}", exc_info=True)
            raise RuntimeError(f"Chunk generation failed: {str(e)}")

    def summarize_chunks(
        self,
        repo_url: Optional[str] = None,
        model: str = "openai/o3-mini",
        custom_prompt: Optional[str] = None,
        max_tokens: int = 100,
        batch_size: int = 30,
    ) -> None:
        """Generate summaries for repository chunks.

        Args:
            repo_url: URL or path of the repository
            model: The model to use for summarization
            custom_prompt: Optional custom prompt for summarization
            max_tokens: Maximum tokens in each summary
            batch_size: Size of batches for processing chunks (default: 20)
        """
        if not repo_url:
            # If repo URL is not provided, use the first repository
            repo_url = next(iter(self.repositories.keys()))

        repo_chunks = self.repositories[repo_url]["chunks"]
        self.logger.info(f"Generating summaries for {len(repo_chunks)} chunks")
        self.logger.debug(f"Using model: {model}, batch_size: {batch_size}")

        try:
            # Get all raw content for batch processing
            raw_contents = [chunk.content_raw for chunk in repo_chunks]

            # Process all chunks in batches
            self.logger.debug("Starting batch processing of summaries")
            summaries = summarize_content(
                text_content=raw_contents,
                model=model,
                custom_prompt=custom_prompt,
                max_tokens=max_tokens,
                batch_size=batch_size,
            )

            # Update chunks with their summaries
            for chunk, summary in zip(repo_chunks, summaries):
                chunk.content_processed = summary

            self.logger.info("Successfully generated summaries for all chunks")
        except Exception as e:
            self.logger.error(f"Failed to generate summaries: {str(e)}", exc_info=True)
            raise RuntimeError(f"Summary generation failed: {str(e)}")

    def generate_embeddings(
        self,
        repo_url: str,
        embedding_from: str = "both",  # "raw", "processed", or "both"
        embedding_model: str = "openai/text-embedding-ada-002",
        batch_size: int = 100,
    ) -> None:
        """Generate embeddings for repository chunks.

        Args:
            repo_url: URL or path of the repository
            embedding_from: Source for embeddings ("raw", "processed", or "both")
            embedding_model: Model to use for embeddings
            batch_size: Size of batches for processing embeddings (default: 100)
        """
        repo_chunks = self.repositories[repo_url]["chunks"]
        self.logger.info(f"Generating embeddings for {len(repo_chunks)} chunks")
        self.logger.debug(f"Using model: {embedding_model}, batch_size: {batch_size}")

        try:
            if embedding_from in ["raw", "both"]:
                # Generate raw content embeddings
                self.logger.info("Generating raw content embeddings")
                texts = [chunk.content_raw for chunk in repo_chunks]
                try:
                    embeddings = generate_embeddings(
                        texts=texts,
                        model=embedding_model,
                        api_key=self.api_key,
                        batch_size=batch_size,
                    )
                    for chunk, embedding in zip(repo_chunks, embeddings):
                        chunk.embedding_raw = embedding
                    self.logger.info("Successfully generated raw content embeddings")
                except Exception as e:
                    self.logger.error(f"Failed to generate raw content embeddings: {str(e)}", exc_info=True)
                    raise

            if embedding_from in ["processed", "both"]:
                # Generate processed content embeddings
                self.logger.info("Processing content embeddings")
                # Only include chunks that have processed content
                chunks_with_processed = [c for c in repo_chunks if c.content_processed is not None]
                if not chunks_with_processed:
                    self.logger.warning("No processed content found, skipping processed embeddings")
                    return

                self.logger.info(f"Generating processed content embeddings for {len(chunks_with_processed)} chunks")
                texts = [chunk.content_processed for chunk in chunks_with_processed]
                try:
                    embeddings = generate_embeddings(
                        texts=texts,
                        model=embedding_model,
                        api_key=self.api_key,
                        batch_size=batch_size,
                    )
                    for chunk, embedding in zip(chunks_with_processed, embeddings):
                        chunk.embedding_processed = embedding
                    self.logger.info("Successfully generated processed content embeddings")
                except Exception as e:
                    self.logger.error(f"Failed to generate processed content embeddings: {str(e)}", exc_info=True)
                    raise

        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {str(e)}", exc_info=True)
            raise RuntimeError(f"Embedding generation failed: {str(e)}")

    def intialize_qdrant(
        self,
        qdrant_url: str,
        collection_name: str,
        collection_vector_size: int = 1536,
        qdrant_api_key: Optional[str] = None,
        batch_size: int = 100,
    ) -> None:
        """Initialize the Qdrant manager and create a collection.

        Args:
            qdrant_url: URL of the Qdrant server
            collection_name: Name of the Qdrant collection
            collection_vector_size: Size of the vectors in the collection
            qdrant_api_key: API key for the Qdrant server
            batch_size: Batch size for Qdrant operations
        """
        if not self.qdrant_manager:
            self.logger.info("Initializing Qdrant manager")
            try:
                # Initialize Qdrant manager
                self.qdrant_manager = QdrantManager(
                    url=qdrant_url,
                    api_key=qdrant_api_key,
                    batch_size=batch_size,
                )
                # Ensure collection exists
                self.qdrant_manager.create_collection(
                    name=collection_name,
                    vector_size=collection_vector_size,
                )
                self.logger.info(f"Successfully initialized Qdrant collection: {collection_name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize Qdrant: {str(e)}", exc_info=True)
                raise RuntimeError(f"Qdrant initialization failed: {str(e)}")

    def insert_chunks(
        self,
        repo_url: str,
        qdrant_url: str,
        qdrant_collection_name: str,
        qdrant_api_key: Optional[str] = None,
    ) -> None:
        """Insert chunks and their embeddings into the vector database.

        Args:
            repo_url: Repository URL for metadata
            qdrant_url: URL of the Qdrant server
            qdrant_collection_name: Name of the Qdrant collection
            qdrant_api_key: Optional API key for Qdrant server
        """
        try:
            self.intialize_qdrant(
                qdrant_url=qdrant_url,
                collection_name=qdrant_collection_name,
                qdrant_api_key=qdrant_api_key,
            )

            self.logger.info("Preparing vectors for storage")
            repo_chunks = self.repositories[repo_url]["chunks"]
            points = []
            for i, chunk in enumerate(repo_chunks):
                vectors = {}
                if chunk.embedding_raw is not None:
                    vectors["raw"] = chunk.embedding_raw
                if chunk.embedding_processed is not None:
                    vectors["processed"] = chunk.embedding_processed

                if vectors:  # Only add points that have at least one vector
                    points.append(
                        {
                            "id": i,
                            "vectors": vectors,
                            "payload": {
                                "content": chunk.content_raw,
                                "content_processed": chunk.content_processed,
                                "source_file": chunk.source_file,
                                "start_line": chunk.start_line,
                                "end_line": chunk.end_line,
                                "chunk_type": chunk.chunk_type,
                                "context": chunk.context,
                                "repository": repo_url,
                                "embedding_model": self.embedding_model,
                            },
                        }
                    )

            self.logger.info(f"Storing {len(points)} vectors in Qdrant")
            self.qdrant_manager.insert_points(
                collection_name=qdrant_collection_name,
                points=points,
            )
            self.logger.info(f"Successfully stored {len(points)} chunks in Qdrant")
        except Exception as e:
            self.logger.error(f"Failed to store vectors: {str(e)}", exc_info=True)
            raise RuntimeError(f"Vector storage failed: {str(e)}")

    def index_repository(
        self,
        # Parsing parameters
        repo_url: str,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        # Chunking parameters
        strategy: ChunkingStrategy = ChunkingStrategy.FILE,
        min_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        file_types: Optional[List[str]] = None,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
        # Summarization parameters
        summarize: bool = True,
        summary_model: str = "openai/o3-mini",
        summary_prompt: Optional[str] = None,
        summary_max_tokens: int = 100,
        summary_batch_size: int = 20,
        # Embedding parameters
        embedding_from: str = "both",  # "raw", "processed", or "both"
        embedding_batch_size: int = 100,
    ) -> Dict[str, Any]:
        """Index a repository's content with enhanced error handling and logging.

        This is a convenience method that runs all steps in sequence:
        1. Parse the repository
        2. Generate chunks with optional filtering
        3. Generate summaries (optional)
        4. Generate embeddings (raw and/or processed)
        5. Store in vector database

        Args:
            repo_url: URL or path of the repository
            strategy: Chunking strategy to use (FILE, MARKER, or SEMANTIC)
            min_tokens: Minimum number of tokens per chunk
            max_tokens: Maximum number of tokens per chunk
            file_types: List of file extensions to include (e.g. ['.py', '.md'])
            chunk_size: Target size of each chunk in characters
            overlap: Number of characters to overlap between chunks
            summarize: Whether to generate summaries for chunks
            summary_model: Model to use for summarization
            summary_prompt: Custom prompt for summarization
            summary_max_tokens: Maximum tokens in summaries
            summary_batch_size: Batch size for summarization
            embedding_from: Which content to embed ("raw", "processed", or "both")
            embedding_batch_size: Batch size for embedding generation

        Returns:
            Dict containing indexing statistics and metrics
        """
        # Initialize metrics
        metrics = {
            "start_time": datetime.now(),
            "steps": {},
            "errors": [],
            "total_chunks": 0,
            "processed_chunks": 0,
            "total_files": 0,
            "processed_files": 0
        }

        self.logger.info(f"Starting indexing for repository: {repo_url}")
        self.logger.debug(f"Configuration: strategy={strategy}, summarize={summarize}")

        try:
            # Step 1: Parse repo
            self.logger.info("Step 1/5: Parsing repository")
            metrics["steps"]["parsing"] = {"start_time": datetime.now()}
            if repo_url not in self.repositories:
                self.parse_repo(
                    repo_url=repo_url,
                    include_patterns=include_patterns,
                    exclude_patterns=exclude_patterns,
                )
            metrics["steps"]["parsing"]["end_time"] = datetime.now()
            metrics["total_files"] = len(self.repositories[repo_url]["content"])
            self.logger.info(f"Parsed {metrics['total_files']} files")

            # Step 2: Generate chunks with filters
            self.logger.info("Step 2/5: Generating chunks")
            metrics["steps"]["chunking"] = {"start_time": datetime.now()}
            try:
                self.generate_chunks(
                    repo_url=repo_url,
                    strategy=strategy,
                    min_tokens=min_tokens,
                    max_tokens=max_tokens,
                    file_types=file_types,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )
            except Exception as e:
                self.logger.error(f"Chunk generation failed: {str(e)}", exc_info=True)
                metrics["errors"].append({"step": "chunking", "error": str(e)})
                raise RuntimeError(f"Chunk generation failed: {str(e)}")
            metrics["steps"]["chunking"]["end_time"] = datetime.now()
            metrics["total_chunks"] = len(self.repositories[repo_url]["chunks"])
            self.logger.info(f"Generated {metrics['total_chunks']} chunks")

            # Step 3: Generate summaries if requested
            if summarize:
                self.logger.info("Step 3/5: Generating summaries")
                metrics["steps"]["summarization"] = {"start_time": datetime.now()}
                try:
                    self.summarize_chunks(
                        repo_url=repo_url,
                        model=summary_model,
                        custom_prompt=summary_prompt,
                        max_tokens=summary_max_tokens,
                        batch_size=summary_batch_size,
                    )
                except Exception as e:
                    self.logger.warning(f"Summarization failed: {str(e)}", exc_info=True)
                    metrics["errors"].append({"step": "summarization", "error": str(e)})
                metrics["steps"]["summarization"]["end_time"] = datetime.now()

            # Step 4: Generate embeddings
            self.logger.info("Step 4/5: Generating embeddings")
            metrics["steps"]["embeddings"] = {"start_time": datetime.now()}
            try:
                self.generate_embeddings(
                    repo_url=repo_url,
                    embedding_from=embedding_from,
                    batch_size=embedding_batch_size,
                )
            except Exception as e:
                self.logger.error(f"Embedding generation failed: {str(e)}", exc_info=True)
                metrics["errors"].append({"step": "embeddings", "error": str(e)})
                raise RuntimeError(f"Embedding generation failed: {str(e)}")
            metrics["steps"]["embeddings"]["end_time"] = datetime.now()

            # Step 5: Store in database
            self.logger.info("Step 5/5: Storing in database")
            metrics["steps"]["storage"] = {"start_time": datetime.now()}
            try:
                self.insert_chunks(repo_url)
            except Exception as e:
                self.logger.error(f"Vector storage failed: {str(e)}", exc_info=True)
                metrics["errors"].append({"step": "storage", "error": str(e)})
                raise RuntimeError(f"Vector storage failed: {str(e)}")
            metrics["steps"]["storage"]["end_time"] = datetime.now()

        except Exception as e:
            metrics["end_time"] = datetime.now()
            metrics["status"] = "failed"
            metrics["error"] = str(e)
            self.logger.error(f"Indexing failed: {str(e)}", exc_info=True)
            raise

        metrics["end_time"] = datetime.now()
        metrics["status"] = "completed"
        metrics["duration"] = (metrics["end_time"] - metrics["start_time"]).total_seconds()

        # Log final summary
        self.logger.info("\nIndexing Summary:")
        self.logger.info(f"Status: {metrics['status']}")
        self.logger.info(f"Total Duration: {metrics['duration']:.2f} seconds")
        self.logger.info(f"Files Processed: {metrics['total_files']}")
        self.logger.info(f"Chunks Generated: {metrics['total_chunks']}")
        if metrics['errors']:
            self.logger.warning("\nWarnings/Errors occurred during indexing:")
            for error in metrics['errors']:
                self.logger.warning(f"- {error['step']}: {error['error']}")

        return metrics

    def search(
        self,
        query: str,
        limit: int = 5,
        chunk_type: Optional[str] = None,
        file_extension: Optional[str] = None,
        vector_name: str = "raw",  # "raw" or "processed"
        batch_size: int = 100,
    ) -> List[dict]:
        """Search for relevant content.

        Args:
            query: Search query
            limit: Maximum number of results to return
            chunk_type: Filter by chunk type (e.g., "code", "documentation")
            file_extension: Filter by file extension
            vector_name: Which vector to search against ("raw" or "processed")
            batch_size: Batch size for embedding generation

        Returns:
            List of dictionaries containing search results with scores and metadata
        """
        self.logger.info(f"Searching with query: {query}")
        self.logger.debug(f"Search parameters: limit={limit}, vector_name={vector_name}")

        try:
            # Get query embedding
            self.logger.debug("Generating query embedding")
            try:
                query_embedding = generate_embeddings(
                    texts=[query],
                    model=self.embedding_model,
                    api_key=self.api_key,
                    batch_size=batch_size
                )[0]
            except Exception as e:
                self.logger.error(f"Failed to generate query embedding: {str(e)}", exc_info=True)
                raise RuntimeError(f"Query embedding generation failed: {str(e)}")

            # Build filter conditions
            filter_conditions = {}
            if chunk_type:
                filter_conditions["chunk_type"] = chunk_type
            if file_extension:
                filter_conditions["source_file"] = {"path": f".*\\.{file_extension}$"}

            if filter_conditions:
                self.logger.debug(f"Applying filters: {filter_conditions}")

            # Search using manager
            self.logger.debug("Executing vector search")
            try:
                results = self.qdrant_manager.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    vector_name=vector_name,
                    limit=limit,
                    filter_conditions=filter_conditions,
                )
            except Exception as e:
                self.logger.error(f"Vector search failed: {str(e)}", exc_info=True)
                raise RuntimeError(f"Vector search failed: {str(e)}")

            # Convert SearchResult objects to dicts
            formatted_results = [
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

            self.logger.info(f"Found {len(formatted_results)} results")
            return formatted_results

        except Exception as e:
            self.logger.error(f"Search operation failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Search failed: {str(e)}")
